import argparse
import os
import torch
import numpy as np
from utils import merge_with_mask, find_last_checkpoint
from plot import plot_ins_sem_labels
from unet import UNet
from dataset import TestDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import ctypes
import time
import cv2
from sklearn.metrics import silhouette_score

# global variables
MAX_IMGS = 20
D = 384
NPROC_CLUSTER = 25
NPROC_TILE = 10
SIZE_MIN, SIZE_MAX = 10, 3000
CC_SCALE = 4
SCRATCH = "../tmp"
BICO_FOLDER = "../bico"

# shared memory arrays
sh_sem_preds, sh_ins_preds, sh_offs = [], [], []
# queues for synchronizing processes
sh_mem_q = mp.Queue()
cluster_q = mp.Queue()
res_q = mp.Queue()
tile_q = mp.Queue()
cluster_res_qs = [mp.Queue() for _ in range(NPROC_TILE)]

def is_small_on_border(cc_nb, cc_img):
    on_border = np.unique(np.concatenate([np.unique(cc_img[:,0]), np.unique(cc_img[0,:]), np.unique(cc_img[:,-1]), np.unique(cc_img[-1,:])]))
    on_border = on_border[on_border != 0]
    ind = np.zeros((cc_nb), dtype=bool)
    res = np.zeros(cc_img.shape, dtype=bool)
    for i in on_border:
        res[cc_img == i] = True
        ind[i] = True

    for i in range(1, cc_nb):
        is_i = cc_img == i
        if np.sum(is_i) < SIZE_MIN:
            res[cc_img == i] = True
            ind[i] = True

    return ind, res


def mk_subdir(dir, fn):
    t = fn.split("/")
    for i, sub_dir in enumerate(t[:-1]):
        dir = os.path.join(dir, sub_dir)
        if not os.path.exists(dir):
            os.mkdir(dir)
    return dir, t[-1]


def cluster_proc(done):
    print("cluster proc started", flush=True)
    while True:
        (i, n_obj, n, cc_fl, emb_fl, cs_fl, out_fl, npoints, d, tile_nproc) = cluster_q.get()
        if i == -1:
            done.set()
            print("cluster proc finished", flush=True)
            return
        else:
            bico_call = os.path.join(BICO_FOLDER, "cluster")
            bico_call += " \"%s\" \"%s\" %i %i %i \"%s\" 5 &> /dev/null" % (cs_fl, emb_fl, n_obj, d, npoints, out_fl)
            os.system(bico_call)
            emb = np.loadtxt(emb_fl, dtype=np.float32, delimiter=',')
            labs = np.loadtxt(out_fl, dtype=int)
            s = silhouette_score(emb, labs)
            cluster_res_qs[tile_nproc].put((n, n_obj, s))


def pick_cluster(ranks, i, is_fp):
    # pick the best cluster based on the collected statistics
    n_objs = ranks[:,0]
    r = ranks[:,1]
    i_max = np.where(r == np.max(r))[0][0]
    n_obj = int(n_objs[i_max])

    instance_mask = np.zeros_like(is_fp, dtype=int)

    out_fl = os.path.join(SCRATCH, "%i_%i.txt" % (i, n_obj))
    labs = np.loadtxt(out_fl, dtype=int)
    instance_mask[is_fp] += labs

    return instance_mask, n_obj


def tile_proc(done, proc_nb, d):
# take network output, do all the clusterings, pick the best one
    print("TILE proc started", flush=True)
    while True:
        i = tile_q.get()
        if i == -1:
            res_q.put(-2)
            done.set()
            print("TILE proc finished", flush=True)
            return
        else:
            sem_mask = np.copy(sh_sem_preds[i][1])

            ins_pred = np.copy(sh_ins_preds[i][1])

            is_fp = sem_mask == 1
            cc_nb, cc_img = cv2.connectedComponents(is_fp.astype(np.uint8))
            ind_sb, sb = is_small_on_border(cc_nb, cc_img)
            sem_mask[sb] = 0
            n_fg = np.sum(sem_mask == 1)

            if n_fg < SIZE_MIN:
                np.copyto(sh_ins_preds[i][1][0, :, :], np.zeros((D, D)))
                res_q.put(i)
            else:
                inds = np.arange(0, cc_nb)[np.logical_not(ind_sb)]
                cc_img[sb] = 0
                for j, ind in enumerate(inds):
                    cc_img[cc_img == ind] = j
                cc_nb = np.size(inds) - 1

                if cc_nb == 1:
                    np.copyto(sh_ins_preds[i][1][0, :, :], cc_img)
                    res_q.put(i)
                else:
                    is_fp = sem_mask == 1
                    ccs = cc_img[is_fp]

                    one_hot_ccs = np.eye(cc_nb)[ccs - 1] * CC_SCALE
                    embeddings = ins_pred[:, is_fp].transpose(1, 0)
                    embeddings = np.append(one_hot_ccs, embeddings, axis=1)

                    cc_fl = os.path.join(SCRATCH, "%i_cc.txt" % i)
                    np.savetxt(cc_fl, ccs, fmt="%i", delimiter=",")

                    n = embeddings.shape[0]
                    emb_fl = os.path.join(SCRATCH, "%i.txt" % i)
                    np.savetxt(emb_fl, embeddings, fmt="%.4f", delimiter=",")

                    cs_fl = os.path.join(SCRATCH, "%i_coreset.txt" % i)
                    n_mean = cc_nb
                    n_min, n_max = max(2, n_mean - 2), n_mean + 2

                    bico_call = os.path.join(BICO_FOLDER, "BICO_Quickstart")
                    bico_call += " \"%s\" %i %i %i %i \"%s\" 10 , &>/dev/null" % (emb_fl, n, n_mean, d + cc_nb, n // 2, cs_fl)
                    os.system(bico_call)
                    while len(open(cs_fl).readlines()) <= n_max:
                        os.system(bico_call)

                    for n_obj in range(n_min, n_max + 1):
                        out_fl = os.path.join(SCRATCH, "%i_%i.txt" % (i, n_obj))
                        cluster_q.put((i, n_obj, n_max - n_min + 1, cc_fl, emb_fl, cs_fl, out_fl, n, d + cc_nb, proc_nb))

                    ranks = np.zeros((n_max - n_min + 1, 2))
                    for n_obj in range(n_min, n_max + 1):
                        (n, n_obj, silhouette) = cluster_res_qs[proc_nb].get()
                        ranks[n_obj - n_min, :] = [n_obj, silhouette]
                    pred_inst, n_pred = pick_cluster(ranks, i, is_fp)

                    ### save to output shared array
                    sh_ins_preds[i][0].acquire()
                    np.copyto(sh_ins_preds[i][1][0, :, :], pred_inst)
                    sh_ins_preds[i][0].release()

                    res_q.put(i)

                    os.remove(cs_fl)
                    os.remove(cc_fl)
                    os.remove(emb_fl)
                    for n_obj in range(n_min, n_max + 1):
                        os.remove(os.path.join(SCRATCH, "%i_%i.txt" % (i, n_obj)))


def collector_proc(out_dir, dataset, done, n_gpus):
    """
    This is the major process that collects clustering results from res_q and chooses the best clustering and saves the results.
    :param out_dir:
    :param dataset:
    :param done:
    :param n_gpus:
    :return:
    """
    print("collector proc started", flush=True)

    res = {}
    finished_gpus = 0
    finished_tile_proc = 0
    while True:
        (i) = res_q.get()
        if i == -1:
            # a major gpu process finished
            finished_gpus += 1
            if finished_gpus == n_gpus:
                # if all gpu processes finished, send a finish signal to the clustering processes
                for _ in range(NPROC_TILE):
                    tile_q.put(-1)
        elif i == -2:
            # tile process finished
            finished_tile_proc += 1
            if finished_tile_proc == NPROC_TILE:
                for _ in range(NPROC_CLUSTER):
                    cluster_q.put((-1, 0, 0, "", "", "", "", 0, 0, 0))
                # finish all the clustering processes then exit
                done.set()
                print("collector proc finished", flush=True)
                return
        else:
            pred_sem = np.copy(sh_sem_preds[i][1])
            pred_inst = np.copy(sh_ins_preds[i][1][0,:,:])
            offs = np.copy(sh_offs[i][1])
            img_nb, x ,y, d_img = tuple(offs)

            n_pred = np.max(pred_inst)
            print("%i %i-%i (%i) -- %i clusters" % (img_nb, x, y, i, n_pred), flush=True)

            # put back this i index into the shared memory queue -
            # this means this shared memory can be used for next predictions coming out of GPU
            sh_mem_q.put(i)

            # res dictionary combines predictions of individual tiles into full images
            if not (img_nb in res):
                res[img_nb] = (np.zeros((2, d_img, d_img), dtype=int), 0)
            mask_img, count = res[img_nb]
            mask_img = merge_with_mask(mask_img, pred_inst, pred_sem, x, y)
            count += 1

            # if we have all the tiles for this image, save results to files
            if count == dataset.n_per_img(img_nb):
                fn = dataset.img_fls[img_nb]
                mask_inst = mask_img[0]
                mask_sem = mask_img[1]
                print("saving %s" % fn, flush=True)
                sub_out_dir, fn_short = mk_subdir(out_dir, fn)
                np.save(os.path.join(sub_out_dir, "%s_pred.npy" % fn_short[:-4]), mask_img)
                print("plotting %s" % fn, flush=True)
                plot_ins_sem_labels(dataset.read_file(fn), mask_inst, mask_sem, np.max(mask_inst), os.path.join(sub_out_dir, "%s_pred.png" % fn_short[:-4]))
                del res[img_nb]
            else:
                res[img_nb] = (mask_img, count)




def dispatch(sem_preds, ins_preds, offs):
    # save predictions coming from the GPU to a free shared memory slot
    # then put information in the cluster_q for the clustering processes to cluster these results
    for j in range(sem_preds.shape[0]):
        sem_pred = sem_preds[j]
        sem_pred = np.argmax(sem_pred, axis=0)
        ins_pred = ins_preds[j]
        off = offs[j]

        i = sh_mem_q.get()

        # copy to shared memory
        sh_sem_preds[i][0].acquire()
        np.copyto(sh_sem_preds[i][1][:,:], sem_pred)
        sh_sem_preds[i][0].release()
        sh_ins_preds[i][0].acquire()
        np.copyto(sh_ins_preds[i][1][:,:,:], ins_pred)
        sh_ins_preds[i][0].release()
        sh_offs[i][0].acquire()
        np.copyto(sh_offs[i][1][:], off)
        sh_offs[i][0].release()

        tile_q.put(i)



def run_pred(gpu, img_dir, n_gpus, n_dims, batch_size, load, dir_checkpoint, dones):
    torch.manual_seed(0)
    n_class = 3
    net = UNet(n_channels=1, n_classes=n_class, n_dim=n_dims, bilinear=True)

    if args.gpus == 0:
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(gpu)
        device = torch.device('cuda:%i' % gpu)

        dist.init_process_group(
            backend='nccl',
            world_size=args.gpus,
            rank=gpu
        )
    net.to(device)

    net.eval()

    if load < 0:
        load = find_last_checkpoint(dir_checkpoint)

    fl = os.path.join(dir_checkpoint, 'cp_%i.pth' % load)
    if n_gpus == 0:
        net.load_state_dict(torch.load(fl))
        print('Model loaded from %s' % (fl), flush=True)
    else:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        net.load_state_dict(torch.load(fl, map_location=map_location))
        print('(%i) Model loaded from %s' % (gpu, fl), flush=True)

    dataset = TestDataset(img_dir)
    if n_gpus == 0:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=n_gpus, rank=gpu)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=sampler)

    print("Starting inference:\n" 
        "\tBatch size:      %i\n"
        "\tImages:          %i\n"
        "\tSize:            %i\n"
        "\tDevice:          %s" % (batch_size, dataset.n_imgs(), len(dataset), device), flush=True
    )

    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            imgs = batch['image']
            offs = batch['offs']

            imgs = imgs.to(device)
            sem_predict, ins_predict = net(imgs)
            sem_preds = F.softmax(sem_predict, dim=1)

            dispatch(sem_preds.cpu().data.numpy(), ins_predict.cpu().data.numpy(), offs.cpu().data.numpy())

    res_q.put(-1)
    for done in dones:
        done.wait()



def get_args():
    parser = argparse.ArgumentParser(description='Infer instances',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dims', type=int, default=16,
                        help='Dimensionality of representations', dest='n_dims')
    parser.add_argument('-i', '--img_dir', type=str, default="../sample_images",
                        help='Folder with images to process', dest='img_dir')
    parser.add_argument('-o', '--out_dir', type=str, default="../amap_res",
                        help='Output folder', dest='out_dir')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-f', '--load', dest='load', type=int, default=-1,
                        help='Load model from a checkpoint, -1: find the highest number checkpoint in the checkpoint folder')
    parser.add_argument('-dc', '--dir_checkpoint', dest='dir_checkpoint', type=str, default="../checkpoints",
                        help='Path to the checkpoints folder')
    parser.add_argument('-g', '--gpus', dest='gpus', type=int, default=0, help='Number of GPUs, 0 results in running prediction on CPU')
    return parser.parse_args()


def create_arrays(dim):
    """
    Create shared memory for instance predictions, semantic predictions and respective offsets.
    The content of these shared arrays will be used for clustering.
    """
    global sh_sem_preds, sh_ins_preds, sh_offs

    print("creating arrays..", flush=True)
    SH_SEM = [D, D]
    print("1..", flush=True)
    sh_sem_preds = [ mp.Array(ctypes.c_float, SH_SEM[0]*SH_SEM[1]) for _ in range(MAX_IMGS) ]
    print("2..", flush=True)
    sh_sem_preds = [(sharr, np.frombuffer(sharr.get_obj(), dtype=np.float32, count=SH_SEM[0]*SH_SEM[1]).reshape(SH_SEM)) for sharr in sh_sem_preds]

    SH_INS = [dim, D, D]
    print("3..", flush=True)
    sh_ins_preds = [ mp.Array(ctypes.c_float, SH_INS[0]*SH_INS[1]*SH_INS[2]) for _ in range(MAX_IMGS) ]
    print("4..", flush=True)
    sh_ins_preds = [(sharr, np.frombuffer(sharr.get_obj(), dtype=np.float32, count=SH_INS[0]*SH_INS[1]*SH_INS[2]).reshape(SH_INS)) for sharr in sh_ins_preds]
    print("5..", flush=True)

    sh_offs = [ mp.Array(ctypes.c_float, 4) for _ in range(MAX_IMGS) ]
    print("6..", flush=True)
    sh_offs = [(sharr, np.frombuffer(sharr.get_obj(), dtype=np.int32, count=4).reshape(4)) for sharr in sh_offs]
    print("done..", flush=True)


def start_workers(out_dir, dataset, n_gpus, d):
    """
    Start clustering processes and the process collecting results.
    Return table of flags that when set mean a give process has finished.
    :param out_dir:
    :param dataset:
    :param n_gpus:
    :return:
    """
    dones = []

    print("starting workers..", flush=True)
    for i in range(MAX_IMGS):
        sh_mem_q.put(i)
    done = mp.Event()
    p = mp.Process(target=collector_proc, args=(out_dir, dataset, done, n_gpus))
    dones.append(done)
    p.start()
    for j in range(NPROC_CLUSTER):
        done = mp.Event()
        p = mp.Process(target=cluster_proc, args=(done,))
        dones.append(done)
        p.start()
    for j in range(NPROC_TILE):
        done = mp.Event()
        p = mp.Process(target=tile_proc, args=(done,j,d))
        dones.append(done)
        p.start()
    return dones


if __name__ == '__main__':
    time0 = time.time()
    args = get_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
        print('Created output directory', flush=True)

    if not os.path.exists(SCRATCH):
        os.mkdir(SCRATCH)

    create_arrays(args.n_dims)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    dataset = TestDataset(args.img_dir)
    dones = start_workers(args.out_dir, dataset, args.gpus, args.n_dims)

    gpu_p = []
    if args.gpus == 0:
        p = mp.Process(target=run_pred, args=(0, args.img_dir, args.gpus, args.n_dims, args.batch_size, args.load, args.dir_checkpoint, dones))
        p.start()
        gpu_p.append(p)
    else:
        for gpu in range(args.gpus):
            p = mp.Process(target=run_pred, args=(gpu, args.img_dir, args.gpus, args.n_dims, args.batch_size, args.load, args.dir_checkpoint, dones))
            p.start()
            gpu_p.append(p)

    for p in gpu_p:
        p.join()

    td = time.time() - time0
    hours, remainder = divmod(td, 3600)
    minutes, seconds = divmod(remainder, 60)
    print('finished in: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)), flush=True)

