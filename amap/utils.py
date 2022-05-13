import numpy as np
import re, os
from plot import plot_ins_sem_labels
from scipy.ndimage.measurements import label
#from sklearn.cluster import MeanShift, MiniBatchKMeans
#from multiprocessing import Pool, Queue
#from functools import partial
import tifffile
import cv2
from skimage.morphology import skeletonize


TARGET_RES = 0.022724609375

################ CHECKPOINT #########################

def find_last_checkpoint(dir_checkpoint):
    fls = list(filter(lambda x: re.match(r'cp_([0-9]+).pth', x), os.listdir(dir_checkpoint)))
    if len(fls) > 0:
        nbs = list(map(lambda x: int(re.match(r'cp_([0-9]+).pth', x).group(1)), fls))
        return np.max(nbs)
    else:
        return -1

################ TIF RESOLUTION #########################

def get_resolution(tif_fl, sh):
    tif = tifffile.TiffFile(tif_fl)
    if "XResolution" in tif.pages[0].tags:
        x1, x2 = tif.pages[0].tags["XResolution"].value
        if tif.pages[0].tags["ResolutionUnit"].value == 3: # RESUNIT.CENTIMETER
            x2 = x2 * 10000
        return (x2 / x1) * (tif.pages[0].shape[0] / sh)
    else:
        return TARGET_RES

################ CREATE MASKS FOR NETWORK TRAINING #########################

def thinner_SD(m_sd, m_fp):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # skeletonize SD and dilate by 3 pixels
    skeleton = (skeletonize(m_sd)).astype(np.uint8)
    m_sd = cv2.dilate(skeleton, kernel, iterations=1)
    m_sd[m_sd > 0] = 1

    m_fp = cv2.dilate(m_fp, kernel, iterations=2)
    m_fp[m_sd > 0] = 0
    m_fp[m_fp > 0] = 1
    return m_sd, m_fp


def make_train_files(img_dir="./sample_images", out_dir="./data", nephrin_channel=0, min_pixels=10, tmp_dir="./tmp"):
    fls = list(filter(lambda x: re.match(r'([0-9]+)(.+)_bin.tif', x), os.listdir(img_dir)))
    prefs = list(map(lambda x: re.match(r'(.+)_bin.tif', x).group(1), fls))
    print("Found %i labeled image files" % len(prefs))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    for pref in prefs:
        print("Converting %s ..." % pref)
        fl = os.path.join(img_dir, pref + ".tif")
        img = tifffile.imread(fl)
        if len(img.shape) > 2:
            img = img[nephrin_channel]
        m_sem = cv2.imread(os.path.join(img_dir, "%s_bin.tif" % pref), cv2.IMREAD_GRAYSCALE)
        m_sem[m_sem < 45] = 0
        m_sem[m_sem > 160] = 2
        m_sem[m_sem >= 45] = 1

        res = get_resolution(fl, img.shape[-1])
        scale = round(res / TARGET_RES, 2)

        m_sd = np.zeros_like(m_sem)
        m_sd[m_sem == 2] = 1
        m_fp = np.zeros_like(m_sem)
        m_fp[m_sem == 1] = 1

        if scale != 1:
            new_w, new_h = int(scale * img.shape[1]), int(scale * img.shape[0])
            assert new_w > 0 and new_h > 0, 'Scale is too small'
            img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)
            m_sd = cv2.resize(m_sd, (new_w, new_h), cv2.INTER_NEAREST)
            m_fp = cv2.resize(m_fp, (new_w, new_h), cv2.INTER_NEAREST)

        m_sd, m_fp = thinner_SD(m_sd, m_fp)
        m_fp[m_sd > 0] = 0
        m_sem = m_sd * 2 + m_fp
        img = img.astype(np.float32)
        img = img / np.max(img)

        m = m_sem.copy()
        m[m == 2] = 0

        m, ncomponents = label(m, np.ones((3, 3), dtype=np.int))
        nbs, counts = np.unique(m, return_counts=True)
        rem = False
        for i in range(nbs.size):
            if counts[i] < min_pixels:
                nb = nbs[i]
                nbs[i] = -1
                m_sem[m == nb] = 0
                m[m == nb] = 0
                rem = True
        if rem:
            m_old = m.copy()
            nb = 1
            for nb_old in nbs:
                if nb_old > 0:
                    m[m_old == nb_old] = nb
                    nb += 1

        img_mask = np.concatenate([np.expand_dims(img, axis=0), np.expand_dims(m, axis=0), np.expand_dims(m_sem, axis=0)])
        np.save(os.path.join(out_dir, "%s.npy" % (pref)), img_mask)
        plot_ins_sem_labels(img_mask[:1,:,:], img_mask[1,:,:], img_mask[2,:,:], ncomponents, os.path.join(tmp_dir, "%s_masks.png" % (pref)))

################ TABLE OF TRAINING FILES WITH UNDERREPRESENTED FILES REPLICATED #########################

def make_duplicate_table(dirs):
    with open("../../duplicate.txt", 'w') as f:
        for dir, n in dirs:
            fls = list(filter(lambda x: re.match(r'(.+).npy', x), os.listdir(dir)))
            for fl in fls:
                pref = fl[:-4]
                f.write("%s;%i\n" % (pref, n))


def read_duplicate_table(fl):
    with open(fl, 'r') as f:
        lns = f.readlines()
    res = {}
    for ln in lns:
        t = ln.strip().split(";")
        res[t[0]] = int(t[1])
    return res

################ EXTRACT INST MASKS #########################

def merge_with_mask(mask_img, pred_mask, pred_sem, x, y):
    max_label = np.max(mask_img)
    pred_labels = np.unique(pred_mask)
    pred_labels = pred_labels[pred_labels > 0]
    mask_ins = mask_img[0,x:(x+pred_mask.shape[0]),y:(y+pred_mask.shape[1])]
    mask_sem = mask_img[1,x:(x+pred_mask.shape[0]),y:(y+pred_mask.shape[1])]
    for pred_l in pred_labels:
        is_pred_l = pred_mask == pred_l
        mask_values = mask_ins[is_pred_l]
        mask_ls, counts = np.unique(mask_values, return_counts=True)
        not_zero = mask_ls != 0
        enough_counts = (counts / np.sum(is_pred_l)) > 0.1
        mask_ls = mask_ls[not_zero & enough_counts]
        counts = counts[not_zero & enough_counts]
        if mask_ls.size == 0:
            mask_ins[is_pred_l] = max_label + 1
            max_label += 1
        else:
            mask_l = mask_ls[np.where(counts == np.max(counts))[0][0]]
            mask_ins[is_pred_l] = mask_l
    # merge semantic segm - maximum of class
    mask_sem = np.max(np.append(np.expand_dims(mask_sem, 0), np.expand_dims(pred_sem, 0), axis=0), axis=0)

    mask_img[0,x:(x + pred_mask.shape[0]), y:(y + pred_mask.shape[1])] = mask_ins
    mask_img[1,x:(x + pred_mask.shape[0]), y:(y + pred_mask.shape[1])] = mask_sem
    return mask_img

"""
def gen_instance_mask_ms(sem_pred, ins_pred):
    embeddings = ins_pred[:, sem_pred].transpose(1, 0)
    clustering = MeanShift(bandwidth=2.5, n_jobs=24).fit(embeddings)
    labels = clustering.labels_
    labels_unique = np.unique(labels)
    n_obj = labels_unique.size

    instance_mask = np.zeros_like(sem_pred, dtype=np.int)
    for i in range(n_obj):
        lbl = np.zeros_like(labels, dtype=np.uint8)
        lbl[labels == i] = i + 1
        instance_mask[sem_pred] += lbl

    return instance_mask, n_obj

def is_connected(coords):
    visited = [False] * coords.shape[0]
    check_pixels = [0]

    while len(check_pixels) > 0:
        i = check_pixels.pop()
        visited[i] = True
        x, y = tuple(coords[i,:])
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                wh = np.where((coords[:,0] == x + dx) & (coords[:,1] == y + dy))[0]
                if (wh.size > 0) and not visited[wh[0]]:
                    check_pixels.append(wh[0])
    return np.all(visited)


global embeddings, coords
res_q = Queue()
NPROC = 40


def rank_cluster_p(i, n_objs, size_min, size_max):
    n_obj = n_objs[i]
    #clustering = KMeans(n_obj, n_jobs=20).fit(embeddings)
    clustering = MiniBatchKMeans(n_obj, batch_size=1000).fit(embeddings)
    #clustering = Birch(threshold=0.25, n_clusters=n_obj).fit(embeddings)
    labs = np.unique(clustering.labels_)
    #n_objs = len(labs)
    spatially_conn = np.zeros((n_obj))
    lab_sizes = np.zeros((n_obj))
    for label in labs:
        spatially_conn[label] = is_connected(coords[clustering.labels_ == label,:])
        lab_sizes[label] = np.sum(clustering.labels_ == label)
    conn = np.sum(spatially_conn) / n_obj
    sz = np.sum((lab_sizes > size_min) & (lab_sizes <= size_max)) / n_obj
    size_std = np.std(lab_sizes)
    res_q.put((i, conn, sz, size_std)) #, clustering

def rank_cluster(n_obj, embeddings, coords, size_min, size_max):
    #clustering = KMeans(n_obj, n_jobs=20).fit(embeddings)
    clustering = MiniBatchKMeans(n_obj).fit(embeddings)
    spatially_conn = np.zeros((n_obj))
    lab_sizes = np.zeros((n_obj))
    for label in range(n_obj):
        spatially_conn[label] = is_connected(coords[clustering.labels_ == label,:])
        lab_sizes[label] = np.sum(clustering.labels_ == label)
    conn = np.sum(spatially_conn) / n_obj
    sz = np.sum((lab_sizes > size_min) & (lab_sizes <= size_max)) / n_obj
    size_std = np.std(lab_sizes)
    return (conn, sz, size_std), clustering

def gen_instance_mask(sem_pred, ins_pred):
    size_min, size_max = 10, 3000
    slope, intercept = 0.002158, 5.184565
    instance_mask = np.zeros_like(sem_pred, dtype=np.int)
    n_fg = np.sum(sem_pred)
    n_mean = slope * n_fg + intercept
    n_min, n_max = int(n_mean * 0.7), int(n_mean * 1.3)
    if n_fg < size_min:
        return instance_mask, 0

    #print(os.getenv("OMP_NUM_THREADS"), flush=True)
    #print("%i %i - %i" % (n_mean, n_min, n_max), flush=True)

    global embeddings, coords
    embeddings = ins_pred[:, sem_pred].transpose(1, 0)
    coords = np.indices(sem_pred.shape)
    coords = coords[:, sem_pred].transpose(1,0) # N x 2
    n_objs = list(range(n_min, n_max+1))
    ranks = np.zeros((n_max-n_min+1,3))

    #clusterings = []
    pool = Pool(processes=NPROC)
    pool.map(partial(rank_cluster_p, n_objs=n_objs, size_min=size_min, size_max=size_max), range(len(n_objs)))

    for i in range(len(n_objs)):
        j, conn, size, size_std = res_q.get()
        #(conn, size, size_std), clustering = rank_cluster(n_objs[i], embeddings, coords, size_min=size_min, size_max=size_max)
        ranks[j,:] = [conn, size, size_std]
        #clusterings.append(clustering)
    pool.terminate()
    #pool.join()

    ranks[:, 0] = np.argsort(np.argsort(1 - ranks[:, 0]))
    ranks[:, 1] = np.argsort(np.argsort(-ranks[:, 1]))
    ranks[:, 2] = np.argsort(np.argsort(ranks[:, 2]))
    ranks = np.sum(ranks, axis=1)
    i = np.where(ranks == np.min(ranks))[0][0]
    n_obj = n_objs[i]
    clustering = MiniBatchKMeans(n_obj).fit(embeddings)
    print("       clusters: %i" % n_obj, flush=True)

    for i in range(n_obj):
        lbl = np.zeros_like(clustering.labels_, dtype=np.uint8)
        lbl[clustering.labels_ == i] = i + 1
        instance_mask[sem_pred] += lbl

    return instance_mask, n_obj



################ POSTPROCESS CLEAN UP ####################

def get_coord_conn_comp(pred_inst, ins):
    pos_x, pos_y = np.where(pred_inst == ins)
    coords = np.concatenate([np.expand_dims(pos_x, 1), np.expand_dims(pos_y, 1), np.zeros((pos_x.size, 1))+ins, np.zeros((pos_x.size, 1))], axis=1)
    coords = coords.astype(np.int)
    visited = np.zeros_like(pred_inst)
    check_pixels = []
    curr_label = 0
    for i in range(coords.shape[0]):
        if coords[i,3] == 0:
            curr_label += 1
            check_pixels.append(i)
            while len(check_pixels) > 0:
                i = check_pixels.pop()
                coords[i,3] = curr_label
                x, y = tuple(coords[i,:2])
                visited[x,y] = 1
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if (x+dx >= 0) and (x+dx < pred_inst.shape[0]) and (y+dy >=0) and (y+dy < pred_inst.shape[1]) and\
                                (pred_inst[x+dx, y +dy] == ins) and (visited[x+dx, y +dy] == 0):
                            wh = np.where((coords[:,0] == x+dx) & (coords[:,1] == y+dy))[0]
                            check_pixels.append(wh[0])
    return coords


def remove_on_borders_and_small(coords, sz, min_size):
    ccs, counts = np.unique(coords[:,2:], axis=0, return_counts=True)
    on_border_or_small = np.zeros((coords.shape[0]), dtype=np.bool)
    for i in range(ccs.shape[0]):
        is_cc = (coords[:,2] == ccs[i,0]) & (coords[:,3] == ccs[i,1])
        if counts[i] < min_size:
            on_border_or_small[is_cc] = True
        else:
            coords_cc = coords[is_cc,:]
            on_b = np.any(coords_cc[:,0] == 0) or np.any(coords_cc[:,0] == sz[0]-1) or np.any(coords_cc[:,1] == 0) or np.any(coords_cc[:,1] == sz[1]-1)
            on_border_or_small[is_cc] = on_b
    return coords[np.logical_not(on_border_or_small),:]

def multidim_intersect(arr1_view, arr2):
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr2.dtype).reshape(-1, arr2.shape[1])

def touching_border(coords1, coords2):
    coords1 = coords1.copy()
    coords1_view = coords1.view([('', coords1.dtype)] * coords1.shape[1])
    border = np.zeros((0,2), dtype=np.int32)
    margins = np.array(np.meshgrid([-1,0,1], [-1,0,1])).T.reshape(-1,2)
    for i in range(margins.shape[0]):
        border = np.append(border, multidim_intersect(coords1_view, coords2 + margins[i,:]), axis=0)
    if border.size > 0:
        border = np.unique(border, axis=0)
    return border

def get_center(coords):
    dx = np.abs(np.mean(coords[:, 0]) - coords[:,0])
    ix = np.where(dx == np.min(dx))[0]
    coords1 = coords[ix,:]
    dy = np.abs(np.mean(coords1[:, 1]) - coords1[:, 1])
    iy = np.where(dy == np.min(dy))[0][0]
    #x = int(np.mean(coords[:,0]))
    #y = int(np.mean(coords[coords[:,0] == x,1]))
    return np.array(coords1[iy,:], dtype=int)

def intermediates(p1, p2):
    nb_points = max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)
    return np.array([[p1[0] + i * x_spacing, p1[1] +  i * y_spacing] for i in range(1, nb_points+1)]).astype(np.int)

def not_membrane_separated(coords1, coords2, pred_sem):
    p1 = get_center(coords1)
    p2 = get_center(coords2)
    #print("(%i %i) (%i %i)" % (p1[0], p1[1], p2[0], p2[1]))
    res = True
    if np.sum((p1-p2)**2)**0.5 > 2:
        ps = intermediates(p1, p2)
        res = np.all(pred_sem[ps[:,0], ps[:,1]] != 2)
    return res

def clean_instance_pred(pred_inst, pred_sem,min_size):
    inst = np.unique(pred_inst)
    inst = inst[inst != 0]
    positions = np.zeros((0,4), dtype=np.int32)

    res = np.zeros_like(pred_inst, dtype=np.int)
    labels = 0
    res_sem = pred_sem.copy()

    if len(inst) == 0:
        print(inst)
        return res, res_sem,labels

    for ins in inst:
        coords = get_coord_conn_comp(pred_inst, ins)
        positions = np.append(positions, coords, axis=0)

    # 1. merge touching
    ccs, counts = np.unique(positions[:,2:4], axis=0, return_counts=True)
    connected = np.zeros((ccs.shape[0], ccs.shape[0]), dtype=np.int)
    for i1 in range(ccs.shape[0]-1):
        coords1 = positions[(positions[:,2] == ccs[i1,0]) & (positions[:,3] == ccs[i1,1]),:]
        for i2 in range(i1 +1, ccs.shape[0]):
            coords2 = positions[(positions[:, 2] == ccs[i2, 0]) & (positions[:, 3] == ccs[i2, 1]), :]
            border = touching_border(coords1[:,:2], coords2[:,:2])
            if border.shape[0] > 0:
                #print("%i-%i  %i-%i" % (ccs[i1,0], ccs[i1,1], ccs[i2,0], ccs[i2,1]))
                #print(border)
                connected[i1,i2] = not_membrane_separated(coords1, coords2, pred_sem)
                #print(connected[i1,i2])
    while np.any(connected):
        i1, i2 = np.where(connected > 0)
        i1 = i1[0]
        i2 = i2[0]
        if counts[i2] > counts[i1]:
            j = i2
            i2 = i1
            i1 = j
        #print("joining %i-%i %i-%i" % (ccs[i1,0], ccs[i1,1], ccs[i2,0], ccs[i2,1]))
        positions[(positions[:,2] == ccs[i2,0]) & (positions[:,3] == ccs[i2,1]), 2:4] = ccs[i1]
        counts[i1] += counts[i2]
        counts = np.delete(counts, i2)
        ccs = np.delete(ccs, i2, axis=0)
        connected[i1,:] += connected[i2,:]
        connected[:,i1] += connected[:,i2]
        connected[i1,i1] = 0
        connected = np.delete(connected, i2, axis=0)
        connected = np.delete(connected, i2, axis=1)

    # 2. remove small and those touching borders
    positions = remove_on_borders_and_small(positions, pred_inst.shape, min_size)

    res_sem[res_sem == 1] = 0
    if positions.shape[0] > 0:
        ccs, counts = np.unique(positions[:,2:4], axis=0, return_counts=True)
        labels = ccs.shape[0]+1
        for i in range(ccs.shape[0]):
            coords_cs = positions[(positions[:,2]==ccs[i,0]) & (positions[:,3]==ccs[i,1]),:]
            #print("label %i %i counts" % (i+1, coords_cs.shape[0]))
            res[coords_cs[:,0],coords_cs[:,1]] = i+1
            res_sem[coords_cs[:, 0], coords_cs[:, 1]] = 1
    return res, res_sem, labels

    #3. same label
    # for each cc other than largest
    # if separated by membrane from largest, renumber, otherwise remove
"""


