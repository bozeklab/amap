import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from discriminative import DiscriminativeLoss
from unet import UNet
from utils import find_last_checkpoint, read_duplicate_table
from dataset import InstDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp


def test_step(loader, net, device, epoch, criterion_disc, criterion_ce):
    net.eval()
    disc_losses = []
    ce_losses = []

    for batch in loader:
        imgs = batch['image']
        sem_masks = batch['sem_mask']
        ins_masks = batch['ins_mask']
        n_ins = batch['n_ins']

        imgs = imgs.to(device)
        sem_masks = sem_masks.to(device)
        ins_masks = ins_masks.to(device)
        sem_predict, ins_predict = net(imgs)

        ce_loss = criterion_ce(sem_predict, sem_masks)
        loss = ce_loss
        ce_losses.append(ce_loss.item())

        disc_loss = criterion_disc(ins_predict, ins_masks, n_ins)
        loss += disc_loss
        disc_losses.append(disc_loss.item())

    print('TEST: %i(%s). DLoss: %.4f CELoss %.4f' % (epoch, device, np.mean(disc_losses), np.mean(ce_losses)), flush=True)



def train_net(net, gpu, device, train_loader, val_loader, dir_checkpoint, epoch1, epochs_all, lr, parallel):
    print("Starting training:\n"           
        "\tTrain size:       %i\n"
        "\tClasses:         %i\n"
        "\tEpochs:          %i\n"
        "\tLearning rate:   %.4f\n"
        "\tDevice:          %s" % (len(train_loader), net.n_classes, epochs_all, lr, device), flush=True
    )

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=100, min_lr=0.000001)

    # Loss Function
    criterion_disc = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5, norm=2, gpu=gpu).to(device)
    criterion_ce = nn.CrossEntropyLoss().to(device)

    if parallel:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[gpu], output_device=gpu)

    for epoch in range(epoch1, epochs_all+epoch1):
        net.train()

        disc_losses = []
        ce_losses = []

        for batch_i, batch in enumerate(train_loader):
            imgs = batch['image']
            sem_masks = batch['sem_mask']
            ins_masks = batch['ins_mask']
            n_ins = batch['n_ins']

            imgs = imgs.to(device)
            sem_masks = sem_masks.to(device)
            ins_masks = ins_masks.to(device)

            sem_predict, ins_predict = net(imgs)

            # Cross Entropy Loss
            ce_loss = criterion_ce(sem_predict, sem_masks)
            loss = ce_loss
            ce_losses.append(ce_loss.item())
            # Discriminative Loss
            disc_loss = criterion_disc(ins_predict, ins_masks, n_ins)
            loss += disc_loss
            disc_losses.append(disc_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('%i(%s). DLoss: %.4f CELoss %.4f' % (epoch, device, np.mean(disc_losses), np.mean(ce_losses)), flush=True)

        scheduler.step(loss)

        if gpu <= 0:
            if device == torch.device("cpu"):
                torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'cp_%i.pth' % epoch))
            else:
                torch.save(net.module.state_dict(), os.path.join(dir_checkpoint, 'cp_%i.pth' % epoch))

        if epoch > 0 and epoch % 100 == 0:
            test_step(val_loader, net, device, epoch, criterion_disc, criterion_ce)


def train(gpu, args, n_class = 3):
    torch.manual_seed(0)
    net = UNet(n_channels=1, n_classes=n_class, n_dim=args.n_dims, bilinear=True)
    if args.gpus == 0:
        device = torch.device('cpu')
        gpu = -1
    else:
        torch.cuda.set_device(gpu)
        device = torch.device('cuda:%i' % gpu)

        dist.init_process_group(
            backend='nccl',
            world_size=args.gpus,
            rank=gpu
        )
    net.to(device)

    if args.load < 0:
        args.load = find_last_checkpoint(args.dir_checkpoint)
    if args.load >= 0:
        fl = os.path.join(args.dir_checkpoint, 'cp_%i.pth' % args.load)
        if args.gpus == 0:
            net.load_state_dict(torch.load(fl))
            print('Model loaded from %s' % (fl), flush=True)
        else:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            net.load_state_dict(torch.load(fl, map_location=map_location))
            print('(%i) Model loaded from %s' % (gpu, fl), flush=True)

    duplicates = {}
    if (args.replicate != "") and os.path.exists(args.replicate):
        duplicates = read_duplicate_table(args.replicate)

    dataset_train = InstDataset(args.dir_data, True, False, duplicate=duplicates)
    dataset_val = InstDataset(os.path.join(args.dir_data, "test"), False, True)
    if args.gpus == 0:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        val_sampler = torch.utils.data.RandomSampler(dataset_val)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=args.gpus, rank=gpu)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, num_replicas=args.gpus, rank=gpu)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=val_sampler)

    try:
        train_net(net=net,
                  gpu=gpu,
                  device=device,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  dir_checkpoint=args.dir_checkpoint,
                  epoch1=args.load+1,
                  epochs_all=args.epochs,
                  lr=args.lr,
                  parallel=args.gpus > 1)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


def get_args():
    parser = argparse.ArgumentParser(description='Train AMAP on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dims', metavar='D', type=int, default=16,
                        help='Dimensionality of representations', dest='n_dims')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.005,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=int, default=-1,
                        help='Load model from a checkpoint, -1: find the highest number checkpoint in the checkpoint folder.')
    parser.add_argument('-g', '--gpus', dest='gpus', type=int, default=0, help='Number of GPUs, 0 results in training on CPU')
    parser.add_argument('-r', '--replicate', dest='replicate', type=str, default="../data/duplicate.txt",
                        help='Duplicate some of the files in training data')
    parser.add_argument('-dc', '--dir_checkpoint', dest='dir_checkpoint', type=str, default="../checkpoints",
                        help='Path to the checkpoints folder')
    parser.add_argument('-dd', '--dir_data', dest='dir_data', type=str, default="../data",
                        help='Path to the train data folder')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.dir_checkpoint):
        os.mkdir(args.dir_checkpoint)
        print('Created checkpoint directory %s' % args.dir_checkpoint, flush=True)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    if args.gpus == 0:
        mp.spawn(train, nprocs=1, args=(args,))
    else:
        mp.spawn(train, nprocs=args.gpus, args=(args,))

