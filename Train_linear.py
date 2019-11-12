#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DataLoader.linear import load_data
from Models.linear import LinearNet
from Loss.linear import LinearLoss
from Utils.utils import AverageMeter
# from utils.parallel import DataParallelModel, DataParallelCriterion

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, linear_backbone, criterion, optimizer, cur_epoch):
    losses = AverageMeter()

    for samples in train_loader:
        img = samples['image']
        landmark_gt = samples['landmarks']
        img.requires_grad = False
        img = img.cuda(non_blocking=True)

        landmark_gt.requires_grad = False
        landmark_gt = landmark_gt.cuda(non_blocking=True)

        linear_backbone = linear_backbone.cuda()

        landmarks = linear_backbone(img)
        loss = criterion(landmark_gt, landmarks, args.train_batchsize)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
    return loss


def validate(my_val_dataloader, linear_backbone, criterion, cur_epoch):
    linear_backbone.eval()
    losses = []
    with torch.no_grad():
        for samples in my_val_dataloader:
            img = samples['image']
            landmark_gt = samples['landmarks']
            img.requires_grad = False
            img = img.cuda(non_blocking=True)

            landmark_gt.requires_grad = False
            landmark_gt = landmark_gt.cuda(non_blocking=True)

            linear_backbone = linear_backbone.cuda()

            landmark = linear_backbone(img)

            loss = torch.mean(
                torch.sum((landmark_gt - landmark)**2,axis=1))
            losses.append(loss.cpu().numpy())

    return np.mean(losses)


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    linear_backbone = LinearNet().cuda()
    if args.resume != '':
        logging.info('Load the checkpoint:{}'.format(args.resume))
        checkpoint = torch.load(args.model_path)
        linear_backbone.load_state_dict(checkpoint['linear_backbone'])
    criterion = LinearLoss()
    optimizer = torch.optim.Adam(
        [{
            'params': linear_backbone.parameters()
        }],
        lr=args.base_lr,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)

    # step 3: data
    # argumetion
    mydataset = load_data(args.dataroot)
    dataloader = DataLoader(
        mydataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)

    my_val_dataset = load_data(args.val_dataroot)
    my_val_dataloader = DataLoader(
        my_val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train_loss = train(dataloader, linear_backbone,
                                      criterion, optimizer, epoch)
        filename = os.path.join(
            str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint({
            'epoch': epoch,
            'linear_backbone': linear_backbone.state_dict()
        }, filename)

        val_loss = validate(my_val_dataloader, linear_backbone, criterion, epoch)

        scheduler.step(val_loss)
        # 第一个参数可以简单理解为保存图的名称，第二个参数是可以理解为Y轴数据，第三个参数可以理解为X轴数据
        # train_loss 单纯L2 loss
        # val_loss 验证数据集的loss
        writer.add_scalars('data/loss', {'val loss': val_loss, 'train loss': train_loss}, epoch)
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Face Alignment Project Trainning')
    # general
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD

    # training
    # -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=1000, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument(
        '--snapshot',
        default='./CheckPoints/snapshot_linear/',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--log_file', default="./CheckPoints/train_linear.logs", type=str)
    parser.add_argument(
        '--tensorboard', default="./CheckPoints/tensorboard_linear", type=str)
    # -- load snapshot
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH')  # TBD

    # --dataset
    parser.add_argument(
        '--dataroot',
        default='./Data/ODATA/TrainData/labels.txt',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--val_dataroot',
        default='./Data/ODATA/TestData/labels.txt',
        type=str,
        metavar='PATH')
    parser.add_argument('--train_batchsize', default=128, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
