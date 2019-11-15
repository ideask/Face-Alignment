#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from DataLoader.pfld import MyDatasets
from Models.pfld import PFLDInference, AuxiliaryNet
from Loss.pfld import PFLDLoss
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


def train(train_loader, pfld_backbone, auxiliarynet, criterion, optimizer, cur_epoch):
    losses = AverageMeter()

    for img, landmark_gt, euler_angle_gt in train_loader:
        img.requires_grad = False
        img = img.cuda(non_blocking=True)

        landmark_gt.requires_grad = False
        landmark_gt = landmark_gt.cuda(non_blocking=True)

        euler_angle_gt.requires_grad = False
        euler_angle_gt = euler_angle_gt.cuda(non_blocking=True)

        pfld_backbone = pfld_backbone.cuda()
        auxiliarynet = auxiliarynet.cuda()

        features, landmarks = pfld_backbone(img)
        angle = auxiliarynet(features)
        weighted_loss, loss = criterion(landmark_gt, euler_angle_gt,
                                    angle, landmarks, args.train_batchsize)

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        losses.update(loss.item())
    return weighted_loss, loss


def validate(my_val_dataloader, pfld_backbone, auxiliarynet, criterion, cur_epoch):
    pfld_backbone.eval()
    auxiliarynet.eval()
    losses = []
    with torch.no_grad():
        for img, landmark_gt, euler_angle_gt in my_val_dataloader:
            img.requires_grad = False
            img = img.cuda(non_blocking=True)

            landmark_gt.requires_grad = False
            landmark_gt = landmark_gt.cuda(non_blocking=True)

            euler_angle_gt.requires_grad = False
            euler_angle_gt = euler_angle_gt.cuda(non_blocking=True)

            pfld_backbone = pfld_backbone.cuda()
            auxiliarynet = auxiliarynet.cuda()

            _, landmark = pfld_backbone(img)

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
    pfld_backbone = PFLDInference().cuda()
    auxiliarynet = AuxiliaryNet().cuda()
    if args.resume != '':
        logging.info('Load the checkpoint:{}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
        auxiliarynet.load_state_dict(checkpoint['auxiliarynet'])
    criterion = PFLDLoss()
    optimizer = torch.optim.Adam(
        [{
            'params': pfld_backbone.parameters()
        }, {
            'params': auxiliarynet.parameters()
        }],
        lr=args.base_lr,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    mydataset = MyDatasets(args.dataroot, transform)
    dataloader = DataLoader(
        mydataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)

    my_val_dataset = MyDatasets(args.val_dataroot, transform)
    my_val_dataloader = DataLoader(
        my_val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, train_loss = train(dataloader, pfld_backbone, auxiliarynet,
                                      criterion, optimizer, epoch)
        filename = os.path.join(
            str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint({
            'epoch': epoch,
            'pfld_backbone': pfld_backbone.state_dict(),
            'auxiliarynet': auxiliarynet.state_dict()
        }, filename)

        val_loss = validate(my_val_dataloader, pfld_backbone, auxiliarynet,
                            criterion, epoch)

        scheduler.step(val_loss)
        # 第一个参数可以简单理解为保存图的名称，第二个参数是可以理解为Y轴数据，第三个参数可以理解为X轴数据
        # weighted_loss带权重计算的 train loss
        # train_loss 单纯L2 loss
        # val_loss 验证数据集的loss
        writer.add_scalar('data/weighted_loss', weighted_train_loss, epoch)
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
        default='./CheckPoints/snapshot_pfld/',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--log_file', default="./CheckPoints/train_pfld.logs", type=str)
    parser.add_argument(
        '--tensorboard', default="./CheckPoints/tensorboard_pfld", type=str)
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
