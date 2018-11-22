import argparse
import os
import numpy as np
import time
import copy

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
from tensorboardX import SummaryWriter

import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '../')
sys.path.append(ROOT_DIR)

from utils.log_helper import *
from models.pointnet_common import *
from part_seg.dataloader import ShapenetPartSegDataset, ToTensor
from utils.save_load import *
from models.pointnet_partseg import PointNetPartSeg, LossPartSeg

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--base_dir', dest='base_dir',
                    default="../data/hdf5_data/", help='project directory ')
parser.add_argument('--datadir', dest='datadir',
                    default="../data/hdf5_data",
                    help='data directory ')
parser.add_argument('--save_dir', dest='save_dir', default='./checkpoints',
                    help='directory to save models')
parser.add_argument('--summary_dir', dest='summary_dir', default='./runs',
                    help='directory to save summary, can load by tensorboard')
parser.add_argument('--resume', default='./checkpoints/model_best.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--gpu_id', dest='gpu_id', type=int,
                    default=1, help='which gpu to choose(-1 for auto choose)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay', '--lr_decay', default=0.5, type=float,
                    metavar='LRD', help='learning rate decay')
parser.add_argument('--lr_decay_step', default=30, type=float,
                    metavar='LRD', help='learning rate decay')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--cat_num', '--cn', default=16, type=int,
                    metavar='CN', help='the number of category')
parser.add_argument('--part_num', '--pn', default=50, type=int,
                    metavar='PN', help='the number of part')

args = parser.parse_args()

writer = SummaryWriter(log_dir=args.summary_dir)


def build_dataloader(base_dir, path, batch_size):
    logger = logging.getLogger('global')
    logger.info("Build dataloader...")

    train_files = os.path.join(path, "train_hdf5_file_list.txt")
    test_files = os.path.join(path, "val_hdf5_file_list.txt")

    train_dataset = ShapenetPartSegDataset(root_dir=base_dir, list_filename=train_files, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = ShapenetPartSegDataset(root_dir=base_dir, list_filename=test_files, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    point_num = train_dataset.get_point_num()

    logger.info('Build dataloader done')

    return train_loader, test_loader, point_num


def main():
    init_log('global', args.save_dir, logging.INFO)
    logger = logging.getLogger('global')

    for arg in vars(args):
        logger.info("{}: {}".format(arg, getattr(args, arg)))

    train_loader, test_loader, point_num = build_dataloader(args.base_dir, args.datadir, args.batch_size)

    model = PointNetPartSeg(in_channels=3, point_num=point_num, cat_num=args.cat_num, part_num=args.part_num,
                            input_trans=True, feature_trans=True)

    parallel = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.gpu_id == -1:
        if torch.cuda.device_count() > 1:
            parallel = True
            logger.info("Let's use {:d} GPUs!".format(torch.cuda.device_count()))
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
    else:
        device_name = "cuda:{}".format(args.gpu_id)
        logger.info("Device name: {}".format(device_name))
        device = torch.device(device_name)

    model.to(device)

    logger.info("*" * 40)
    logger.info(model)
    logger.info("*" * 40)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # weight_decay就是L2正则化
    lr_lambda = lambda epoch: args.lr_decay ** (epoch // args.lr_decay_step)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lr_lambda])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info('restore from {}'.format(args.resume))
            if parallel:
                model, _, args.start_epoch = restore_from_non_parallel(model, optimizer, args.resume)
            else:
                model, _, args.start_epoch = restore_from(model, optimizer, args.resume)


    # set the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    logger.info("Start training...")

    for epoch in range(args.start_epoch, args.epochs):
        logger.info('-' * 30)

        t0 = time.time()
        scheduler.step()
        writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        train_one_epoch(train_loader, model, optimizer, device, epoch, point_num)
        test_acc = test_one_epoch(test_loader, model, device, epoch, point_num)
        t1 = time.time()

        if test_acc > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % 10 == 1:
            filename = os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch + 1))
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                is_best=False,
                filename=filename
            )
            logger.info("Saved model : {}".format(filename))

        print_speed(epoch, t1 - t0, args.epochs)

    save_checkpoint(
        {
            'epoch': epoch + 1,
            'state_dict': best_model_wts,
            'optimizer': optimizer.state_dict()
        },
        is_best=True,
        filename=os.path.join(args.save_dir, 'model_best.pth')
    )

    writer.close()


def train_one_epoch(train_loader, model, optimizer, device, epoch, point_num):
    logger = logging.getLogger('global')
    model.train()

    total_seen = 0
    loss_sum_cls = 0.0
    loss_sum_seg = 0.0
    loss_sum = 0.0
    total_correct_cls = 0.0
    total_correct_seg = 0.0
    num_batches = 0.0

    class_correct = np.zeros((args.cat_num))
    class_total = np.zeros((args.cat_num))
    seg_correct = np.zeros((args.part_num))
    seg_total = np.zeros((args.part_num))

    for i_batch, sample_batched in enumerate(train_loader):
        points = sample_batched['points']
        labels = sample_batched['labels']
        seg = sample_batched['seg']

        points = points.permute(0, 2, 1)  # B*C*N
        points = points.to(device)
        labels = labels.to(device)  # B,
        seg = seg.to(device)  # B * N

        # zero the parameter gradients
        optimizer.zero_grad()

        # label_pre and seg_pre are original output from network before softmax
        # label_pre: B * cat_num
        # seg_pre: B * N * seg_num
        label_pre, seg_pre, end_points = model(points, labels)

        # classify preds (B,) are labels
        _, preds_cls = torch.max(label_pre, 1)
        # segmentation preds (B,N) are aeg labels
        preds_seg = torch.argmax(seg_pre, 1)

        total_loss, label_loss, seg_loss = LossPartSeg(label_pre, seg_pre, labels, seg, 0.6, end_points, device)

        # backward + optimize
        total_loss.backward()
        optimizer.step()

        num_batches = num_batches + 1
        total_seen += sample_batched['labels'].shape[0]
        loss_sum += total_loss.item()
        loss_sum_cls += label_loss.item()
        loss_sum_seg += seg_loss.item()
        total_correct_cls += torch.sum(preds_cls == labels).item()
        total_correct_seg += torch.sum(preds_seg == seg).item()

        # compute average class acc
        labels_numpy = labels.cpu().numpy()
        preds_numpy = preds_cls.cpu().numpy()
        correct_numpy = preds_numpy == labels_numpy
        for i in range(args.cat_num):
            mask = labels_numpy == i
            class_total[i] += np.sum(mask)
            class_correct[i] += np.sum(mask * correct_numpy)

        # compute average seg acc
        seg_numpy = seg.cpu().numpy()
        preds_numpy = preds_seg.cpu().numpy()
        correct_numpy = preds_numpy == seg_numpy
        for i in range(args.part_num):
            mask = seg_numpy == i
            seg_total[i] += np.sum(mask)
            seg_correct[i] += np.sum(mask * correct_numpy)

    epoch_loss = loss_sum / num_batches
    epoch_loss_cls = loss_sum_cls / num_batches
    epoch_loss_seg = loss_sum_seg / num_batches
    epoch_acc_cls = total_correct_cls / total_seen
    epoch_acc_seg = total_correct_seg / total_seen / point_num
    class_ave_acc = np.mean(class_correct / class_total)
    # seg_total_zero_mask = seg_total == 0
    # seg_total += seg_total_zero_mask
    seg_ave_acc = np.mean(seg_correct / seg_total)
    logger.info(
        'Train Total Loss: {:.4f} Classify loss: {:.4f}, Segmentation loss: {:.4f}'.format(epoch_loss, epoch_loss_cls,
                                                                                           epoch_loss_seg))
    logger.info('Classify Acc: {:.4f}, classify average class acc: {:.4f}'.format(epoch_acc_cls, class_ave_acc))
    logger.info('Segmentation Acc: {:.4f}, seg average part acc: {:.4f}'.format(epoch_acc_seg, seg_ave_acc))

    writer.add_scalars("train_loss",
                       {'total loss': epoch_loss,
                        'cls loss': epoch_loss_cls,
                        'seg loss': epoch_loss_seg},
                       epoch)

    writer.add_scalars("train_acc_cls",
                       {'total cls acc': epoch_acc_cls,
                        'ave cls acc': class_ave_acc},
                       epoch)
    writer.add_scalars("train_acc_seg",
                       {'total seg acc': epoch_acc_seg,
                        'ave part acc': seg_ave_acc},
                       epoch)

    # writer.add_scalar('train/loss', epoch_loss, epoch)
    # writer.add_scalar('train/loss_cls', epoch_loss_cls, epoch)
    # writer.add_scalar('train/loss_seg', epoch_loss_seg, epoch)
    # writer.add_scalar('train/acc', epoch_acc, epoch)
    # writer.add_scalar('train/ave_acc', class_ave_acc, epoch)


def test_one_epoch(test_loader, model, device, epoch, point_num):
    logger = logging.getLogger('global')
    model.eval()

    total_seen = 0
    loss_sum_cls = 0.0
    loss_sum_seg = 0.0
    loss_sum = 0.0
    total_correct_cls = 0.0
    total_correct_seg = 0.0
    num_batches = 0.0

    class_correct = np.zeros((args.cat_num))
    class_total = np.zeros((args.cat_num))
    seg_correct = np.zeros((args.part_num))
    seg_total = np.zeros((args.part_num))

    for i_batch, sample_batched in enumerate(test_loader):
        points = sample_batched['points']
        labels = sample_batched['labels']
        seg = sample_batched['seg']

        points = points.permute(0, 2, 1)  # B*C*N
        points = points.to(device)
        labels = labels.to(device)  # B,
        seg = seg.to(device)  # B * N

        # label_pre and seg_pre are original output from network before softmax
        # label_pre: B * cat_num
        # seg_pre: B * N * seg_num
        label_pre, seg_pre, end_points = model(points, labels)

        # classify preds (B,) are labels
        _, preds_cls = torch.max(label_pre, 1)
        # segmentation preds (B,N) are aeg labels
        preds_seg = torch.argmax(seg_pre, 1)

        total_loss, label_loss, seg_loss = LossPartSeg(label_pre, seg_pre, labels, seg, 0.6, end_points, device)

        num_batches = num_batches + 1
        total_seen += sample_batched['labels'].shape[0]
        loss_sum += total_loss.item()
        loss_sum_cls += label_loss.item()
        loss_sum_seg += seg_loss.item()
        total_correct_cls += torch.sum(preds_cls == labels).item()
        total_correct_seg += torch.sum(preds_seg == seg).item()

        # compute average class acc
        labels_numpy = labels.cpu().numpy()
        preds_numpy = preds_cls.cpu().numpy()
        correct_numpy = preds_numpy == labels_numpy
        for i in range(args.cat_num):
            mask = labels_numpy == i
            class_total[i] += np.sum(mask)
            class_correct[i] += np.sum(mask * correct_numpy)

        # compute average seg acc
        labels_numpy = seg.cpu().numpy()
        preds_numpy = preds_seg.cpu().numpy()
        correct_numpy = preds_numpy == labels_numpy
        for i in range(args.part_num):
            mask = labels_numpy == i
            seg_total[i] += np.sum(mask)
            seg_correct[i] += np.sum(mask * correct_numpy)

    epoch_loss = loss_sum / num_batches
    epoch_loss_cls = loss_sum_cls / num_batches
    epoch_loss_seg = loss_sum_seg / num_batches
    epoch_acc_cls = total_correct_cls / total_seen
    epoch_acc_seg = total_correct_seg / total_seen / point_num
    class_ave_acc = np.mean(class_correct / class_total)
    # seg_total_zero_mask = seg_total == 0
    # seg_total += seg_total_zero_mask
    seg_ave_acc = np.mean(seg_correct / seg_total)
    logger.info(
        'Test Total Loss: {:.4f} Classify loss: {:.4f}, Segmentation loss: {:.4f}'.format(epoch_loss, epoch_loss_cls,
                                                                                          epoch_loss_seg))
    logger.info('Classify Acc: {:.4f}, classify average class acc: {:.4f}'.format(epoch_acc_cls, class_ave_acc))
    logger.info('Segmentation Acc: {:.4f}, seg average part acc: {:.4f}'.format(epoch_acc_seg, seg_ave_acc))

    writer.add_scalars("test_loss",
                       {'total loss': epoch_loss,
                        'cls loss': epoch_loss_cls,
                        'seg loss': epoch_loss_seg},
                       epoch)

    writer.add_scalars("test_acc_cls",
                       {'total cls acc': epoch_acc_cls,
                        'ave cls acc': class_ave_acc},
                       epoch)
    writer.add_scalars("test_acc_seg",
                       {'total seg acc': epoch_acc_seg,
                        'ave part acc': seg_ave_acc},
                       epoch)

    return epoch_acc_seg


if __name__ == "__main__":
    main()
