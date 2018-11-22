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
from classification.dataloader import ModelNet40, RotateY, Rotate, Jitter, ToTensor
from utils.save_load import *
from models.pointner_cls import PointNetCLS, LossCLS

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--base_dir', dest='base_dir',
                    default="../", help='project directory ')
parser.add_argument('--datadir', dest='datadir',
                    default="./data/modelnet40_ply_hdf5_2048",
                    help='data directory ')
parser.add_argument('--save_dir', dest='save_dir', default='./checkpoints',
                    help='directory to save models')
parser.add_argument('--summary_dir', dest='summary_dir', default='./runs',
                    help='directory to save summary, can load by tensorboard')
parser.add_argument('--resume', default='./checkpoints/model_best.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu_id', dest='gpu_id', type=int,
                    default=0, help='which gpu to choose')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_points', default=1024, type=int, metavar='N',
                    help='number of points of cloud')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay', '--lr_decay', default=0.7, type=float,
                    metavar='LRD', help='learning rate decay')
parser.add_argument('--lr_decay_step', default=30, type=float,
                    metavar='LRD', help='learning rate decay')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--rotation_option', default='RotateY', type=str,
                    help='Point cloud rotation option: \n '
                         'RotateY: only rotation around y axis\n '
                         'Rotate: rotation randomly')
parser.add_argument('--input_trans', default=1, type=int,
                    help='add transform net on input cloud')
parser.add_argument('--feature_trans', default=1, type=int,
                    help='add transform net on feature')

args = parser.parse_args()

writer = SummaryWriter(log_dir=args.summary_dir)


def build_dataloader(base_dir, path, num_points, batch_size, rotation_option='RotateY'):
    logger = logging.getLogger('global')
    logger.info("Build dataloader...")

    train_files = os.path.join(path, "train_files.txt")
    test_files = os.path.join(path, "test_files.txt")

    if rotation_option == 'RotateY':
        trans = transforms.Compose([RotateY(), Jitter(), ToTensor()])
    elif rotation_option == 'Rotate':
        trans = transforms.Compose([Rotate(), Jitter(), ToTensor()])
    else:
        trans = transforms.Compose([Jitter(), ToTensor()])

    train_dataset = ModelNet40(root_dir=base_dir, list_filename=train_files, num_point=num_points, transform=trans)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = ModelNet40(root_dir=base_dir, list_filename=test_files, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    logger.info('Build dataloader done')

    return train_loader, test_loader


def main():
    # init logger
    init_log('global', args.save_dir, logging.INFO)
    logger = logging.getLogger('global')
    # print arguments
    for arg in vars(args):
        logger.info("{}: {}".format(arg,getattr(args, arg)))

    # device_name = "cuda:{}".format(args.gpu_id)
    # device = torch.device(device_name)

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # build dataloader and model
    train_loader, test_loader = build_dataloader(args.base_dir, args.datadir, args.num_points,
                                                 args.batch_size, args.rotation_option)
    model = PointNetCLS(input_trans=args.input_trans, feature_trans=args.feature_trans)

    # hl_graph = hl.build_graph(model, torch.zeros([5, 3, args.num_points], device=device))
    # hl_graph.save("graph.png", format="png")

    # check GPU numbers and deploy parallel
    parallel = False
    if torch.cuda.device_count() > 1:
        parallel = True
        logger.info("Let's use {:d} GPUs!".format(torch.cuda.device_count()))
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)

    logger.info("*"*40)
    logger.info(model)
    logger.info("*" * 40)

    # optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # weight_decay就是L2正则化
    lr_lambda = lambda epoch: args.lr_decay ** (epoch // args.lr_decay_step)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lr_lambda])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            model, _, args.start_epoch = restore_from_non_parallel(model, optimizer, args.resume)

    # set the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    logger.info("Start training...")

    for epoch in range(args.start_epoch, args.epochs):
        logger.info('-' * 30)

        t0 = time.time()
        scheduler.step()
        writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        train_one_epoch(train_loader, model, optimizer, device, epoch)
        test_acc = test_one_epoch(test_loader, model, device, epoch)
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


def train_one_epoch(train_loader, model, optimizer, device, epoch):
    logger = logging.getLogger('global')
    model.train()

    total_seen = 0
    loss_sum = 0.0
    total_correct = 0.0
    num_batches = 0.0

    class_correct = np.zeros((40))
    class_total = np.zeros((40))

    for i_batch, sample_batched in enumerate(train_loader):
        points = sample_batched['points']
        labels = sample_batched['labels']

        points = points.permute(0, 2, 1)
        points = points.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs, end_points = model(points)

        _, preds = torch.max(outputs, 1)

        loss = LossCLS(outputs, labels, model, device, end_points)

        # backward + optimize
        loss.backward()
        optimizer.step()

        num_batches = num_batches + 1
        total_seen += sample_batched['labels'].shape[0]
        loss_sum += loss.item()
        total_correct += torch.sum(preds == labels).item()

        labels_numpy = labels.cpu().numpy()
        preds_numpy = preds.cpu().numpy()
        correct_numpy = preds_numpy == labels_numpy
        for i in range(40):
            mask = labels_numpy == i
            class_total[i] += np.sum(mask)
            class_correct[i] += np.sum(mask * correct_numpy)

    epoch_loss = loss_sum / num_batches
    epoch_acc = total_correct / total_seen
    class_ave_acc = np.mean(class_correct / class_total)
    logger.info('Train Loss: {:.4f} Acc: {:.4f}, Class_ave_acc: {:.4f}'.format(epoch_loss, epoch_acc, class_ave_acc))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/acc', epoch_acc, epoch)
    writer.add_scalar('train/ave_acc', class_ave_acc, epoch)


def test_one_epoch(test_loader, model, device, epoch):
    logger = logging.getLogger('global')
    model.eval()

    total_seen = 0
    loss_sum = 0.0
    total_correct = 0.0
    num_batches = 0.0

    class_correct = np.zeros((40))
    class_total = np.zeros((40))

    for i_batch, sample_batched in enumerate(test_loader):
        points = sample_batched['points']
        labels = sample_batched['labels']

        points = points.permute(0, 2, 1).type(torch.FloatTensor)
        labels = labels.type(torch.LongTensor)

        points = points.to(device)
        labels = labels.to(device)

        outputs, end_points = model(points)

        _, preds = torch.max(outputs, 1)
        loss = LossCLS(outputs, labels, model, device, end_points)

        num_batches = num_batches + 1
        total_seen += sample_batched['labels'].shape[0]
        loss_sum += loss.item()
        total_correct += torch.sum(preds == labels).item()

        labels_numpy = labels.cpu().numpy()
        preds_numpy = preds.cpu().numpy()
        correct_numpy = preds_numpy == labels_numpy
        for i in range(40):
            mask = labels_numpy == i
            class_total[i] += np.sum(mask)
            class_correct[i] += np.sum(mask * correct_numpy)

    epoch_loss = loss_sum / num_batches
    epoch_acc = total_correct / total_seen
    class_ave_acc = np.mean(class_correct / class_total)
    logger.info('Test Loss: {:.4f} Acc: {:.4f}, Class_ave_acc: {:.4f}'.format(epoch_loss, epoch_acc, class_ave_acc))

    writer.add_scalar('test/loss', epoch_loss, epoch)
    writer.add_scalar('test/acc', epoch_acc, epoch)
    writer.add_scalar('test/ave_acc', class_ave_acc, epoch)

    return epoch_acc


if __name__ == "__main__":
    main()
