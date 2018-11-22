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
from utils.viz_utility import *
import json
import multiprocessing as mp

import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '../')
sys.path.append(ROOT_DIR)

TEST_FILES = os.path.join(BASE_DIR, '../data/hdf5_data/test_hdf5_file_list.txt')

from utils.log_helper import *
from models.pointnet_common import *
from part_seg.dataloader import ShapenetPartSegDataset, ToTensor
from utils.save_load import *
from models.pointnet_partseg import PointNetPartSeg, LossPartSeg

parser = argparse.ArgumentParser()

parser.add_argument('--base_dir', dest='base_dir',
                    default="../data/hdf5_data/", help='project directory ')
parser.add_argument('--datadir', dest='datadir',
                    default="../data/hdf5_data",
                    help='data directory ')
parser.add_argument('--resume', default='./checkpoints/model_best.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu_id', dest='gpu_id', type=int,
                    default=0, help='which gpu to choose(-1 for auto choose)')
parser.add_argument('--cat_num', '--cn', default=16, type=int,
                    metavar='CN', help='the number of category')
parser.add_argument('--part_num', '--pn', default=50, type=int,
                    metavar='PN', help='the number of part')

args = parser.parse_args()


def view_process(q, color_map):
    print('-> view_process: start')

    while True:
        print('-> view_process: queue get')
        sample = q.get()
        if type(sample) == int and sample == -1:
            print('-> view_process: exiting')
            break
        elif type(sample) == dict:
            print('-> view_process: drawing')
            fig1 = mayavi.mlab.figure('GT')
            fig2 = mayavi.mlab.figure('Predict')
            # GT
            pc_viewer(sample['points'], sample['gt'], color_map, fig1, False)
            # Predict
            pc_viewer(sample['points'], sample['predict'], color_map, fig2, False)
            mayavi.mlab.show()


def main():
    color_map = json.load(open(os.path.join(BASE_DIR, '../data/hdf5_data/part_color_mapping.json')))

    ROOT_DIR = os.path.join(BASE_DIR, '../data/hdf5_data/')
    test_dataset = ShapenetPartSegDataset(ROOT_DIR, TEST_FILES)

    print('total samples {}'.format(test_dataset.__len__()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = PointNetPartSeg(in_channels=3, point_num=2048, cat_num=args.cat_num, part_num=args.part_num,
                            input_trans=True, feature_trans=True)
    model.to(device)

    model = load_checkpoint(model, args.resume)

    model.eval()

    totensor = ToTensor()

    q = mp.Queue()
    process_view = mp.Process(target=view_process, args=(q, color_map))
    process_view.start()

    for idx in range(5):
        print('-' * 30)
        print("sample number {}".format(idx))
        sample = test_dataset.__getitem__(idx)
        sample = totensor(sample)

        points = torch.unsqueeze(sample['points'], 0).to(device)
        labels = torch.unsqueeze(sample['labels'], 0).to(device)
        # seg = torch.unsqueeze(sample['seg'],0)

        points = points.permute(0, 2, 1)  # B*C*N

        # label_pre and seg_pre are original output from network before softmax
        # label_pre: B * cat_num
        # seg_pre: B * N * seg_num
        t1 = time.time()
        label_pre, seg_pre, end_points = model(points, labels)
        print('inference time: {:.4f}'.format(time.time() - t1))

        # classify preds (B,) are labels
        _, preds_cls = torch.max(label_pre, 1)
        # segmentation preds (B,N) are aeg labels
        preds_seg = torch.argmax(seg_pre, 1)

        # numpy
        pred_seg_numpy = preds_seg.cpu().numpy()
        pred_seg_numpy = np.squeeze(pred_seg_numpy)

        print('queue put process ...')

        q.put({'points': sample["points"],
               'gt': sample['seg'],
               'predict': pred_seg_numpy})

        print('queue put process finished')

        input("pause")

    q.put(-1)

    process_view.join()


if __name__ == "__main__":
    main()
