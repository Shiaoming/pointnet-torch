import h5py
import os
import json
import torch
import numpy as np
import scipy.linalg as linalg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '../')
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from utils.viz_utility import *

ALL_FILES = os.path.join(BASE_DIR, '../data/indoor3d_sem_seg_hdf5_data/all_files.txt')


class ShapenetSemSegDataset(Dataset):
    """
    shapenet part seg dataset of hdf5 format
    (https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip).
    This dataset(about 346MB) will be load to memory one time.
    """

    def __init__(self, root_dir, list_filename, train=True, test_area='Area_1_conferenceRoom_1', transform=None):
        '''
        Shapenet part seg dataset. This dataset will be load to memory one time.
        :param root_dir:
        :param list_filename:
        :param transform:
        '''
        first = True
        for line in open(list_filename):
            line = os.path.join(root_dir, line)
            f = h5py.File(line.rstrip())
            if first:
                first = False
                #  N * PointNumbers * 3
                self.data = f['data'][:]
                #  N * PointNumbers
                self.seg = f['label'][:]
            else:
                #  N * PointNumbers * 3
                self.data = np.concatenate((self.data, f['data'][:]), axis=0)
                #  N * PointNumbers
                self.seg = np.concatenate((self.seg, f['label'][:]), axis=0)

        room_filelist = [line.rstrip() for line in
                         open(os.path.join(root_dir, 'indoor3d_sem_seg_hdf5_data/room_filelist.txt'))]
        train_idxs = []
        test_idxs = []
        for i, room_name in enumerate(room_filelist):
            if test_area in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        if train:
            self.data = self.data[train_idxs, ...]
            self.seg = self.seg[train_idxs]
        else:
            self.data = self.data[test_idxs, ...]
            self.seg = self.seg[test_idxs]

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def get_point_num(self):
        return int(self.data.shape[1])

    def __getitem__(self, idx):
        sample = {'points': self.data[idx],
                  'seg': self.seg[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'points': torch.from_numpy(sample['points']).type(torch.FloatTensor),
                'seg': torch.tensor(sample['seg']).type(torch.LongTensor)}


if __name__ == "__main__":
    color_map = json.load(open(os.path.join(BASE_DIR, '../data/hdf5_data/part_color_mapping.json')))

    ROOT_DIR = os.path.join(BASE_DIR, '../data/')
    all_dataset = ShapenetSemSegDataset(ROOT_DIR, ALL_FILES, True)

    print('total samples {}'.format(all_dataset.__len__()))

    # input('pause')
    #
    # for idx in range(0, 5):
    #     print('-' * 30)
    #     print("sample number {}".format(idx))
    #     sample = all_dataset.__getitem__(idx)
    #
    #     print('label size: {}'.format(sample["points"].shape))
    #     print('seg size: {}'.format(sample["seg"].shape))
    #
    #     pc_viewer(sample["points"], sample["seg"], color_map)

    train_loader = DataLoader(all_dataset, batch_size=4, shuffle=True, num_workers=1)

    for i_batch, sample_batched in enumerate(train_loader):
        points = sample_batched['points']
        seg = sample_batched['seg']
        print("-"*30)
        print("batch {}".format(i_batch))
        print('points: {}'.format(points.shape))
        print('seg: {}'.format(seg.shape))
