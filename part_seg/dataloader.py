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

TRAIN_FILES = os.path.join(BASE_DIR, '../data/hdf5_data/train_hdf5_file_list.txt')
TEST_FILES = os.path.join(BASE_DIR, '../data/hdf5_data/test_hdf5_file_list.txt')


class ShapenetPartSegDataset(Dataset):
    """
    shapenet part seg dataset of hdf5 format
    (https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip).
    This dataset(about 346MB) will be load to memory one time.
    """

    def __init__(self, root_dir, list_filename, transform=None):
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
                #  N
                self.label = f['label'][:]
                #  N * PointNumbers
                self.seg = f['pid'][:]
            else:
                #  N * PointNumbers * 3
                self.data = np.concatenate((self.data, f['data'][:]), axis=0)
                #  N
                self.label = np.concatenate((self.label, f['label'][:]), axis=0)
                #  N * PointNumbers
                self.seg = np.concatenate((self.seg, f['pid'][:]), axis=0)

        self.label = np.squeeze(self.label, 1)

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def get_point_num(self):
        return int(self.data.shape[1])

    def __getitem__(self, idx):
        sample = {'points': self.data[idx],
                  'labels': self.label[idx],
                  'seg': self.seg[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'points': torch.from_numpy(sample['points']).type(torch.FloatTensor),
                'labels': torch.tensor(int(sample['labels'])).type(torch.LongTensor),
                'seg': torch.tensor(sample['seg']).type(torch.LongTensor)}


if __name__ == "__main__":

    color_map = json.load(open(os.path.join(BASE_DIR, '../data/hdf5_data/part_color_mapping.json')))

    ROOT_DIR = os.path.join(BASE_DIR, '../data/hdf5_data/')
    test_dataset = ShapenetPartSegDataset(ROOT_DIR, TEST_FILES)

    print('total samples {}'.format(test_dataset.__len__()))

    for idx in range(20, 25):
        print('-' * 30)
        print("sample number {}".format(idx))
        sample = test_dataset.__getitem__(idx)
        print("class: {}, max_point:({:.4f},{:.4f},{:.4f}), "
              "min_point:({:.4f},{:.4f},{:.4f})".format(sample["labels"],
                                                        np.max(sample["points"][:, 0]),
                                                        np.max(sample["points"][:, 1]),
                                                        np.max(sample["points"][:, 2]),
                                                        np.min(sample["points"][:, 0]),
                                                        np.min(sample["points"][:, 1]),
                                                        np.min(sample["points"][:, 2])))

        pc_viewer(sample["points"], sample["seg"], color_map)
