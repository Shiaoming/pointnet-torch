import h5py
import os
import torch
import numpy as np
import scipy.linalg as linalg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '../')
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# from utils.viz_utility import *

# ModelNet40 official train/test split
TRAIN_FILES = os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/train_files.txt')
TEST_FILES = os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/test_files.txt')


######################################################################
# Dataset class
# -------------
#
# ``torch.utils.data.Dataset`` is an abstract class representing a
# dataset.
# Your custom dataset should inherit ``Dataset`` and override the following
# methods:
#
# -  ``__len__`` so that ``len(dataset)`` returns the size of the dataset.
# -  ``__getitem__`` to support the indexing such that ``dataset[i]`` can
#    be used to get :math:`i`\ th sample

class ModelNet40(Dataset):
    """
    modelnet40 dataset(https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).
    This dataset(about 415MB) will be load to memory one time.
    """

    def __init__(self, root_dir, list_filename, num_point=1024, transform=None):
        assert num_point <= 2048
        self.data = np.zeros((1, num_point, 3))
        self.label = np.zeros((1, 1))
        # due to the dataset is small, we read all the files in memory,
        # so in dataloader, we only use one thread to load data
        for line in open(list_filename):
            line = os.path.join(root_dir, line)
            f = h5py.File(line.rstrip())
            #  N * PointNumbers * 3
            self.data = np.concatenate((self.data, f['data'][:, 0:num_point, :]), axis=0)
            #  N * PointNumbers
            self.label = np.concatenate((self.label, f['label'][:]), axis=0)
            # self.data.append(f['data'][0,:])
            # self.label.append(f['label'][0,:])
        self.data = np.delete(self.data, 0, axis=0)
        self.label = np.delete(self.label, 0, axis=0)

        self.label = np.squeeze(self.label, 1)

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = {'points': self.data[idx], 'labels': self.label[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


######################################################################
# Transforms
# ----------
#
# One issue we can see from the above is that the samples are not of the
# same size. Most neural networks expect the images of a fixed size.
# Therefore, we will need to write some prepocessing code.
# Let's create three transforms:
#
# -  ``Rotate``: to rotate the point cloud
# -  ``Jitter``: to jitter from image randomly. This is data
#    augmentation.
# -  ``ToTensor``: to convert the numpy images to torch images (we need to
#    swap axes).
#
# We will write them as callable classes instead of simple functions so
# that parameters of the transform need not be passed everytime it's
# called. For this, we just need to implement ``__call__`` method and
# if required, ``__init__`` method. We can then use a transform like this:
#
# ::
#
#     tsfm = Transform(params)
#     transformed_sample = tsfm(sample)
#
# Observe below how these transforms had to be applied both on the image and
# landmarks.
#

class RotateY(object):
    """Rotate the input cloud randomly
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        # 作者的旋转只是绕y轴的，而正常的三维旋转应该是一个轴角（单位向量，2个自由度）
        angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])

        rot_points = sample['points'].dot(rotation_matrix)

        return {'points': rot_points, 'labels': sample['labels']}


class Rotate(object):
    """Rotate the input cloud randomly
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        # 作者的旋转只是绕y轴的，而正常的三维旋转应该是一个轴角（单位向量，2个自由度）
        # a random axis
        a = np.random.randn(3)
        # so(3)
        r_hat = np.array([[0, -a[2], a[1]],
                          [a[2], 0, -a[0]],
                          [-a[1], a[0], 0]])
        # to SO(3) using exponential mapping
        rotation_matrix = linalg.expm(r_hat)

        rot_points = sample['points'].dot(rotation_matrix)

        return {'points': rot_points, 'labels': sample['labels']}


class Jitter(object):
    """Jitter randomly the points in a sample.

    Args:
        sigma: sigma of random,
        clip: clip extent
    """

    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, sample):
        N, C = sample['points'].shape

        jitter = np.clip(self.sigma * np.random.randn(N, C), -1 * self.clip, self.clip)

        jittered_points = sample['points'] + jitter

        return {'points': jittered_points, 'labels': sample['labels']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'points': torch.from_numpy(sample['points']).type(torch.FloatTensor),
                'labels': torch.tensor(sample['labels']).type(torch.LongTensor)}


if __name__ == "__main__":

    ######################################################################
    # Test basic read data

    # train_dataset = ModelNet40(ROOT_DIR,TRAIN_FILES)
    #
    # print(train_dataset.__len__())
    #
    # for idx in range(50):
    #     sample = train_dataset.__getitem__(idx)
    #     print("class: {}, max_point:({:.4f},{:.4f},{:.4f}), min_point:({:.4f},{:.4f},{:.4f})".format(sample["labels"],
    #                                                                          np.max(sample["points"][:, 0]),
    #                                                                          np.max(sample["points"][:, 1]),
    #                                                                          np.max(sample["points"][:, 2]),
    #                                                                          np.min(sample["points"][:, 0]),
    #                                                                          np.min(sample["points"][:, 1]),
    #                                                                          np.min(sample["points"][:, 2])))
    #     pc_viewer(sample["points"])

    ######################################################################
    # Compose transforms
    # ~~~~~~~~~~~~~~~~~~

    # rotate = Rotate()
    # jitter = Jitter()
    # composed = transforms.Compose([Rotate(),
    #                                Jitter()])
    #
    # for i, tsfrm in enumerate([rotate, jitter, composed]):
    #     transformed_sample = tsfrm(sample)
    #     pc_viewer(transformed_sample["points"])

    ######################################################################
    # Iterating through the dataset
    # -----------------------------
    #
    # Let's put this all together to create a dataset with composed
    # transforms.

    transformed_train_dataset = ModelNet40(root_dir=ROOT_DIR,
                                           list_filename=TRAIN_FILES,
                                           transform=transforms.Compose([
                                               Rotate(),
                                               Jitter(),
                                               ToTensor()
                                           ]))

    # for i in range(len(transformed_train_dataset)):
    #     sample = transformed_train_dataset[i]
    #
    #     print(i, sample['points'].size(), sample['labels'].size())
    #
    #     if i == 3:
    #         break

    ######################################################################
    # However, we are losing a lot of features by using a simple ``for`` loop to
    # iterate over the data. In particular, we are missing out on:
    #
    # -  Batching the data
    # -  Shuffling the data
    # -  Load the data in parallel using ``multiprocessing`` workers.
    #
    # ``torch.utils.data.DataLoader`` is an iterator which provides all these
    # features. Parameters used below should be clear. One parameter of
    # interest is ``collate_fn``. You can specify how exactly the samples need
    # to be batched using ``collate_fn``. However, default collate should work
    # fine for most use cases.
    #

    dataloader = DataLoader(transformed_train_dataset, batch_size=50,
                            shuffle=True, num_workers=0)

    print(dataloader)


    # Helper function to show a batch
    def show_pointcloud_batch(sample_batched):
        points_batch, labels_batch = \
            sample_batched['points'], sample_batched['labels']
        batch_size = len(points_batch)

        # for i in range(batch_size):
        #     pc_viewer(points_batch[i])


    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['points'].size(),
              sample_batched['labels'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            show_pointcloud_batch(sample_batched)
            break
