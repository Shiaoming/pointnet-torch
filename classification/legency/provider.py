# coding = utf-8

import os
import sys
import numpy as np
import h5py

FILE_NAME1 = "data/modelnet40_ply_hdf5_2048/ply_data_train0.h5"


def shuffle_data(data, labels):
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """
    TODO:他这里的旋转是绕y轴的，正常的三维旋转应该是一个轴角（单位向量，2个自由度）
    :param batch_data:
    :return:
    """
    rotated_data = np.zeros_like(batch_data, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        pc = batch_data[k]
        rotated_data[k] = np.dot(pc, rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    B, N, C = batch_data.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]  # 格式为：N*PointNumbers*3
    label = f['label'][:]
    return data, label


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return data, label, seg


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def viz_mayavi(points):
    import mayavi.mlab

    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point

    mayavi.mlab.points3d(x, y, z)

    # r = lidar[:, 3]  # reflectance value of point
    # d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    # if vals == "height":
    #     col = z
    # else:
    #     col = d
    # fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    # mayavi.mlab.points3d(x, y, z,
    #                      col,  # Values used for Color
    #                      mode="sphere",
    #                      colormap='gnuplot',  # 'bone', 'copper', 'gnuplot','spectral'
    #                      # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
    #                      figure=fig,
    #                      )
    mayavi.mlab.show()


if __name__ == "__main__":
    data, label = load_h5(FILE_NAME1)
    # data, label, seg = load_h5_data_label_seg(FILE_NAME1)

    viz_mayavi(data[0])
    viz_mayavi(data[1])
    viz_mayavi(data[2])
    viz_mayavi(data[3])
