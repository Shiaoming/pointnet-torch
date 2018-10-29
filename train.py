import provider
import os
import numpy as np
import torch
from models.pointnet_cls import ModelCls
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import time

NUM_CHANNEL = 3
NUM_POINT = 1024
BATCH_SIZE = 32
BASE_LEARNING_RATE = 0.001
DECAY_STEP = 200000
DECAY_RATE = 0.7
MAX_EPOCH = 250

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))


def train_one_epoch(model, criterion, optimizer, scheduler):
    model.train()
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    for fn in range(len(TRAIN_FILES)):
        data, label = provider.load_h5(TRAIN_FILES[train_file_idxs[fn]])
        data = data[:, 0:NUM_POINT, :]  # 从点云数据中取NUM_POINT(default:1024)个点
        data, label, _ = provider.shuffle_data(data, label)
        label = np.squeeze(label)

        size = data.shape[0]
        num_batches = size // BATCH_SIZE

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            scheduler.step()  # step every batch
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            batch_data = data[start_idx:end_idx, :, :]
            batch_label = label[start_idx:end_idx]

            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(batch_data)
            jittered_data = provider.jitter_point_cloud(rotated_data)

            inputs = torch.from_numpy(jittered_data).type(torch.FloatTensor)
            inputs = inputs.unsqueeze(1)
            labels = torch.from_numpy(batch_label).type(torch.LongTensor)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs, tran1, tran2 = model(inputs)
                _, preds = torch.max(outputs, 1)
                classify_loss = criterion(outputs, labels)

                # Enforce the transformation as orthogonal matrix
                t1diff = torch.mean(torch.matmul(tran1, tran1.permute(0, 2, 1)), 0)
                t2diff = torch.mean(torch.matmul(tran2, tran2.permute(0, 2, 1)), 0)
                mat_t1diff = nn.MSELoss()(t1diff, torch.eye(3, device=device))
                mat_t2diff = nn.MSELoss()(t2diff, torch.eye(64, device=device))
                mat_diff_loss = mat_t1diff + mat_t2diff
                loss = classify_loss + mat_diff_loss * 0.001

                # backward + optimize
                loss.backward()
                optimizer.step()

            total_seen += BATCH_SIZE
            loss_sum += loss.item()
            total_correct += torch.sum(preds == labels).item()

        epoch_loss = loss_sum / num_batches
        epoch_acc = total_correct / total_seen
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        return epoch_loss, epoch_acc


def eval_one_epoch(model, criterion):
    model.eval()
    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for fn in range(len(TEST_FILES)):
        data, label = provider.load_h5(TEST_FILES[fn])
        data = data[:, 0:NUM_POINT, :]  # 从点云数据中取NUM_POINT(default:1024)个点
        data, label, _ = provider.shuffle_data(data, label)
        label = np.squeeze(label)

        size = data.shape[0]
        num_batches = size // BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            batch_data = data[start_idx:end_idx, :, :]
            batch_label = label[start_idx:end_idx]

            inputs = torch.from_numpy(batch_data).type(torch.FloatTensor)
            inputs = inputs.unsqueeze(1)
            labels = torch.from_numpy(batch_label).type(torch.LongTensor)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs, tran1, tran2 = model(inputs)
            _, preds = torch.max(outputs, 1)
            classify_loss = criterion(outputs, labels)

            # Enforce the transformation as orthogonal matrix
            t1diff = torch.mean(torch.matmul(tran1, tran1.permute(0, 2, 1)), 0)
            t2diff = torch.mean(torch.matmul(tran2, tran2.permute(0, 2, 1)), 0)
            mat_t1diff = nn.MSELoss()(t1diff, torch.eye(3, device=device))
            mat_t2diff = nn.MSELoss()(t2diff, torch.eye(64, device=device))
            mat_diff_loss = mat_t1diff + mat_t2diff
            loss = classify_loss + mat_diff_loss * 0.001

            total_seen += BATCH_SIZE
            loss_sum += loss.item()
            total_correct += torch.sum(preds == labels).item()

    epoch_loss = loss_sum / num_batches
    epoch_acc = total_correct / total_seen
    print('Testing Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


if __name__ == "__main__":
    net = ModelCls(NUM_CHANNEL, NUM_POINT)
    net.to(device)

    '''定义代价函数和优化器'''
    # TODO: 检查CrossEntropyLoss和tf.nn.sparse_softmax_cross_entropy_with_logits
    criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
    optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=0)  # weight_decay就是L2正则化
    # TODO: DECAY_STEP不是在epoch上的，而应该是在global step上的
    lambda1 = lambda batch_step: DECAY_RATE ** (batch_step * BATCH_SIZE // DECAY_STEP)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

    '''训练网络'''
    lr = []
    train_loss = []
    train_acc = []
    eval_loss = []
    eval_acc = []
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times
        print("---------- EPOCH:{}/{} ----------".format(epoch, MAX_EPOCH))
        t = time.time()

        t_loss, t_acc = train_one_epoch(net, criterion, optimizer, scheduler)
        e_loss, e_acc = eval_one_epoch(net, criterion)

        print("time spend one epoch {:.3f}s, remain {:.3f}s".
              format(time.time() - t, (MAX_EPOCH - epoch) * (time.time() - t)))

        train_loss.append(t_loss)
        train_acc.append(t_acc)
        eval_loss.append(e_loss)
        eval_acc.append(e_acc)
        lr.append(scheduler.get_lr())

    plt.ion()

    plt.figure()
    plt.plot(train_loss, label="train_loss")
    plt.plot(eval_loss, label="eval_loss")
    plt.show()

    plt.figure()
    plt.plot(train_acc, label="train_acc")
    plt.plot(eval_acc, label="eval_acc")
    plt.show()

    plt.figure()
    plt.plot(lr, label="lr")
    plt.show()

    print('Finished Training')
