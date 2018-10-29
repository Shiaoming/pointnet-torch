import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InputTransformNet(nn.Module):

    def __init__(self, input_channel_num=3, point_num=1024, K=3):
        super(InputTransformNet, self).__init__()
        self.K = K
        # input: 3 channels(x,y,z)
        # output: 64 channels(64 conv kernel)
        # kernel size: 1 x 3
        self.conv1 = nn.Conv2d(input_channel_num, 64, (1, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, (1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 1024, (1, 1))
        self.bn3 = nn.BatchNorm2d(1024)
        self.max_pool = nn.MaxPool2d((point_num, 1))
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 3 * self.K)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        self.fc3.bias = torch.nn.Parameter(torch.eye(3).view(3 * self.K))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        transform = self.fc3(x)
        transform = transform.view(-1, 3, self.K)
        return transform


class FeatureTransformNet(nn.Module):

    def __init__(self, input_channel_num=3, point_num=1024, K=64):
        super(FeatureTransformNet, self).__init__()
        self.K = K
        # input: 3 channels(x,y,z)
        # output: 64 channels(64 conv kernel)
        # kernel size: 1 x 3
        self.conv1 = nn.Conv2d(input_channel_num, 64, (1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, (1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 1024, (1, 1))
        self.bn3 = nn.BatchNorm2d(1024)
        self.max_pool = nn.MaxPool2d((point_num, 1))
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, self.K * self.K)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        self.fc3.bias = torch.nn.Parameter(torch.eye(self.K).view(self.K * self.K))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        transform = self.fc3(x)
        transform = transform.view(-1, self.K, self.K)
        return transform
