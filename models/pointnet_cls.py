from models.transform_nets import InputTransformNet, FeatureTransformNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelCls(nn.Module):

    def __init__(self, input_channel_num=3, point_num=1024):
        super(ModelCls, self).__init__()
        self.transform1 = InputTransformNet(1, point_num)
        self.conv1 = nn.Conv2d(1, 64, (1, 3))
        self.conv2 = nn.Conv2d(64, 64, (1, 1))
        self.transform2 = FeatureTransformNet(64, point_num)
        self.conv3 = nn.Conv2d(64, 64, (1, 1))
        self.conv4 = nn.Conv2d(64, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 1024, (1, 1))
        self.max_pool = nn.MaxPool2d((point_num, 1))
        self.fc1 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.7)
        self.fc3 = nn.Linear(256, 40)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

    def forward(self, x):
        tran1 = self.transform1(x)
        x = x.squeeze(1)
        x = torch.matmul(x, tran1)
        x.unsqueeze_(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        tran2 = self.transform2(x)
        x = x.squeeze(3)
        x = x.permute(0, 2, 1)
        x = torch.matmul(x, tran2)
        x = x.permute(0, 2, 1)
        x.unsqueeze_(3)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = self.max_pool(x)

        x = x.view(-1, 1024)
        x = self.dropout1(self.bn6(self.fc1(x)))
        x = self.dropout2(self.bn7(self.fc2(x)))
        x = self.fc3(x)

        return x, tran1, tran2
