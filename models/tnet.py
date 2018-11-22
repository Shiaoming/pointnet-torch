import torch
import torch.nn as nn
from models.fcn import FCN, PFCN


class InputTransformNet(nn.Module):
    '''
    input -> SFCN(64) -> SFCN(128) -> SFCN(1024) -> MaxPool -> FCN(512) -> FCN(256) -> Linear(3*3)
    '''
    def __init__(self, in_channels=3, point_num=1024, K=3):
        super(InputTransformNet, self).__init__()
        self.K = K
        self.pfcn1 = PFCN(in_channels, 64)
        self.pfcn2 = PFCN(64, 128)
        self.pfcn3 = PFCN(128, 1024)
        self.max_pool = nn.MaxPool1d(kernel_size=point_num)
        self.fcn1 = FCN(1024, 512)
        self.fcn2 = FCN(512, 256)

        self.fc3 = nn.Linear(256, 3 * self.K)

        self.fc3.bias = torch.nn.Parameter(torch.eye(3).view(3 * self.K))

        # self.fcn3 = FCN(256, 3 * self.K, bn=None, activation=torch.tanh)
        #
        # self.fcn3.fc.bias = torch.nn.Parameter(torch.eye(3).view(3 * self.K))

    def forward(self, x):
        # x: B * C * N
        x = self.pfcn1(x)
        x = self.pfcn2(x)
        x = self.pfcn3(x)
        x = self.max_pool(x)

        # B,C,1 -> B,C
        x = x.squeeze(2)
        x = self.fcn1(x)
        x = self.fcn2(x)
        # x = self.fcn3(x)
        x = self.fc3(x)

        # transform: B * 3 * 3
        transform = x.view(-1, 3, self.K)
        return transform


class FeatureTransformNet(nn.Module):
    '''
        input -> SFCN(64) -> SFCN(128) -> SFCN(1024) -> MaxPool -> FCN(512) -> FCN(256) -> Linear(K*K)
    '''

    def __init__(self, in_channels=3, point_num=1024, K=64):
        super(FeatureTransformNet, self).__init__()
        self.K = K
        self.pfcn1 = PFCN(in_channels, 64)
        self.pfcn2 = PFCN(64, 128)
        self.pfcn3 = PFCN(128, 1024)
        self.max_pool = nn.MaxPool1d(kernel_size=point_num)
        self.fcn1 = FCN(1024, 512)
        self.fcn2 = FCN(512, 256)

        self.fc3 = nn.Linear(256, self.K * self.K)

        self.fc3.bias = torch.nn.Parameter(torch.eye(self.K).view(self.K * self.K))

        # self.fcn3 = FCN(256, self.K * self.K, bn=None, activation=torch.tanh)

        # self.fcn3.fc.bias = torch.nn.Parameter(torch.eye(self.K).view(self.K * self.K))

    def forward(self, x):
        # x: B * C * N
        x = self.pfcn1(x)
        x = self.pfcn2(x)
        x = self.pfcn3(x)
        x = self.max_pool(x)

        # B,C,1 -> B,C
        x = x.squeeze(2)
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fc3(x)

        # transform: B * K * K
        transform = x.view(-1, self.K, self.K)
        return transform


if __name__ == "__main__":
    p_in = torch.randn((5, 3, 1024))

    itn = InputTransformNet(3, 1024, 3)
    trans = itn(p_in)

    print(itn)
    print("input: ", p_in.shape)
    print("output: ", trans.shape)
    print(trans)
    assert trans.shape == torch.Size([5, 3, 3])

    p_in = torch.randn((5, 64, 1024))
    ftn = FeatureTransformNet(64, 1024, 64)
    trans = ftn(p_in)

    print(ftn)
    print("input: ", p_in.shape)
    print("output: ", trans.shape)
    # print(trans)
    assert trans.shape == torch.Size([5, 64, 64])
