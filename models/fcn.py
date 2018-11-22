import torch
import torch.nn as nn


# stat:not support Conv1d,BatchNorm1d
# from torchstat import stat

class PFCN(nn.Module):
    '''
        Point-wise FCN
        shared fully connected network implementation using conv1d
    '''

    def __init__(self, in_channels, out_channels, bn=True, activation=torch.relu):
        super(PFCN, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=1, stride=1, padding=0, bias=True)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.activation = activation

    def forward(self, x):
        # input shape: B * C_in * N
        # output shape: B * C_out * N
        x = self.conv(x)
        if self.bn != None:
            x = self.bn(x)
        if self.activation != None:
            x = self.activation(x)
        return x


class FCN(nn.Module):
    '''
        fully connected network
    '''

    def __init__(self, in_channels, out_channels, bn=True, dropout=True, activation=torch.relu):
        super(FCN, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels)
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(p=0.7)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm1d(out_channels)

        self.activation = activation

    def forward(self, x):
        x = self.fc(x)
        if self.bn != None:
            x = self.bn(x)
        if self.dropout != None:
            x = self.dropout(x)
        if self.activation != None:
            x = self.activation(x)
        return x


if __name__ == "__main__":
    p_in = torch.randn((5, 3, 1024))

    sfcn = PFCN(3, 10)
    f_out = sfcn(p_in)

    print(sfcn)
    print("input: ", p_in.shape)
    print("output: ", f_out.shape)
    assert f_out.shape == torch.Size([5, 10, 1024])

    p_in = torch.randn((5, 3))
    fcn = FCN(3, 10)
    f_out = fcn(p_in)

    print(" ")
    print(fcn)
    print("input: ", p_in.shape)
    print("output: ", f_out.shape)
    assert f_out.shape == torch.Size([5, 10])
