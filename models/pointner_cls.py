import torch
import torch.nn as nn
from models.fcn import PFCN, FCN
from models.pointnet_common import PointNetCommon


class PointNetCLS(nn.Module):
    '''
        point-net for classification
        PointNetCommon output(1024) -> FCN(512) -> FCN(256) -> FCN(K)
    '''

    def __init__(self, in_channels=3, point_num=1024, K=40, input_trans=False, feature_trans=False):
        super(PointNetCLS, self).__init__()
        self.pn_common = PointNetCommon(in_channels, point_num, input_trans, feature_trans)

        self.fcn1 = FCN(1024, 512)
        self.fcn2 = FCN(512, 256)
        self.fcn3 = FCN(256, K, bn=False, dropout=False, activation=None)

    def forward(self, x):
        x, f, end_points = self.pn_common(x)

        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)

        return x, end_points


def LossCLS(outputs, labels, model, device, end_points):
    criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
    classify_loss = criterion(outputs, labels)

    # Enforce the transformation as orthogonal matrix
    mat_loss = 0
    if 'input_trans' in end_points:
        tran = end_points['input_trans']

        diff = torch.mean(torch.matmul(tran, tran.permute(0, 2, 1)), 0)
        mat_loss += nn.MSELoss()(diff, torch.eye(3, device=device))

    if 'feature_trans' in end_points:
        tran = end_points['feature_trans']

        diff = torch.mean(torch.matmul(tran, tran.permute(0, 2, 1)), 0)
        mat_loss += nn.MSELoss()(diff, torch.eye(tran.shape[1], device=device))

    loss = classify_loss + mat_loss * 0.001

    return loss


if __name__ == "__main__":
    p_in = torch.randn((5, 3, 1024))
    pncls = PointNetCLS(in_channels=3, point_num=1024, K=40, input_trans=True, feature_trans=True)
    output, _ = pncls(p_in)

    print(pncls)
    print("input: ", p_in.shape)
    print("output: ", output.shape)

    assert output.shape == torch.Size([5, 40])
