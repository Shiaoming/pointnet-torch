import torch
import torch.nn as nn
from models.fcn import PFCN, FCN
from models.pointnet_common import PointNetCommon


class PointNetSemSeg(nn.Module):
    '''
        point-net for classification
        PointNetCommon output(1024) -> FCN(256) -> FCN(128) -|
                |                                            |--> SFCN(512) -> SFCN(256) -> SFCN(K) -> output
                -> point_wise_feature(1024) -----------------|
    '''

    def __init__(self, in_channels=9, point_num=1024, K=13):
        super(PointNetSemSeg, self).__init__()
        self.pn_common = PointNetCommon(in_channels, point_num, False, False)

        self.fcn1 = FCN(1024, 256)
        self.fcn2 = FCN(256, 128)

        self.pfcn1 = PFCN(1024 + 128, 512)
        self.pfcn2 = PFCN(512, 256)
        self.pfcn3 = PFCN(256, K)

    def forward(self, x):
        # x: B*C*N
        x, f, _ = self.pn_common(x)
        # x: B*1024
        # f: B*1024*N

        x = self.fcn1(x)
        x = self.fcn2(x)

        # x: B*128
        # tile op
        x = x.unsqueeze(-1) # x: B * 128 * 1
        x = x.repeat(1, 1, f.shape[2])  # copy data, similar to numpy.tile.
        # ones = torch.ones((1, f.shape[2]), device=x.device)  # ones: 1 * N
        # x = x.mul(ones)  # x: B * 128 * N

        x = torch.cat((x,f),1)

        x = self.pfcn1(x)
        x = self.pfcn2(x)
        x = self.pfcn3(x)

        return x # x: B*K*N

def LossSemSeg(pred,seg):
    criterion = nn.CrossEntropyLoss()

    # size of seg_pred is B * K * N
    # size of seg is B * N
    loss = criterion(pred, seg)

    return loss

if __name__=="__main__":
    batch_size = 45
    in_channels = 9
    point_num = 768
    K = 15

    points = torch.randn((batch_size, in_channels, point_num))
    seg = torch.randint(0, K, (batch_size, point_num)).type(torch.LongTensor)

    model = PointNetSemSeg(in_channels,point_num,K)

    output = model(points)

    print(model)
    print("input: ", points.shape)
    print("seg: ", seg.shape)
    print("output: ", output.shape)

    assert output.shape == torch.Size([batch_size, K, point_num])
