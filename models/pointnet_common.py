import torch
import torch.nn as nn
from models.fcn import PFCN, FCN
from models.tnet import InputTransformNet, FeatureTransformNet


class PointNetCommon(nn.Module):
    '''
        common part of point-net
        input -> (T-Net) -> SFCN(64) -> SFCN(64) -> (T-Net) -> SFCN(64) -> SFCN(128) -> SFCN(1024) -> MaxPool -> output
                                                                                          |
                                                                                          v
                                                                                point_wise_feature
    '''

    def __init__(self, in_channels=3, point_num=1024, input_trans=False, feature_trans=False):
        super(PointNetCommon, self).__init__()
        self.input_trans = input_trans
        self.feature_trans = feature_trans

        if self.input_trans:
            self.inputTrans = InputTransformNet(in_channels, point_num, 3)

        self.pfcn1 = PFCN(in_channels, 64)
        self.pfcn2 = PFCN(64, 64)

        if self.feature_trans:
            self.featureTrans = FeatureTransformNet(64, point_num, 64)

        self.pfcn3 = PFCN(64, 64)
        self.pfcn4 = PFCN(64, 128)
        self.pfcn5 = PFCN(128, 1024)

        self.max_pool = nn.MaxPool1d(kernel_size=point_num)

    def forward(self, x):
        # x: B * C * N
        end_points = {}
        if self.input_trans:
            # inTrans: B * C * C
            inTrans = self.inputTrans(x)

            if x.shape[1] > 3:
                # only need transform the xyz part
                xyz = x[:, 0:3, :]
                xyz = xyz.permute(0, 2, 1)
                xyz = torch.matmul(xyz, inTrans)
                xyz = xyz.permute(0, 2, 1)
                x[:, 0:3, :] = xyz
            else:
                x = x.permute(0, 2, 1)
                x = torch.matmul(x, inTrans)
                x = x.permute(0, 2, 1)

            end_points["input_trans"] = inTrans

        x = self.pfcn1(x)
        x = self.pfcn2(x)

        if self.feature_trans:
            fTrans = self.featureTrans(x)
            x = x.permute(0, 2, 1)
            x = torch.matmul(x, fTrans)
            x = x.permute(0, 2, 1)

            end_points["feature_trans"] = fTrans

        x = self.pfcn3(x)
        x = self.pfcn4(x)
        x = self.pfcn5(x)
        point_wise_feature = x

        x = self.max_pool(x)

        # B,C,1 -> B,C
        x = x.squeeze(2)

        return x, point_wise_feature, end_points


if __name__ == "__main__":
    pnc = PointNetCommon(in_channels=3, point_num=1024)

    p_in = torch.randn((5, 3, 1024))
    f_out, pwf, _ = pnc(p_in)

    print(pnc)
    print("input: ", p_in.shape)
    print("feature output: ", f_out.shape)
    print("point-wise feature: ", pwf.shape)

    assert f_out.shape == torch.Size([5, 1024])
    assert pwf.shape == torch.Size([5, 64, 1024])
