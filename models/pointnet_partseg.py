import torch
import torch.nn as nn
from models.fcn import PFCN, FCN
from models.tnet import InputTransformNet, FeatureTransformNet
import hiddenlayer as hl


class PointNetPartSeg(nn.Module):
    '''
        point-net for part segmentation
        input -> (T-Net) -> SFCN(64) -> SFCN(128) -> SFCN(128)-> (T-Net) -> SFCN(512) -> SFCN(2048) -> MaxPool -> FCN(256)
                              |            |             |                     |              |           |          |
                              v            v             v                     v              v           v          v
                             out1         out2          out3                  out4           out5      out_max    FCN(256)
                              |            |             |                     |              |           |          |
                              ----------------------------------------------------------------------------|          v
                                                                                                          v     FCN(cat_num)
                                                         SFCN(part_num) <- SFCN(128) <- SFCN(256) <- SFCN(256)       |
                                                           |                                                         |
                                                           v                                                         v
                                                         seg_out                                                  cls_out
    '''

    def __init__(self, in_channels=3, point_num=1024, cat_num=16, part_num=3, input_trans=False, feature_trans=False):
        '''
        :param in_channels: input point channel number
        :param point_num: point number of cloud
        :param cat_num: category number
        :param part_num: part number
        :param input_trans: bool
        :param feature_trans: bool
        '''
        super(PointNetPartSeg, self).__init__()

        self.input_trans = input_trans
        self.feature_trans = feature_trans
        self.cat_num = cat_num

        # common part
        if self.input_trans:
            self.inputTrans = InputTransformNet(in_channels, point_num, 3)

        self.pfcn1 = PFCN(in_channels, 64)
        self.pfcn2 = PFCN(64, 128)
        self.pfcn3 = PFCN(128, 128)

        if self.feature_trans:
            self.featureTrans = FeatureTransformNet(128, point_num, 128)

        self.pfcn4 = PFCN(128, 512)
        self.pfcn5 = PFCN(512, 2048)

        self.max_pool = nn.MaxPool1d(kernel_size=point_num)

        # classification
        self.fcn1 = FCN(2048, 256)
        self.fcn2 = FCN(256, 256)
        self.fcn3 = FCN(256, cat_num, bn=False, dropout=False, activation=None)

        # segmentation
        self.pfcn6 = PFCN(64 + 128 + 128 + 512 + 2048 + 2048 + cat_num, 256)
        self.pfcn7 = PFCN(256, 256)
        self.pfcn8 = PFCN(256, 128)
        self.pfcn9 = PFCN(128, part_num, bn=False, activation=None)

    def forward(self, x, labels):
        '''
        :param x: B * C * N ( B: batch_size, C: channel number, N: point number)
        :param labels: B * 1
        :return:
        '''
        # common part
        end_points = {}
        if self.input_trans:
            # inTrans: B * 3 * 3
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

        out1 = self.pfcn1(x)  # B*64*N
        out2 = self.pfcn2(out1)  # B*128*N
        out3 = self.pfcn3(out2)  # B*128*N

        if self.feature_trans:
            fTrans = self.featureTrans(out3)
            x = out3.permute(0, 2, 1)
            x = torch.matmul(x, fTrans)
            x = x.permute(0, 2, 1)

            end_points["feature_trans"] = fTrans

        out4 = self.pfcn4(x)  # B*512*N
        out5 = self.pfcn5(out4)  # B*2048*N

        out_max = self.max_pool(out5)  # B*2048*1

        # classification
        x = out_max.squeeze(2)  # B * 2048 * 1 --> B * 2048
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)

        # segmentation
        # convert label to one-hot representation
        def convert_label_to_one_hot(self, labels):
            label_one_hot = torch.zeros((labels.shape[0], self.cat_num), device=labels.device)
            for idx in range(labels.shape[0]):
                label_one_hot[idx, labels[idx]] = 1
            return label_one_hot

        labels = convert_label_to_one_hot(self, labels)
        labels = labels.unsqueeze(-1)
        out_max = torch.cat((out_max, labels), 1)  # B * (2048+cat_num) * 1

        # tile op
        ones = torch.ones((1, out5.shape[2]), device=labels.device)  # ones: 1 * N
        out_max = out_max.mul(ones)  # out_max: B * (2048+cat_num) * N

        # out_max = out_max.repeat(1,1,out5.shape[2]) # copy data, similar to numpy.tile.

        x2 = torch.cat((out_max, out1, out2, out3, out4, out5),
                       1)  # B * (64 + 128 + 128 + 512 + 2048 + 2048 + cat_num) * N

        x2 = self.pfcn6(x2)
        x2 = self.pfcn7(x2)
        x2 = self.pfcn8(x2)
        x2 = self.pfcn9(x2)

        return x, x2, end_points


def LossPartSeg(l_pred, seg_pred, label, seg, weight, end_points, device):
    '''
    :param l_pred: B * cat_num
    :param seg_pred: B * part_num * N
    :param label: B
    :param seg: B * N (each point has a seg id)
    :param weight: the weight of seg_loss
    :param end_points: end_points
    :param device: device
    :return:
    '''
    criterion = nn.CrossEntropyLoss()
    label_loss = criterion(l_pred, label)

    # size of seg_pred is B * part_num * N
    # size of seg is B * N
    seg_loss = criterion(seg_pred, seg)

    # Enforce the transformation as orthogonal matrix
    mat_diff_loss = 0
    if 'input_trans' in end_points:
        tran = end_points['input_trans']

        diff = torch.mean(torch.matmul(tran, tran.permute(0, 2, 1)), 0)
        mat_diff_loss += nn.MSELoss()(diff, torch.eye(3, device=device))

    if 'feature_trans' in end_points:
        tran = end_points['feature_trans']

        diff = torch.mean(torch.matmul(tran, tran.permute(0, 2, 1)), 0)
        mat_diff_loss += nn.MSELoss()(diff, torch.eye(tran.shape[1], device=device))

    total_loss = weight * seg_loss + (1 - weight) * label_loss + mat_diff_loss * 1e-3

    return total_loss, label_loss, seg_loss


if __name__ == "__main__":
    batch_size = 5
    in_channels = 6
    point_num = 2049
    cat_num = 12
    part_num = 50

    points = torch.randn((batch_size, in_channels, point_num))
    label = torch.randint(0, 2, (batch_size,)).type(torch.LongTensor)
    seg = torch.randint(0, part_num, (batch_size, point_num)).type(torch.LongTensor)

    pnseg = PointNetPartSeg(in_channels=in_channels, point_num=point_num, cat_num=cat_num, part_num=part_num,
                            input_trans=True, feature_trans=True)
    label_pre, seg_pre, end_points = pnseg(points, label)

    total_loss, label_loss, seg_loss = LossPartSeg(label_pre, seg_pre, label, seg, 1, end_points, device="cpu")

    # hl_graph = hl.build_graph(pnseg, (p_in, l_in))
    # hl_graph.save("graph.png", format="png")

    print(pnseg)
    print("input: ", points.shape)
    print("input_labels: ", label.shape)
    print("cls_out: ", label_pre.shape)
    print("seg_out: ", seg_pre.shape)
    print("total_loss: ", total_loss)
    print("label_loss: ", label_loss)
    print("seg_loss: ", seg_loss)

    assert label_pre.shape == torch.Size([batch_size, cat_num])
    assert seg_pre.shape == torch.Size([batch_size, part_num, point_num])
