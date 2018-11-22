import argparse
import numpy as np

from torch.utils.data import DataLoader

from utils.log_helper import *
from models.pointnet_common import *
from classification.dataloader import ModelNet40, ToTensor
from utils.save_load import *

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--datadir', dest='datadir',
                    default="G:/PycharmProjects/pointnet-torch/data/modelnet40_ply_hdf5_2048",
                    help='data directory ')
parser.add_argument('--save_dir', dest='save_dir', default='./checkpoints',
                    help='directory to save models')
parser.add_argument('--resume', default='./checkpoints/model_best.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu_id', dest='gpu_id', type=int,
                    default=0, help='which gpu to choose')
parser.add_argument('--num_points', default=1024, type=int, metavar='N',
                    help='number of points of cloud')
parser.add_argument('--input_trans', default=0, type=int,
                    help='add transform net on input cloud')
parser.add_argument('--feature_trans', default=0, type=int,
                    help='add transform net on feature')

args = parser.parse_args()


def build_dataloader(path, batch_size):
    logger = logging.getLogger('global')
    logger.info("Build dataloader...")

    test_files = os.path.join(path, "test_files.txt")

    test_dataset = ModelNet40(list_filename=test_files,
                              transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    logger.info('Build dataloader done')

    return test_loader


def main():
    print(args)
    init_log('global', args.save_dir, logging.INFO)
    logger = logging.getLogger('global')

    device_name = "cuda:{}".format(args.gpu_id)
    device = torch.device(device_name)

    test_loader = build_dataloader(args.datadir, args.batch_size)
    model = PointNetCLS(input_trans=args.input_trans, feature_trans=args.feature_trans)
    model.to(device)
    logger.info(model)

    assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
    model = load_checkpoint(model, args.resume)

    test_one_epoch(test_loader, model, device)


def test_one_epoch(test_loader, model, device):
    logger = logging.getLogger('global')
    model.eval()

    total_seen = 0
    loss_sum = 0.0
    total_correct = 0.0
    num_batches = 0.0
    class_correct = np.zeros((40))
    class_total = np.zeros((40))
    for i_batch, sample_batched in enumerate(test_loader):
        points = sample_batched['points']
        labels = sample_batched['labels']

        points = points.permute(0, 2, 1).type(torch.FloatTensor)
        labels = labels.type(torch.LongTensor)

        points = points.to(device)
        labels = labels.to(device)

        outputs = model(points)

        _, preds = torch.max(outputs, 1)
        loss = LossCLS(outputs, labels, model, device)

        num_batches = num_batches + 1
        total_seen += sample_batched['labels'].shape[0]
        loss_sum += loss.item()
        total_correct += torch.sum(preds == labels).item()

        labels_numpy = labels.cpu().numpy()
        preds_numpy = preds.cpu().numpy()
        correct_numpy = preds_numpy == labels_numpy
        for i in range(40):
            mask = labels_numpy == i
            class_total[i] += np.sum(mask)
            class_correct[i] += np.sum(mask * correct_numpy)

    epoch_loss = loss_sum / num_batches
    epoch_acc = total_correct / total_seen
    class_ave_acc = np.mean(class_correct / class_total)
    logger.info('Test Loss: {:.4f} Acc: {:.4f}, Class_ave_acc: {:.4f}'.format(epoch_loss, epoch_acc, class_ave_acc))


if __name__ == "__main__":
    main()
