

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


def test_img(net_glob_first, net_glob_second, datatest, args):
    net_glob_first.eval()
    net_glob_second.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data = data.to(args.device)
        target = target.to(args.device)

        data = data.view(-1, args.seq_dim, args.input_dim)
        dataFirst = data[:, :int(args.seq_dim / 2), :]
        dataSecond = data[:, int(args.seq_dim / 2):, :]

        # target = target[:, 0]

        # if args.gpu != -1:
        #     data, target = data.cuda(), target.cuda()
        # xLengths = [28 for i in range(data.shape[0])]

        tempOut, h_split_layer_tensor = net_glob_first(dataFirst)
        # h_split_layer_tensor = torch.zeros(h_split_layer_tensor.shape)
        outputs = net_glob_second(dataSecond, h_split_layer_tensor)
        # tempOut, h_split_layer_tensor = net_glob_first(data)
        # outputs = net_glob_second(data, h_split_layer_tensor)
        # log_probs = net_g(data, xLengths)
        # sum up batch loss
        test_loss += F.cross_entropy(outputs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return np.asscalar(accuracy.numpy()), test_loss
