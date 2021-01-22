import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from sampling.sampling import mnist_iid, mnist_noniid
from models.UpdateSplit import LocalUpdate

from models.Networks import IRNN_first, IRNN_second
from models.FedAvg import FedAvg
from models.testSplit import test_img
import argparse
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import datetime

import datetime
import time

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--return_counts", type=bool, default=True)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

args = parser.parse_args()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
print('args.device = ' + str(args.device))
args.dataset = 'mnist'
args.seed = 2
torch.manual_seed(args.seed)
args.num_channels = 1
args.iid = True
args.num_classes = 10

args.start = 0
args.round = 2
args.local_ep = 1
# args.num_users = 8
# args.frac = .25
args.num_users = 1
args.frac = 1.

args.local_bs = 8
args.bs = args.local_bs

args.lr = .000001
args.verbose = True
args.seq_dim = 28 * 28  # 28

# load dataset and split users
trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))], )
dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)


def index_dataset(dataset):
    x_train = np.zeros((len(dataset), 1 * 28, 28))
    # x_train = np.zeros((len(dataset), 2 * 28, 28))
    y_train = np.zeros((len(dataset), 2), dtype=np.int64)

    i = 0
    for _, data in enumerate(dataset):
        x_train[i, 0 * 28: 1 * 28, :] = data[0].numpy()
        # x_train[i, 1 * 28: 2 * 28, :] = data[0].numpy()
        y_train[i, 0] = data[1]
        y_train[i, 1] = i
        i = i + 1

    dataset_train = TensorDataset(
        torch.Tensor(x_train),
        torch.Tensor(y_train).type(torch.LongTensor),
    )

    return dataset_train


# dataset_train = index_dataset(dataset_train)
# dataset_test = index_dataset(dataset_test)

if args.iid:
    dict_users = mnist_iid(dataset_train, args.num_users)
else:
    dict_users = mnist_noniid(dataset_train, args.num_users)

print(dataset_train[0][0].shape)

# build model(s)

args.input_dim = 1
args.hidden_dim = 64
args.layer_dim = 1
args.output_dim = 10

# net_glob = LSTM(args=args).to(args.device)
# net_glob_first = modelFirstGRU(args.input_dim, args.hidden_dim, args.layer_dim, args.output_dim, args.local_bs, args.device).to(args.device)
# net_glob_second = modelSecondGRU(args.input_dim, args.hidden_dim, args.layer_dim, args.output_dim, args.local_bs, args.device).to(args.device)

net_glob_first = IRNN_first(args.input_dim, args.hidden_dim, args.output_dim).to(args.device)
net_glob_second = IRNN_second(args.input_dim, args.hidden_dim, args.output_dim).to(args.device)

net_glob_first.train()
net_glob_second.train()

# copy weights
w_glob_first = net_glob_first.state_dict()
w_glob_second = net_glob_second.state_dict()

# training
loss_train = []
loss_test = []
accuracy_train = []
accuracy_test = []

STA = []
END = []

cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []
args.iter = 0

# for iter in range(args.round):
for iter in range(args.start, args.round):

    args.iter = iter
    w_first_locals, w_second_locals, loss_locals = [], [], []
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    # idxs_users = np.array([0, 3], dtype=np.int)
    # idxs_users = np.array([3], dtype=np.int)

    STA.append(time.time())

    for idx in idxs_users:
        print('user_index = ' + str(idx))
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], idx=idx)
        w_first, w_second, loss = local.train(netFirst=copy.deepcopy(net_glob_first).to(args.device),
                                              netSecond=copy.deepcopy(net_glob_second).to(args.device))
        w_first_locals.append(copy.deepcopy(w_first))
        w_second_locals.append(copy.deepcopy(w_second))
        loss_locals.append(copy.deepcopy(loss))

    # update global weights
    w_first_glob = FedAvg(w_first_locals)
    w_second_glob = FedAvg(w_second_locals)

    # copy weight to net_glob
    net_glob_first.load_state_dict(w_first_glob)
    net_glob_second.load_state_dict(w_second_glob)

    END.append(time.time())

    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
    loss_train.append(loss_avg)

    acc_test, _loss_test_ = test_img(net_glob_first, net_glob_second, dataset_test, args)
    loss_test.append(_loss_test_)
    accuracy_test.append(acc_test)
    print('------------------------------------------------------->\t' + str(acc_test))

    np.save('./newSave/' + 'split_loss_train_' + str(iter) + '.npy', np.array(loss_train))
    np.save('./newSave/' + 'split_loss_test_' + str(iter) + '.npy', np.array(loss_test))
    np.save('./newSave/' + 'split_accuracy_test_' + str(iter) + '.npy', np.array(accuracy_test))
    np.save('./newSave/' + 'split_STA_' + str(iter) + '.npy', np.array(STA))
    np.save('./newSave/' + 'split_END_' + str(iter) + '.npy', np.array(END))

net_glob_first.eval()
net_glob_second.eval()
acc_train, _loss_train_ = test_img(net_glob_first, net_glob_second, dataset_train, args)
acc_test, _loss_test_ = test_img(net_glob_first, net_glob_second, dataset_test, args)
print("Training accuracy: {:.2f}".format(acc_train))
print("Testing accuracy: {:.2f}".format(acc_test))

np.save('./save/' + str(datetime.datetime.now()) + '_split_loss_train.npy', np.array(loss_train))
np.save('./save/' + str(datetime.datetime.now()) + '_split_loss_test.npy', np.array(loss_test))
np.save('./save/' + str(datetime.datetime.now()) + '_split_accuracy_test.npy', np.array(accuracy_test))

np.save('./save/' + str(datetime.datetime.now()) + '_split_STA.npy', np.array(STA))
np.save('./save/' + str(datetime.datetime.now()) + 'split_END.npy', np.array(END))
