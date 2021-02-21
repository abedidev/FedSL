import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):

    def __init__(self, args, dataset=None, idxs=None, idx=None):

        self.idx = idx

        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs),
                                    batch_size=self.args.local_bs, shuffle=True)

        self.iter = args.iter

    def train(self, netFirst, netSecond):
        netFirst.train()
        netSecond.train()
        # train and update
        optimizerFirst = torch.optim.SGD(netFirst.parameters(), lr=self.args.lr, momentum=0.5)
        optimizerSecond = torch.optim.SGD(netSecond.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                images = images.view(-1, self.args.seq_dim, self.args.input_dim)
                imagesFirst = images[:, :int(self.args.seq_dim / 2), :]
                imagesSecond = images[:, int(self.args.seq_dim / 2):, :]

                # First
                optimizerFirst.zero_grad()
                clientOut, clientH = netFirst(imagesFirst)

                # Second
                optimizerSecond.zero_grad()
                labels = labels.clone()
                h = clientH
                # out = clientOut.detach().requires_grad_()
                logits = netSecond(imagesSecond, h)
                loss = self.loss_func(logits, labels.long())
                loss.backward()
                optimizerSecond.step()

                # First
                h_grads = h.grad
                clientH.backward(h_grads)
                optimizerFirst.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return netFirst.state_dict(), netSecond.state_dict(), sum(epoch_loss) / len(epoch_loss)
