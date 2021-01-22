import torch
from torch import nn


class IRNN_first(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IRNN_first, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True, bias=True)

        # self.output_weights = nn.Linear(hidden_size, output_size)

        # Parameters initialization
        self.rnn.state_dict()['weight_hh_l0'].copy_(torch.eye(hidden_size))
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_hh_l0.data.fill_(0)
        self.rnn.state_dict()['weight_ih_l0'].copy_(torch.randn(hidden_size, input_size) / hidden_size)

    def forward(self, inp):
        _, hnn = self.rnn(inp)
        # out = self.output_weights(hnn[0])
        # return out
        return _, hnn


class IRNN_second(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IRNN_second, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True, bias=True)
        self.output_weights = nn.Linear(hidden_size, output_size)

        # Parameters initialization
        self.rnn.state_dict()['weight_hh_l0'].copy_(torch.eye(hidden_size))
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_hh_l0.data.fill_(0)
        self.rnn.state_dict()['weight_ih_l0'].copy_(torch.randn(hidden_size, input_size) / hidden_size)

    def forward(self, inp, hPrevious):
        _, hnn = self.rnn(inp, hPrevious)
        out = self.output_weights(hnn[0])
        return out


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = args.hidden_dim

        # Number of hidden layers
        self.layer_dim = args.layer_dim

        # Building your LSTM
        # batch_first = True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.lstm = nn.LSTM(args.input_dim, args.hidden_dim, args.layer_dim, batch_first=True, bidirectional=True)

        # Readout layer
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(args.hidden_dim * 2, args.output_dim)

    def forward(self, x, xLengths):
        # Initialize hidden state with zeros
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim)

        # Initialize cell state
        # c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim)

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        x = torch.nn.utils.rnn.pack_padded_sequence(x, xLengths, batch_first=True)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


class LSTMSF(nn.Module):

    def __init__(self, args):
        super(LSTMSF, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.layer_dim = args.layer_dim
        self.batch_size = args.local_bs

        # Building your LSTM
        # batch_first = True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.lstm = nn.LSTM(args.input_dim, args.hidden_dim, args.layer_dim, batch_first=True, bidirectional=True)

        # Readout layer
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(args.hidden_dim * 2, args.output_dim)

        self.h = self.init_hc()
        self.c = self.init_hc()

    def init_hc(self):
        return torch.zeros(self.layer_dim * 2, self.batch_size, self.hidden_dim)

    def forward(self, x, xLengths):
        # h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(args.device)
        # c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(args.device)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, xLengths, batch_first=True)

        # out, (hn, cn) = self.lstm(x, (h0, c0))
        out, (self.h, self.c) = self.lstm(x, (self.h.detach(), self.c.detach()))

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


class modelFirstGRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, batch_size, device):
        super(modelFirstGRU, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.batch_size = batch_size

        self.h = self.init_h(self.batch_size)

        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)

    def init_h(self, bs):
        return torch.zeros(self.layer_dim, bs, self.hidden_dim)

    def forward(self, x):
        self.h = self.init_h(x.size(0)).to(self.device)
        out, hn = self.gru(x, self.h)

        return out, hn
        # return out, self.init_h(self.batch_size).requires_grad_()


class modelSecondGRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, batch_size, device):
        super(modelSecondGRU, self).__init__()
        self.device = device
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hPrevious):
        out, hn = self.gru(x, hPrevious)
        out = self.fc(out[:, -1, :])
        return out


