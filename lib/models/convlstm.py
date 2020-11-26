import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from lib.models.i3d.i3d import InceptionI3d
from lib.models.i3d.i3d import Unit3D

# Adapted from https://github.com/automan000/Convolution_LSTM_PyTorch/blob/master/convolution_lstm.py
class ConvLSTMCell(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 input_dropout_rate=0.0,
                 reccurent_drouput_rate=0.0):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.input_dropout_rate = np.clip(input_dropout_rate, 0.0, 1.0)
        self.reccurent_drouput_rate = np.clip(reccurent_drouput_rate, 0.0, 1.0)

        self.padding = int((kernel_size - 1) / 2)

        self.input_dropout = nn.Dropout(
            p=self.input_dropout_rate) if 0.0 < self.input_dropout_rate < 1.0 else nn.Identity()
        self.reccurent_drouput = nn.Dropout(
            p=self.reccurent_drouput_rate) if 0.0 < self.input_dropout_rate < 1.0 else nn.Identity()

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        x = self.input_dropout(x)
        h = self.reccurent_drouput(h)
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape, use_cuda=False):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        if use_cuda:
            self.Wci = self.Wci.cuda()
            self.Wcf = self.Wcf.cuda()
            self.Wco = self.Wco.cuda()
        h = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]))
        c = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]))
        if use_cuda:
            h, c = h.cuda(), c.cuda()
        return (h, c)


class _ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 batch_first=False,
                 input_dropout_rate=0.0,
                 reccurent_dropout_rate=0.0):
        super(_ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.batch_first = batch_first
        if not isinstance(input_dropout_rate, list):
            self.input_dropout_rate = [input_dropout_rate] * self.num_layers
        if not isinstance(reccurent_dropout_rate, list):
            self.reccurent_dropout_rate = [reccurent_dropout_rate] * self.num_layers

        self._all_layers = nn.ModuleList()
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i],
                                self.hidden_channels[i],
                                self.kernel_size,
                                self.input_dropout_rate[i],
                                self.reccurent_dropout_rate[i])
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input, hidden_state=None):
        """
        Partially adapted code from
        https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input = input.permute(1, 0, 2, 3, 4)

        internal_state = []
        outputs = []
        n_steps = input.size(1)
        for t in range(n_steps):
            x = input[:, t, :, :, :]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if t == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width), use_cuda=input.is_cuda)
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            outputs.append(x)
        outputs = torch.stack(outputs, dim=1)       # outputs.shape == (b, t, c, h, w)

        return outputs, (x, new_c)

class _Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(_Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = 0#self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

class _SqueezeDim2(nn.Module):
    def __init__(self):
        super(_SqueezeDim2, self).__init__()

    def forward(self, x):
        return x.squeeze(dim=2)

class _GlobalAvgPool(nn.Module):
    def __init__(self):
        super(_GlobalAvgPool, self).__init__()

    def forward(self, x):
        # x.shape == (batch_size, c, h, w)
        return x.mean(dim=(2, 3))

class ConvLSTM(nn.Module):
    def __init__(self, args):
        super(ConvLSTM, self).__init__()
        self.steps = args.steps
        self.num_classes = args.num_classes

        if args.feature_extractor == 'I3D':
            self.feature_extractor = InceptionI3d(final_endpoint='MaxPool3d_3a_3x3')
            self.feature_extractor.build()
            pretrained_state_dict = torch.load(os.path.join('lib', 'models', 'i3d', 'rgb_imagenet.pt'))
            actual_state_dict = self.feature_extractor.state_dict()
            # filter out unnecessary keys
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in actual_state_dict}
            # overwrite entries in the existing state dict
            actual_state_dict.update(pretrained_state_dict)
            # load the new state dict
            self.feature_extractor.load_state_dict(actual_state_dict)
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            raise Exception('Wrong --feature_extractor option. ' + args.feature_extractor + ' unknown')

        self.conv3d_bottleneck = nn.Sequential(
            _Unit3D(in_channels=192, output_channels=192, kernel_shape=[5, 1, 1], padding=0),
            _SqueezeDim2(),
        )

        self.conv_lstm = _ConvLSTM(input_channels=192,
                                   hidden_channels=[64, 32, 64],
                                   kernel_size=3,
                                   batch_first=True,
                                   input_dropout_rate=args.dropout,
                                   reccurent_dropout_rate=args.dropout)
        self.classifier = nn.Sequential(
            _GlobalAvgPool(),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, x):
        # x.shape == (batch_size, steps, C, chunk_size, H, W)
        scores = []     # shape (batch_size, steps, num_classes)
        features_x = []     # shape (batch_size, steps, CC, HH, WW)

        for step in range(self.steps):
            x_t = x[:, step]
            appo = self.feature_extractor.extract_features(x_t)
            feat_x = self.conv3d_bottleneck(appo)     # feat_x.shape == (batch_size, CC, HH, WW)
            features_x.append(feat_x)
        features_x = torch.stack(features_x, dim=1)
        features_x = self.conv_lstm(features_x)
        for step in range(self.steps):
            x_t = features_x[:, step]
            out = self.classifier(x_t)
            scores.append(out)

        scores = torch.stack(scores, dim=1)     # scores.shape == (batch_size, steps, num_classes)
        return scores