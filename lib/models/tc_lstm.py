import torch
import torch.nn as nn

import math

from .feature_extractor import build_feature_extractor

class DilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilatation=1, pad_start=True):
        '''
        :param in_channels: shape of feature dimension. In our case it is the dimension of the feature vector
        :param out_channels: number of filters
        :param kernel_size: this conceptually means how many timesteps to look at (the width of the "sliding window")
        :param dilatation: how many dilatation to apply. dilatation == 1 means no dilatation
        :param pad_start: whether to apply pad at the start of the input tensor or not.  If applied, the
                            length of the output timesequence will be the same as input
        '''
        super(DilatedCausalConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilatation)
        self.pad_start = pad_start
        if pad_start:
            self.amount_pad = (kernel_size - 1) * dilatation

    def forward(self, x):
        '''
        In order to make the convolution causal, it is necessary to add zero-padding at the start of the input.
        The amount of padding is dependent on kernel_size and dilatation. Notice that a direct consequence
        of adding causal padding is that the output timesequence_length is the same as input.
        In a nutshell, we have feature vectors stacked along time and there is a sliding window
        of length kernel_size that performs convolution along time for all the time sequence.
        :param x: Input tensor of shape (batch_size, feat_vect_dim, timesequence_length)
        :return: tensor of shape (batch_size, num_filters, timesequence_length)
        '''
        if self.pad_start:
            tensor_pad = torch.zeros(x.shape[0], x.shape[1], self.amount_pad)
            x = torch.cat((tensor_pad, x), dim=2)
        x = self.conv(x)
        return x

class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
        self.conv1 = nn.Sequential(
            # in this case the convolution is not dilated; it takes two(== kernel_size) time-consecutive feature
            # vectors and returns a new one
            DilatedCausalConv1d(args.feat_vect_dim, args.feat_vect_dim, 2, 1, pad_start=False),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        # x.shape == (batch_size, feat_vect_dim, timesequence_lenght)
        out1 = self.conv1(x)
        # add residual connection
        out1 += x.mean(dim=2, keepdim=True)
        return out1

class TCLSTM(nn.Module):
    '''
    This model has a causal convolution before lstm.
    Notice that the causal convolution inserted here is not dilated(check dilatation parameter), it only takes
    two(check kernel_size and pad_star parameters) consecutive feature vectors and returns a new one that will
    be fed to the lstm, so predictions are made every 0.5 s
    '''
    def __init__(self, args):
        super(TCLSTM, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps

        self.feature_extractor = build_feature_extractor(args)

        self.tcn = TCN(args)

        self.lstm = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.drop = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        scores = torch.zeros(x.shape[0], x.shape[1] // 2, self.num_classes, dtype=x.dtype)

        for step in range(0, self.enc_steps, 2):
            x_t = x[:, step]
            x_tplus1 = x[:, step+1]
            out_t = self.feature_extractor(x_t, torch.zeros(1))  # second input is optical flow, in our case will not be used
            out_tplus1 = self.feature_extractor(x_tplus1, torch.zeros(1))

            out_t = out_t.unsqueeze(2)
            out_tplus1 = out_tplus1.unsqueeze(2)
            out = torch.cat((out_t, out_tplus1), dim=2)

            out = self.tcn(out).squeeze(2)
            # out.shape == (batch_size, feat_vect_dim)

            h_n, c_n = self.lstm(out, (h_n, c_n))
            out = self.classifier(self.drop(h_n))  # out.shape == (batch_size, num_classes)

            scores[:, step, :] = out
        return scores

    def step(self, camera_input, h_n, c_n):
        out = self.feature_extractor(camera_input, torch.zeros(1))
        out = self.tcn(out)
        h_n, c_n = self.lstm(out, (h_n, c_n))
        out = self.classifier(h_n)
        return out, h_n, c_n