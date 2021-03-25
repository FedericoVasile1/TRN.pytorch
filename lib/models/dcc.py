import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class DilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilatation=1, pad_start=True):
        '''
        Dilated causual convolution.
        Dilated means that the filter at each moment do not look at close spatial
        pixels, instead it does a jump equal to dilatation_value between a pixel and the next one to look at. This
        allows larger receptive field.
        Causal means that at each moment the convolution can not look information in the future, this is
        done by adding padding before the input at the first timestep, in this way the convolution at the first
        timestep will look at padding+input1, at the second timestep will look at input1+input2, and so on..
        :param in_channels: shape of feature dimension. In our case it is the dimension of the feature vector
        :param out_channels: number of filters
        :param kernel_size: this conceptually means how many feature vectors to look at
                            the same time (the width of the "window")
        :param dilatation: how many dilatation to apply. dilatation == 1 means no dilatation
        :param pad_start: whether to apply pad at the start of the input tensor or not.  If applied, the
                            length of the output timesequence will be the same as input
        '''
        super(DilatedCausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilatation = dilatation
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

    def step(self, x):
        assert x.shape[-1] == self.kernel_size
        x = self.conv(x)
        return x

class DCCBlock(nn.Module):
    '''
    A dilated causal convolution block made up of convolution and residual connection.
    A linear transformation must be done before the residual connection in case of different dimensions.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dilatation, dropout=0.1, pad_start=True):
        super(DCCBlock, self).__init__()
        self.conv = DilatedCausalConv1d(in_channels, out_channels, kernel_size, dilatation, pad_start)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lin_transf = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """
        :param x: torch tensor of shape (batch_size, in_channels, timesequence_length)
        """
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        x = self.lin_transf(x.permute(0, 2, 1)).permute(0, 2, 1)
        out += x
        return out

    def step(self, x):
        """
        :param x: torch tensor of shape (batch_size, in_channels, timesequence_length <= kernel_size)
        :return: torch tensor of shape (batch_size, out_channels, 1)
        """
        if x.shape[-1] > self.conv.kernel_size:
            raise Exception('The step function applies a single convolution operation without moving the filter'
                            'through time, so the timesequence lenght of the input must be at most equal to '
                            'the kernel size of the convolutional filter')

        amount_pad = (self.conv.kernel_size - x.shape[-1]) * self.conv.dilatation
        if amount_pad > 0:
            tensor_pad = torch.zeros(x.shape[0], x.shape[1], amount_pad).to(x.device)
            x = torch.cat((tensor_pad, x), dim=2)

        out = self.conv.step(x)
        out = self.relu(out)
        x = x[:, :, -1:]
        x = self.lin_transf(x.permute(0, 2, 1)).permute(0, 2, 1)
        out += x
        return out

class DCCModel(nn.Module):
    def __init__(self, args):
        self.num_layers = args.num_layers
        self.kernel_sizes = args.kernel_sizes.split(',')
        self.dilatation_rates = args.dilatation_rates.split(',')
        self.num_filters = args.num_filters.split(',')
        # convert elements from string to int
        self.kernel_sizes = list(map(int, self.kernel_sizes))
        self.dilatation_rates = list(map(int, self.dilatation_rates))
        self.num_filters = list(map(int, self.num_filters))

        assert self.num_layers == len(self.kernel_sizes) == len(self.dilatation_rates) == len(self.num_filters)

        self.feature_extractor = build_feature_extractor(args)

        self.dcc_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            in_channels = self.feature_extractor.fusion_size if i==0 else self.num_filters[i-1]
            self.dcc_blocks.append(DCCBlock(in_channels,
                                            self.num_filters[i],
                                            self.kernel_sizes[i],
                                            self.dilatation_rates[i],
                                            args.dropout))

        self.classifier = nn.Linear(self.num_filters[-1], args.num_classes)

    def forward(self, x):
        # x.shape == (batch_size, timesequence_length, in_channels)
        transf_x = torch.zeros(x.shape[0], x.shape[1], self.feature_extractor.fusion_size).to(dtype=x.dtype,
                                                                                              device=x.device)

        for step in range(x.shape[1]):
            transf_x[:, step] = self.feature_extractor(x[:, step])

        out = transf_x.permute(0, 2, 1)
        for l in self.dcc_blocks:
            out = l(out)
        out = self.classifier(out.permute(0, 2, 1))
        return out

    def step(self, x):
        pass