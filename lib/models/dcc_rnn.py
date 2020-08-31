import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor
from .rnn import MyGRUCell

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
        :param kernel_size: this conceptually means how many timesteps to look at (the width of the "sliding window")
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
    def __init__(self, in_channels, out_channels, kernel_size, dilatation, pad_start):
        super(DCCBlock, self).__init__()
        self.conv = DilatedCausalConv1d(in_channels, out_channels, kernel_size, dilatation, pad_start)
        self.relu = nn.ReLU()
        self.lin_transf = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """
        :param x: torch tensor of shape (batch_size, in_channels, timesequence_length)
        """
        out = self.conv(x)
        out = self.relu(out)
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

class DCCRNN(nn.Module):
    '''
    This model is made up of the following block: feature extractor > DCC > lstm > classifier
    '''
    def __init__(self, args):
        super(DCCRNN, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps

        self.feature_extractor = build_feature_extractor(args)

        self.dcc_kernel_size = 2
        self.dcc = DCCBlock(self.feature_extractor.fusion_size,
                            self.feature_extractor.fusion_size,
                            self.dcc_kernel_size,
                            1,
                            True)

        if args.model == 'DCCLSTM':
            self.rnn = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
            self.model = 'DCCLSTM'
        elif args.model == 'DCCGRU':
            self.rnn = MyGRUCell(self.feature_extractor.fusion_size, self.hidden_size)
            self.model = 'DCCGRU'
        else:
            raise Exception('Model ' + args.model + ' here is not supported')
        self.drop = nn.Identity()#nn.Dropout(args.dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        self.feat_vects_queue = []
        self.queue_size = self.dcc_kernel_size

    def forward(self, x, flush_queue=True):
        """
        :param x: torch tensor of shape (batch_size, enc_steps, feat_vect_dim)
        """
        h_n = torch.zeros(x.shape[0],
                          self.hidden_size,
                          device=x.device,
                          dtype=x.dtype)
        c_n = torch.zeros(x.shape[0],
                          self.hidden_size,
                          device=x.device,
                          dtype=x.dtype) if self.model == 'DCCLSTM' else torch.zeros(1).cpu()
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)

        if flush_queue:
            self.feat_vects_queue = []

        for step in range(self.enc_steps):
            x_t = x[:, step]
            out = self.feature_extractor(x_t, torch.zeros(1).cpu())

            out = out.unsqueeze(2)      # out_t.shape == (batch_size, feat_vect_dim, 1)
            self.feat_vects_queue.append(out)
            if step >= self.queue_size:
                assert len(self.feat_vects_queue) == self.queue_size + 1
                self.feat_vects_queue.pop(0)
            out = torch.cat(self.feat_vects_queue, dim=2)
            assert out.shape[-1] <= self.queue_size

            out = self.dcc.step(out).squeeze(2)     # out.shape == (batch_size, feat_vect_dim)

            h_n, c_n = self.rnn(out, (h_n, c_n))
            out = self.classifier(self.drop(h_n))  # out.shape == (batch_size, num_classes)

            scores[:, step] = out
        return scores

    def step(self, x, h_n, c_n, flush_queue=False):
        """
        :param x: torch tensor of shape (batch_size == 1, feat_vect_dim)
        """
        if flush_queue:
            self.feat_vects_queue = []
            
        out = self.feature_extractor(x, torch.zeros(1).cpu())

        out = out.unsqueeze(2)
        self.feat_vects_queue.append(out)
        if len(self.feat_vects_queue) == self.queue_size + 1:
            self.feat_vects_queue.pop(0)
        out = torch.cat(self.feat_vects_queue, dim=2)

        out = self.dcc(out).squeeze(2)
        h_n, c_n = self.rnn(out, (h_n, c_n))
        out = self.classifier(h_n)
        return out, h_n, c_n

if __name__ == '__main__':
    torch.manual_seed(0)

    TIMESEQUENCE_LENGTH = 4
    KERNEL_SIZE = 2
    i = torch.randn(2, 2, TIMESEQUENCE_LENGTH)
    d = DCCBlock(2, 2, KERNEL_SIZE, 1, True)

    # do a total forward pass
    v1 = d(i)

    # do a step-by-step forward pass
    v2 = []
    for timestep in range(1, TIMESEQUENCE_LENGTH + 1, 1):
        if timestep < KERNEL_SIZE:
            appo = d.step(i[:, :, :timestep])
        else:
            appo = d.step(i[:, :, timestep - KERNEL_SIZE:timestep])
        v2.append(appo)
    v2 = torch.cat(v2, dim=2)

    assert True == torch.all(torch.eq(v1, v2))