import torch
import torch.nn as nn
from torchvision import models
from torch.nn.parameter import Parameter

import math

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=[-2, -1])

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, input_height, input_width):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)       # in order to preserve input spatial dimensions

        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,  # we will concat input and hidden state
                              out_channels=self.hidden_channels * 4,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=self.padding,
                              bias=True)

        # TODO: figure out how to initialize this
        self.Wci = Parameter(torch.randn(1, self.hidden_channels, input_height, input_width)).div(math.sqrt(self.hidden_channels))
        self.Wcf = Parameter(torch.randn(1, self.hidden_channels, input_height, input_width)).div(math.sqrt(self.hidden_channels))
        self.Wco = Parameter(torch.randn(1, self.hidden_channels, input_height, input_width)).div(math.sqrt(self.hidden_channels))

    def forward(self, x, h, c):
        x_cat_h = torch.cat([x, h], dim=1)      # concatenate along channel dimension
        conv = self.conv(x_cat_h)

        i, f, tmp_c, o = torch.chunk(conv, 4, dim=1)
        i = torch.sigmoid(i + self.Wci * c)
        f = torch.sigmoid(f + self.Wcf * c)
        c = f * c + i * torch.tanh(tmp_c)
        o = torch.sigmoid(o + self.Wco * c)
        h = o * torch.tanh(c)

        return h, c

    def init_hidden(self, batch_size, hidden, shape):
        return (torch.zeros(batch_size, hidden, shape[0], shape[1]),
                torch.zeros(batch_size, hidden, shape[0], shape[1]))

'''
At this moment, this model is designed to be trained only in a end to end way, i.e.  starting from frames
'''
class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, args, input_channels=-1, hidden_channels=[16], kernel_size=3, step=-1):
        super(ConvLSTM, self).__init__()
        if args.model != 'CONVLSTM':
            raise Exception('wrong model name, expected CONVLSTM, given ' + args.model)
        if args.feature_extractor != 'RESNET2+1D':
            raise Exception('wrong feature extractor name, expected RESNET2+1D, given ' + args.feature_extractor)
        if args.camera_feature != 'video_frames_24fps':
            raise Exception('At the moment, this model is designed to be trained only starting from frames,'
                            ' expected video_frames_24fps, given ' + args.camera_feature)

        self.num_classes = args.num_classes
        input_channels = 512        # !!! HARD-CODED; SEE FEATURE_EXTRACTOR COMMENTS BELOW!
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = args.enc_steps

        self.feature_extractor = models.video.r2plus1d_18(pretrained=True)
        self.feature_extractor = nn.Sequential(
            # - drop adaptiveavgpool and fc from feature extractor
            # - given an input shape of (batch_size, 3, chunk_size, 112, 112)
            #             out1.shape == (batch_size, 512, 1, 7, 7)          -> hence, channels == 512
            *list(self.feature_extractor.children())[:-2],  # out1
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, 7, 7)    # 7x7 HARD-CODED
            setattr(self, name, cell)
            self._all_layers.append(cell)

        self.classifier = nn.Sequential(
            GlobalAvgPool(),            #TODO: try also other alternatives
            nn.Linear(self.hidden_channels[-1], self.num_classes),
        )

    def forward(self, input):
        # input.shape == (batch_size, steps, C, chunk_size, H, W)
        internal_state = []     # list of num_layers elements, each one is a tuple containing hidden state and cell state
        outputs = torch.zeros(input.shape[0], input.shape[1], self.num_classes, dtype=torch.float32)

        for step in range(self.step):
            x = input[:, step]
            x = self.feature_extractor(x).squeeze(2)       # x.shape == (batch_size, 512, 7, 7)
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    h = h.to(input.device)
                    c = c.to(input.device)
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)  # the hidden state 'x' returned will be the input of the next layer
                internal_state[i] = (x, new_c)

            out = self.classifier(x)        # the hidden state of the last layer will be used to generate the output
            outputs[:, step] = out

        return outputs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    convlstm = ConvLSTM(None, input_channels=3, hidden_channels=[64, 32], kernel_size=3, step=4).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    input = torch.randn(2, 4, 3, 1, 112, 112).to(device)
    target = torch.randint(2, (2, 4, 2)).to(device)

    output = convlstm(input)
    print(output.shape)
    print(output)