import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

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
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)       # in order to preserve input spatial dimensions

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
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape, device):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))

'''
At this moment, this model is designed to be trained only in a end to end way, i.e.  starting from frames
'''
class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, args, input_channels=-1, hidden_channels=[256, 128], kernel_size=3, step=-1):
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
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
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

        assert self.step == input.shape[1], 'steps mismatch'

        for step in range(self.step):
            x = input[:, step]
            x = self.feature_extractor(x).squeeze(2)       # x.shape == (batch_size, 512, 7, 7)
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width), device=input.device)
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

    input = Variable(torch.randn(2, 4, 3, 1, 112, 112)).to(device)
    target = Variable(torch.randint(2, (2, 4, 2))).to(device)

    output = convlstm(input)
    print(output.shape)
    print(output)