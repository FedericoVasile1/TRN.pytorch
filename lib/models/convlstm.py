import torch.nn as nn
import torch

from .feature_extractor import build_feature_extractor

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class SqueezeChunk(nn.Module):
    def __init__(self):
        super(SqueezeChunk, self).__init__()

    def forward(self, x):
        return x.squeeze(2)

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        # The shape of x must be (batch_size, num_channels, height, width)
        return x.mean(dim=(2, 3))

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state(i.e. number of filters of the conv layer).
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2     # in order to preserve spatial dimensions
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,     # we will concat input and hidden state
                              out_channels=4 * self.hidden_dim,      # 4 gates
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              stride=1,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        """
        Parameters
        ----------
        input_tensor: torch tensor of shape (batch_size, 512, 7, 7)
            This tensor is the output feature maps of the feature extractor. Currently, 512-7-7 are hard-coded
            since we support few feature extractors and all of these output feature maps of that dimensions
        cur_state: tuple of two torch tensor of shape (batch_size, hidden_dim, 7, 7)
            The current hidden state and cell state.
        Returns
        -------
        h_next, c_next: both torch tensor of shape (batch_size, hidden_dim, 7, 7)
            The updated hidden state and cell state
        """
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        # Actually (height, width) == (7, 7). See ConvLSTMCell.forward for further explaination.
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    """
    A Full Convolutional lstm model. It is made up of: feature extractor > convlstm > classifier
    Notice that here we work only with RGB feature maps; neither optical flow fusion nor feature vectors are allowed.
    """
    def __init__(self, args):
        super(ConvLSTM, self).__init__()
        if args.camera_feature != 'resnet3d_featuremaps' and args.camera_feature != 'video_frames_24fps':
            raise Exception('Wrong camera_feature option, this model supports only feature maps. '
                            'Change this option to \'resnet3d_featuremaps\' or switch to end to end training with the '
                            'following options: --camera_feature video_frames_24fps --feature_extractor RESNET2+1D')

        #  TODO: These hyperparameters values are temporarily hard-coded; they should be set via command line
        #   arguments and so assigned here via args.<..argument..>
        hidden_dim = [64]
        kernel_size = [(3, 3)]
        num_layers = 1
        bias = True

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.num_classes = args.num_classes
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.steps = args.enc_steps

        self.feature_extractor = build_feature_extractor(args)
        # HARD-CODED: the feature extractors actually supported all returns feature maps of shape (512, 7, 7)
        self.input_dim = 512
        self.H, self.W = (7, 7)

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

        self.classifier = nn.Sequential(
            Flatten(),      # TODO try also other variants, such as global average pooling
            nn.Linear(hidden_dim[-1] * self.H * self.W, self.num_classes),
        )

        # TODO: try also this variant below
        '''
        # NOTICE THAT TO USE THIS WE MUST HAVE hidden_dim[-1] == self.num_classes
        assert hidden_dim[-1] == self.num_classes
        self.classifier = nn.Sequential(
            GlobalAveragePooling(),
        )
        '''

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor: torch tensor either of shape:
          if --feature_extractor == resnet2+1d  =>  (batch_size, enc_steps, 3, chunk_size, 112, 112)
          elif --camera_feature == resnet3d_featuremaps => (batch_size, enc_steps, 512, 7, 7)

        Returns
        -------
        scores: torch tensor of shape (batch_size, enc_steps, num_classes)
        """
        batch_size = input_tensor.shape[0]
        scores = torch.zeros(batch_size, self.steps, self.num_classes)

        # Since the init is done in forward, can send image size here(to be precise, it is not the
        # image size; it is the feature maps size outputted by the feature extractor)
        hidden_state = self._init_hidden(batch_size=batch_size, image_size=(self.H, self.W))

        for step in range(self.steps):
            input_t = input_tensor[:, step]

            feature_maps = self.feature_extractor(input_t, torch.zeros(1).cpu())
            # feature_maps.shape == (batch_size, 512, 7, 7)

            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                # forward pass in convlstm cell
                h, c = self.cell_list[layer_idx](input_tensor=feature_maps, cur_state=[h, c])
                hidden_state[layer_idx] = (h, c)
                feature_maps = h

            scores[:, step] = self.classifier(h)

        return scores

    def step(self, input_tensor, hidden_state):
        # input_tensor.shape == (batch_size, 3, chunk_size, 112, 112) || (batch_size, 512, 7, 7)
        batch_size = input_tensor.shape[0]
        scores = torch.zeros(batch_size, self.num_classes)

        feature_maps = self.feature_extractor(input_tensor, torch.zeros(1).cpu())

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            h, c = self.cell_list[layer_idx](input_tensor=feature_maps, cur_state=[h, c])
            hidden_state[layer_idx] = (h, c)
            feature_maps = h

        scores = self.classifier(h)

        return scores, hidden_state

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param