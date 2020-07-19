import torch
import torch.nn as nn
import numpy as np

from .feature_extractor import build_feature_extractor
from .convlstm import ConvLSTM

'''
Actually this model is only for evaluation. The two submodules(discrimiantor and action) must be trained separately
'''
class DiscrActConvLSTM2(nn.Module):
    def __init__(self, args):
        super(DiscrActConvLSTM2, self).__init__()
        self.num_classes = args.num_classes

        self.backgr_vect = torch.zeros(1, self.num_classes)   # (batch_size, num_classes)
        self.backgr_vect[0, 0] = 100.0

        # The discriminator convlstm discriminate between background and action
        self.discr = ConvLSTM(args)

    def forward(self, x):
        pass

    def step(self, camera_input, h, c, target):
        # camera_input.shape == (batch_size, 3, chunk_size, 112, 112)
        out = self.discr.step(camera_input, (h, c))
        assert out.shape[1] == 2, 'size mismatch, provided wrong input to step function'

        if out.argmax().item() == 1:
            if np.argmax(target) != 0:
                out = torch.zeros_like(self.backgr_vect)
                out[0, np.argmax(target)] = 100.0
            else:
                out = torch.zeros_like(self.backgr_vect)
                out[0, 1] = 100.0
        else:
            self.is_first = True
            out = self.backgr_vect

        return out, h, c