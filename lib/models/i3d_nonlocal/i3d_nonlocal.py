import torch
import torch.nn as nn

from lib.models.i3d import InceptionI3d
from lib.models.i3d_nonlocal.Non_local_pytorch.lib.non_local_dot_product import NONLocalBlock3D

class I3dNonLocal(nn.Module):
    def __init__(self, args):
        super(I3dNonLocal, self).__init__()

        self.model = InceptionI3d(chunk_size=9)
        self.model.load_state_dict(torch.load('rgb_imagenet.pt'))
        self.model.replace_logits(args.num_classes)
        self.freeze_partial_layers()

        self.model.end_points['nl1'] = NONLocalBlock3D(832)        # after mixed4f
        self.model.end_points['nl2'] = NONLocalBlock3D(832)        # after mixed4f
        self.model.end_points['nl3'] = NONLocalBlock3D(832)        # after mixed5b
        self.model.end_points['nl4'] = NONLocalBlock3D(1024)       # after mixed5c

        self.model.VALID_ENDPOINTS = (
            'Conv3d_1a_7x7',
            'MaxPool3d_2a_3x3',
            'Conv3d_2b_1x1',
            'Conv3d_2c_3x3',
            'MaxPool3d_3a_3x3',
            'Mixed_3b',
            'Mixed_3c',
            'MaxPool3d_4a_3x3',
            'Mixed_4b',
            'Mixed_4c',
            'Mixed_4d',
            'Mixed_4e',
            'Mixed_4f',
            'nl1',
            'nl2',
            'MaxPool3d_5a_2x2',
            'Mixed_5b',
            'nl3'
            'Mixed_5c',
            'nl4',
            'Logits',
            'Predictions',
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def freeze_partial_layers(self):
        LAYERS_TO_FREEZE = (
            'Conv3d_1a_7x7',
            'MaxPool3d_2a_3x3',
            'Conv3d_2b_1x1',
            'Conv3d_2c_3x3',
            'MaxPool3d_3a_3x3',
            'Mixed_3b',
            'Mixed_3c',
            'MaxPool3d_4a_3x3',
            'Mixed_4b',
            'Mixed_4c',
            'Mixed_4d',
            'Mixed_4e',
            #'Mixed_4f',
        )

        for end_point in LAYERS_TO_FREEZE:
            for param in self.model._modules[end_point].parameters():
                param.requires_grad = False