import os

import torch
import torch.nn as nn
from torchvision import models

from lib.models.i3d.i3d import InceptionI3d

class CNN3D(nn.Module):
    def __init__(self, args):
        super(CNN3D, self).__init__()

        if not args.model_input.endswith(str(args.chunk_size)):
            raise Exception('--chunk_size and --model_input must indicate the same chunk size')
        if 'maxpool3d5a' not in args.model_input:
            raise Exception('Actually, only maxpool3d5a featuremaps are supported')

        if args.model == 'CNN3D':
            if args.feature_extractor == 'I3D':
                self.model = InceptionI3d()
                self.model.load_state_dict(torch.load(os.path.join('lib', 'models', 'i3d', 'rgb_imagenet.pt')))

                # TODO choose which part of the network to train, now it's all trainable
                self.model.freeze_partial_layers()
                self.model.replace_logits(args.num_classes)
                # the forward starts from mixed5b layer
                self.model.VALID_ENDPOINTS = self.model.VALID_ENDPOINTS[-4:]
            else:
                raise Exception('Wrong --feature_extractor option, ' + args.feature_extractor + ' is not supported')
        else:
            raise Exception('Wrong model option, ' + args.model + ' model is not supported')

    def forward(self, x):
        # x.shape(batch_size, C, chunk_size, H, W)
        x = self.model(x)
        return x