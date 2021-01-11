import os

import torch
import torch.nn as nn
from torchvision import models

from lib.models.i3d.i3d import InceptionI3d

class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        if args.E2E == '':
            # starting from features extracted
            self.feat_vect_dim = args.feat_vect_dim

            # To put or not a linear layer between the feature extractor and the recurrent model
            if args.put_linear:
                self.fusion_size = args.neurons
                self.feature_extractor = nn.Sequential(
                    nn.Linear(self.feat_vect_dim, self.fusion_size),
                    nn.ReLU(inplace='true'),
                )
            else:
                self.fusion_size = self.feat_vect_dim
                self.feature_extractor = nn.Identity()

        else:
            # Actually this part is empty, it should be fill in case you want to do a CNN+RNN end to
            #  end training. So here you would put the self.feature_extractor object, similar to CNN2D.py and CNN3D.py
            if args.model not in ('CNN3D', 'CNN2D'):
                raise NotImplementedError('Actually we support end to end training only for CNN3D and CNN2D models')
            else:
                # we should not get here
                raise NotImplementedError()

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

_FEATURE_EXTRACTORS = {
    'THUMOS': FeatureExtractor,
    'JUDO': FeatureExtractor,
}

def build_feature_extractor(args):
    func = _FEATURE_EXTRACTORS[args.dataset]
    return func(args)