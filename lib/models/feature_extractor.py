import os

import torch
import torch.nn as nn
from torchvision import models

from lib.models.i3d.i3d import InceptionI3d

class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        # Actually, we support only features- pre-extracted training
        self.feature_extractor = nn.Identity()

        # To put or not a linear layer between the feature extractor and the recurrent model
        if args.put_linear:
            self.fusion_size = args.neurons
            self.lin_transf = nn.Sequential(
                nn.Linear(self.feat_vect_dim, self.fusion_size),
                nn.ReLU(inplace='true'),
            )
        else:
            self.fusion_size = args.feat_vect_dim
            self.lin_transf = nn.Identity()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.lin_transf(x)
        return x

_FEATURE_EXTRACTORS = {
    'THUMOS': FeatureExtractor,
    'JUDO': FeatureExtractor,
}

def build_feature_extractor(args):
    func = _FEATURE_EXTRACTORS[args.dataset]
    return func(args)