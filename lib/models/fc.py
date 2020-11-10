import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class FC(nn.Module):
    def __init__(self, args):
        super(FC, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes

        self.feature_extractor = build_feature_extractor(args)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_extractor.fusion_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, args.num_classes)
        )

    def forward(self, x):
        # x.shape == (batch_size, feat_vect_dim)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x