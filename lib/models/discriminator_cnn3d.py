import torch
import torch.nn as nn
from torchvision import models

from .feature_extractor import build_feature_extractor

class DiscriminatorCNN3D(nn.Module):
    def __init__(self, args):
        super(DiscriminatorCNN3D, self).__init__()
        if args.camera_feature != 'video_frames_24fps':
            raise Exception('Wrong camera_feature option')

        if args.model == 'DISCRIMINATORCNN3D':
            self.feature_extractor = models.video.r2plus1d_18(pretrained=True)
            # freeze all but last(i.e. layer4) conv layers
            #for param in self.feature_extractor.parameters():
            #    param.requires_grad = False
            #for param in self.feature_extractor.layer4.parameters():
            #    param.requires_grad = True
            self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features,
                                                  args.num_classes)  # requires_grad == True by default
        else:
            raise Exception('Wrong model option, ' + args.model + ' model is not supported')

    def forward(self, x):
        x = self.feature_extractor(x)
        return x