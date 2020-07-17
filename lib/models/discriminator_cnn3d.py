import torch
import torch.nn as nn
from torchvision import models

from lib.models.c3d import C3D

class DiscriminatorCNN3D(nn.Module):
    def __init__(self, args):
        super(DiscriminatorCNN3D, self).__init__()
        if args.camera_feature != 'video_frames_24fps':
            raise Exception('Wrong camera_feature option')

        if args.model == 'DISCRIMINATORCNN3D':
            if args.feature_extractor == 'RESNET2+1D':
                self.feature_extractor = models.video.r2plus1d_18(pretrained=True)
                for param in self.feature_extractor.parameters():  # TODO: better understand which part should be freezed
                    param.requires_grad = False
                self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features,
                                                      args.num_classes)  # requires_grad == True by default
            elif args.feature_extractor == 'C3D':
                self.feature_extractor = C3D(args)
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                self.feature_extractor.fc6 = nn.Linear(8192, 4096)
                self.feature_extractor.fc7 = nn.Linear(4096, 4096)
                self.feature_extractor.fc8 = nn.Linear(4096, args.num_classes)
            else:
                raise Exception('Wrong feature_extractor option, ' + args.feature_extractor + ' model is not supported')
        else:
            raise Exception('Wrong model option, ' + args.model + ' model is not supported')

    def forward(self, x):
        x = self.feature_extractor(x)
        return x