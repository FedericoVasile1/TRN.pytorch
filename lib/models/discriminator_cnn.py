import torch
import torch.nn as nn
from torchvision import models

from .feature_extractor import build_feature_extractor

class DiscriminatorCNN(nn.Module):
    def __init__(self, args):
        super(DiscriminatorCNN, self).__init__()
        if args.camera_feature != 'video_frames_24fps':
            raise Exception('Wrong camera_feature option')

        if args.model == 'DISCRIMINATORCNN':
            if args.feature_extractor == 'RESNET18':
                self.feature_extractor = models.resnet18(pretrained=True)
                for param in self.feature_extractor.layer1.parameters():
                    param.requires_grad = False
                for param in self.feature_extractor.layer2.parameters():
                    param.requires_grad = False
                self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, args.num_classes)
            elif args.feature_extractor == 'VGG16':
                self.feature_extractor = models.vgg16(pretrained=True)
                for param in self.feature_extractor.features[17:].parameters():
                    param.requires_grad = False
                self.feature_extractor.classifier = nn.Sequential(
                    nn.Linear(25088, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, args.num_classes)
                )
            else:
                raise Exception('Wrong feature_extractor option, ' + args.feature_extractor + ' is not supported')
        else:
            raise Exception('Wrong model option, ' + args.model + ' model is not supported')

    def forward(self, x):
        x = self.feature_extractor(x)
        return x