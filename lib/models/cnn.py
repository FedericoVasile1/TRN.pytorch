import torch
import torch.nn as nn
from torchvision import models

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        if args.camera_feature != 'video_frames_24fps':
            raise Exception('Wrong camera_feature option')

        if args.model == 'CNN':
            if args.feature_extractor == 'RESNET50':
                self.feature_extractor = models.resnet50(pretrained=True)
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features,
                                                      args.num_classes)
            else:
                raise Exception('Wrong feature_extractor option, ' + args.feature_extractor + ' is not supported')
        else:
            raise Exception('Wrong model option, ' + args.model + ' model is not supported')

    def forward(self, x):
        # x.shape(batch_size, C, H, W)
        x = self.feature_extractor(x)
        return x