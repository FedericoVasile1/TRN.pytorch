import torch
import torch.nn as nn
from torchvision import models

class CNN3D(nn.Module):
    def __init__(self, args):
        super(CNN3D, self).__init__()
        if args.camera_feature != 'video_frames_24fps':
            raise Exception('Wrong camera_feature option')

        if args.model == 'RESNET2+1D':
            self.feature_extractor = models.video.r2plus1d_18(pretrained=True)
            #for param in self.feature_extractor.parameters():
            #    param.requires_grad = False
            self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features,
                                                  args.num_classes) # requires_grad == True by default
        else:
            raise Exception('Wrong model option, ' + args.model + ' model is not supported')

    def forward(self, x):
        # x.shape(batch_size, C, chunk_size, H, W)
        x = self.feature_extractor(x)
        return x