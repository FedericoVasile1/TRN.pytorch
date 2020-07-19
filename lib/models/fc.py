import torch
import torch.nn as nn
from torchvision import models

class FC(nn.Module):
    def __init__(self, args):
        super(FC, self).__init__()
        if args.camera_feature == 'video_frames_24fps':
            raise Exception('Wrong camera_feature option')

        if args.model == 'FC':
            self.model = nn.Sequential(
                nn.Linear(args.feat_vect_dim, args.num_classes),
            )
        else:
            raise Exception('Wrong model option, ' + args.model + ' model is not supported')

    def forward(self, x):
        # x.shape(batch_size, feat_vect_dim)
        x = self.model(x)
        return x