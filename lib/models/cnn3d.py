import torch
import torch.nn as nn
from torchvision import models

from lib.models.i3d import InceptionI3d
from lib.models.i3d_nonlocal.i3d_nonlocal import I3dNonLocal

class CNN3D(nn.Module):
    def __init__(self, args):
        super(CNN3D, self).__init__()
        if 'video_frames_' not in args.camera_feature:
            raise Exception('Wrong camera_feature option. The chosen model can only work in end to end training')

        if args.model == 'CNN3D' or args.model == 'DISCRIMINATORCNN3D':
            if args.feature_extractor == 'RESNET2+1D':
                self.feature_extractor = models.video.r2plus1d_18(pretrained=True)
                self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, args.num_classes)

                # choose which part of the network to train, now it's all trainable
            elif args.feature_extractor == 'I3D':
                self.feature_extractor = InceptionI3d()
                # load i3d weights from imagenet + kinetics training
                self.feature_extractor.load_state_dict(torch.load('rgb_imagenet.pt'))
                self.feature_extractor.replace_logits(args.num_classes)

                # choose which part of the network to train, now it's all trainable
            elif args.feature_extractor == 'I3DNONLOCAL':
                self.feature_extractor = I3dNonLocal()

            else:
                raise Exception('Wrong feature_extractor option, ' + args.feature_extractor + ' is not supported')
        else:
            raise Exception('Wrong model option, ' + args.model + ' model is not supported')

    def forward(self, x):
        # x.shape(batch_size, C, chunk_size, H, W)
        x = self.feature_extractor(x)
        return x