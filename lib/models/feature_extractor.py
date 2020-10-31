import torch
import torch.nn as nn
from torchvision import models

from lib.models.i3d import InceptionI3d

class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        if args.E2E == '':
            # starting from features extracted
            self.feat_vect_dim = args.feat_vect_dim
            self.feature_extractor = nn.Identity()   # no feature extractor needed
        else:
            # starting from frames, so choose a feature extractor
            if args.feature_extractor == 'VGG16':
                self.feature_extractor = models.vgg16(pretrained=True)

                self.feat_vect_dim = self.feature_extractor.classifier[0].out_features
                self.feature_extractor.classifier = self.feature_extractor.classifier[:2]   # extract fc6 feature vector

                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            elif args.feature_extractor == 'RESNET34':
                self.feature_extractor = models.resnet34(pretrained=True)
                self.feat_vect_dim = self.feature_extractor.fc.in_features

                self.feature_extractor.fc = nn.Identity()   # extract feature vector

                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            elif args.feature_extractor == 'RESNET2+1D':
                self.feature_extractor = models.video.r2plus1d_18(pretrained=True)
                self.feat_vect_dim = self.feature_extractor.fc.in_features

                self.feature_extractor.fc = nn.Identity()

                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            elif args.feature_extractor == 'I3D':
                self.feature_extractor = InceptionI3d()
                self.feature_extractor.load_state_dict(torch.load('rgb_imagenet.pt'))
                self.feat_vect_dim = 1024
                self.feature_extractor.dropout = nn.Identity()
                self.feature_extractor.logits = nn.Identity()

                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            else:
                raise Exception('Wrong --feature_extractor option. '+args.feature_extractor+' unknown')

        # To put or not a linear layer between the feature extractor and the recurrent model
        if args.put_linear:
            self.fusion_size = args.neurons
            self.input_linear = nn.Sequential(
                nn.Linear(self.feat_vect_dim , self.fusion_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.fusion_size = self.feat_vect_dim
            self.input_linear = nn.Identity()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.input_linear(x)
        return x

_FEATURE_EXTRACTORS = {
    'THUMOS': FeatureExtractor,
    'JUDO': FeatureExtractor,
}

def build_feature_extractor(args):
    func = _FEATURE_EXTRACTORS[args.dataset]
    return func(args)