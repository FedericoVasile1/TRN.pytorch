import torch
import torch.nn as nn
from torchvision import models

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class THUMOSFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(THUMOSFeatureExtractor, self).__init__()
        if args.inputs != 'camera':
            raise (RuntimeError('Unknown inputs of {}'.format(args.inputs)))

        if args.camera_feature != 'video_frames_24fps':
            # starting from features extracted
            if args.feat_vect_dim == -1:
                raise Exception('Specify the dimension of the feature vector via feat_vect_dim option')
            self.feat_vect_dim = args.feat_vect_dim
            self.feature_extractor = None
        else:
            # starting from frames
            if args.feature_extractor == 'VGG16':
                self.feature_extractor = models.vgg16(pretrained=True)
                self.feature_extractor.classifier = self.feature_extractor.classifier[:2]
                self.feat_vect_dim = self.feature_extractor.classifier.out_features
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
            elif args.feature_extractor == 'RESNET2+1D':
                self.feature_extractor = models.video.r2plus1d_18(pretrained=True)
                self.feature_extractor.fc = nn.Identity()
                self.feat_vect_dim = self.feature_extractor.fc.out_features
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
            else:
                raise Exception('Feature extractor model not supported: '+args.feature_extractor)

        if args.put_linear:
            self.fusion_size = args.neurons
            self.input_linear = nn.Sequential(
                nn.Linear(self.feat_vect_dim , self.fusion_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.fusion_size = self.feat_vect_dim
            self.input_linear = nn.Identity()

    def forward(self, camera_input, motion_input):
        if self.feature_extractor is not None:
            camera_input = self.feature_extractor(camera_input)
        return self.input_linear(camera_input)

_FEATURE_EXTRACTORS = {
    'THUMOS': THUMOSFeatureExtractor,
}

def build_feature_extractor(args):
    func = _FEATURE_EXTRACTORS[args.dataset]
    return func(args)
