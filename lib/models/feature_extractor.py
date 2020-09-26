import torch
import torch.nn as nn
from torchvision import models

from lib.models.i3d import InceptionI3d

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze(2)

class THUMOSFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(THUMOSFeatureExtractor, self).__init__()
        if args.inputs in ['camera', 'motion', 'multistream']:
            self.with_camera = 'motion' not in args.inputs
            self.with_motion = 'camera' not in args.inputs
        else:
            raise (RuntimeError('Unknown inputs of {}'.format(args.inputs)))

        self.camera_feature = args.camera_feature
        if args.camera_feature != 'video_frames_24fps':
            # starting from features extracted
            if args.feat_vect_dim == -1:
                raise Exception('Specify the dimension of the feature vector via feat_vect_dim option')
            # in case of feature maps, the feat_vect_dim is intended to be the channels dimension of the feature maps
            self.feat_vect_dim = args.feat_vect_dim
            self.feature_extractor = nn.Identity()   # no feature extractor needed
        else:
            # starting from frames, so choose a feature extractor
            if args.feature_extractor == 'VGG16':
                self.feature_extractor = models.vgg16(pretrained=True)

                # Depending on the model, the feature extractor must return feature maps or feature vectors
                if args.model == 'LSTMATTENTION' or args.model == 'CONVLSTM' or args.model == 'SELFATTENTION':
                    # here we need to return the feature maps, so remove adaptiveavgpool and linear.
                    # The output shape is (batch_size, 512, 7, 7)
                    self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
                    self.feat_vect_dim = 512    # HARD-CODED; number of **channels** of the output feature maps
                else:
                    self.feat_vect_dim = self.feature_extractor.classifier[0].out_features
                    self.feature_extractor.classifier = self.feature_extractor.classifier[:2]   # extract fc6 feature vector

                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            elif args.feature_extractor == 'RESNET34':
                self.feature_extractor = models.resnet34(pretrained=True)
                self.feat_vect_dim = self.feature_extractor.fc.in_features

                # Depending on the model, the feature extractor must return feature maps or feature vectors
                if args.model == 'LSTMATTENTION' or args.model == 'CONVLSTM' or args.model == 'SELFATTENTION':
                    # here we need to return the feature maps, so remove adaptiveavgpool and linear.
                    # The output shape is (batch_size, 512, 7, 7)
                    self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
                else:
                    self.feature_extractor.fc = nn.Identity()   # extract feature vector

                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            elif args.feature_extractor == 'RESNET2+1D':
                self.feature_extractor = models.video.r2plus1d_18(pretrained=True)
                self.feat_vect_dim = self.feature_extractor.fc.in_features

                # Depending on the model, the feature extractor must return feature maps or feature vectors
                if args.model == 'LSTMATTENTION' or args.model == 'CONVLSTM' or args.model == 'SELFATTENTION':
                    # here we need to return the feature maps, so remove adaptiveavgpool and linear.
                    # The output shape before Squeeze() is (batch_size, 512, 1, 7, 7)
                    self.feature_extractor = nn.Sequential(
                        *list(self.feature_extractor.children())[:-2],
                        Squeeze(),
                    )
                else:
                    self.feature_extractor.fc = nn.Identity()

                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            elif args.feature_extractor == 'I3D':
                # TODO: add also feature maps extraction(see resnet2+1d above)
                self.feature_extractor = InceptionI3d()
                self.feature_extractor.load_state_dict(torch.load('rgb_imagenet.pt'))
                self.feat_vect_dim = 1024
                self.feature_extractor.dropout = nn.Identity()
                self.feature_extractor.logits = nn.Identity()

                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            else:
                raise Exception('Feature extractor model not supported: '+args.feature_extractor)

        if self.with_camera and self.with_motion and args.camera_feature != 'resnet3d_featuremaps':
            MOTION_FEAT_VECT = self.feat_vect_dim       # modify here if needed
            self.feat_vect_dim += MOTION_FEAT_VECT

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


    def forward(self, camera_input, motion_input):
        # camera_input.shape == (batch_size, feat_vect_dim)  , motion_input.shape == (batch_size, feat_vect_dim_2)
        if self.with_camera and self.with_motion:
            camera_input = self.feature_extractor(camera_input)
            motion_input = self.feature_extractor(motion_input)

            fusion_input = torch.cat((camera_input, motion_input), 1)
        elif self.with_camera:
            fusion_input = self.feature_extractor(camera_input)
        elif self.with_motion:
            fusion_input = self.feature_extractor(motion_input)

        fusion_input = self.input_linear(fusion_input)
        return fusion_input

_FEATURE_EXTRACTORS = {
    'THUMOS': THUMOSFeatureExtractor,
    'JUDO': THUMOSFeatureExtractor,
}

def build_feature_extractor(args):
    func = _FEATURE_EXTRACTORS[args.dataset]
    return func(args)
