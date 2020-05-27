import torch
import torch.nn as nn
from torchvision import models

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=[-2, -1])

class THUMOSFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(THUMOSFeatureExtractor, self).__init__()

        self.trainable = args.feat_extr_trainable
        # This variable will contain the dimension of the final feature
        #  vector, thanks to the global_avg_pool the dimension of the
        #  feature vector will be equal to the number of filters in the last conv layer
        self.feat_vect_dim = None
        self.modelname = args.feat_extr
        if self.modelname == 'resnet18':
            self.feature_extractor = models.resnet18(pretrained=True)
            self.feature_extractor = nn.Sequential(
                *list(self.feature_extractor.children())[:-2],     # remove linear and adaptive_avgpool
                GlobalAvgPool(),
            )
            self.feat_vect_dim = 512
        elif self.modelname == 'resnet152':
            self.feature_extractor = models.resnet152(pretrained=True)
            self.feature_extractor = nn.Sequential(
                *list(self.feature_extractor.children())[:-2],     # remove linear and adaptive_avgpool
                GlobalAvgPool(),
            )
            self.feat_vect_dim = 2048
        elif self.modelname == 'vgg16':
            self.feature_extractor = models.vgg16(pretrained=True)
            self.feature_extractor = nn.Sequential(
                *list(self.feature_extractor.children())[:-9],     # remove fully-connected layers and maxpool
                GlobalAvgPool(),
            )
            self.feat_vect_dim = 512
        else:
            raise Exception('modelname not found')

        self.input_linear = nn.Sequential(
            nn.Linear(self.feat_vect_dim, self.feat_vect_dim),
            nn.ReLU(inplace=True),
        )

        if not self.trainable:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.input_linear(x)
        return x

_FEATURE_EXTRACTORS = {
    'THUMOS': THUMOSFeatureExtractor,
}

def build_feature_extractor(args):
    func = _FEATURE_EXTRACTORS[args.dataset]
    return func(args)