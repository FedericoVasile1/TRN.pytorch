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

        self.trainable = args.feat_extr_trainable
        # This variable will contain the dimension of the final feature
        #  vector, thanks to the global_avg_pool the dimension of the
        #  feature vector will be equal to the number of filters in the last conv layer
        self.feat_vect_dim = None
        self.modelname = args.feat_extr
        if self.modelname == 'resnet18':
            self.feature_extractor = models.resnet18(pretrained=True)
            self.feature_extractor = nn.Sequential(
                *list(self.feature_extractor.children())[:-1],
                Flatten(),
            )
            self.feat_vect_dim = 512 * 1 * 1    # this line should not be hardcoded
        elif self.modelname == 'resnet152':
            self.feature_extractor = models.resnet152(pretrained=True)
            self.feature_extractor = nn.Sequential(
                *list(self.feature_extractor.children())[:-1],
                Flatten(),
            )
            self.feat_vect_dim = 2048 * 1 * 1   # this line should not be hardcoded
        elif self.modelname == 'vgg16':
            self.feature_extractor = models.vgg16(pretrained=True)
            self.feature_extractor = nn.Sequential(
                *list(self.feature_extractor.children())[:-1],
                Flatten(),
            )
            self.feat_vect_dim = 512 * 7 * 7    # this line should not be hardcoded
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