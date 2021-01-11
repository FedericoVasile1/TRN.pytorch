import torch.nn as nn
from torchvision import models

class CNN2D(nn.Module):
    def __init__(self, args):
        super(CNN2D, self).__init__()

        if args.model == 'CNN2D':
            if args.feature_extractor == 'RESNEXT101':
                raise NotImplementedError()
            else:
                raise Exception('Wrong --feature_extractor option, ' + args.feature_extractor + ' is not supported')
        else:
            raise Exception('Wrong model option, ' + args.model + ' model is not supported')

    def forward(self, x):
        # x.shape(batch_size, C, H, W)
        x = self.model(x)
        return x