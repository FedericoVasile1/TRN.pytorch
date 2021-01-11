import torch.nn as nn
from torchvision import models

class CNN2D(nn.Module):
    def __init__(self, args):
        super(CNN2D, self).__init__()

        if args.model == 'CNN2D':
            if args.feature_extractor == 'RESNEXT101':
                self.model = models.resnext101_32x8d(pretrained=True)

                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.layer4.parameters():
                    param.requires_grad = True
                for param in self.model.avgpool.parameters():
                    param.requires_grad = True

                self.model.fc = nn.Linear(self.model.fc.in_features, args.num_classes)
            else:
                raise Exception('Wrong --feature_extractor option, ' + args.feature_extractor + ' is not supported')
        else:
            raise Exception('Wrong model option, ' + args.model + ' model is not supported')

    def forward(self, x):
        # x.shape(batch_size, C, H, W)
        x = self.model(x)
        return x