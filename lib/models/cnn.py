import torch.nn as nn
from torchvision import models

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        if 'video_frames_' not in args.camera_feature:
            raise Exception('Wrong camera_feature option. The chosen model can only work in end to end training')

        if args.model == 'CNN' or 'DISCRIMINATORCNN':
            if args.feature_extractor == 'RESNET50':
                self.feature_extractor = models.resnet50(pretrained=True)
                #for param in self.feature_extractor.parameters():
                #    param.requires_grad = False
                self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, args.num_classes)
            elif args.feature_extractor == 'RESNET34':
                self.feature_extractor = models.resnet34(pretrained=True)
                #for param in self.feature_extractor.parameters():
                #    param.requires_grad = False
                self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, args.num_classes)
            elif args.feature_extractor == 'VGG16':
                self.feature_extractor = models.vgg16(pretrained=True)
                #for param in self.feature_extractor.parameters():
                #    param.requires_grad = False
                self.feature_extractor.classifier = nn.Sequential(
                    nn.Linear(25088, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, args.num_classes)
                )
            elif args.feature_extractor == 'RESNET152':
                self.feature_extractor = models.resnet152(pretrained=True)
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                for param in self.feature_extractor.layer4.parameters():
                    param.requires_grad = True
                self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, args.num_classes)
            else:
                raise Exception('Wrong feature_extractor option, ' + args.feature_extractor + ' is not supported')
        else:
            raise Exception('Wrong model option, ' + args.model + ' model is not supported')

    def forward(self, x):
        # x.shape(batch_size, C, H, W)
        x = self.feature_extractor(x)
        return x