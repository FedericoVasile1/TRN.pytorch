import torch
import torch.nn as nn
from torchvision import models

class CNN_LateFusion(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        if 'video_frames_' not in args.camera_feature:
            raise Exception('Wrong camera_feature option. The chosen model can only work in end to end training')

        self.chunk_size = args.chunk_size

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
        # x.shape(batch_size, chunk_size, C, H, W)
        all_preds = []
        for ith_frame in range(self.chunk_size):
            pred = self.feature_extractor(x[:, ith_frame])
            all_preds.append(pred)

        all_preds = torch.stack(all_preds)  # all_preds.shape(chunk_size, batch_size, num_classes)
        mean_pred = all_preds.mean(dim=0)   # mean_preds.shape(batch_size, num_classes)
        return mean_pred