import torch
import torch.nn as nn
from torchvision import models

class CNN3D(nn.Module):
    def __init__(self, args):
        super(CNN3D, self).__init__()

        self.feat_extr = models.video.r2plus1d_18(pretrained=True)
        # freeze all but last(i.e. layer4) conv layers
        for param in self.feat_extr.parameters():
            param.requires_grad = False
        for param in self.feat_extr.layer4.parameters():
            param.requires_grad = True
        self.feat_extr.fc = nn.Linear(self.feat_extr.fc.in_features, args.num_classes) # requires_grad == True by default

        self.CHUNK_SIZE = 6

    def forward(self, x):       # x.shape(batch_size, C, chunk_size, H, W)
        x = self.feat_extr(x)
        return x