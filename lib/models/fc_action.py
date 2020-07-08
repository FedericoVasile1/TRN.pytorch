import torch
import torch.nn as nn
from torchvision import models

class FC_Action(nn.Module):
    def __init__(self, args):
        super(FC_Action, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(int(args.feat_vect_dim), args.num_classes)
        )

    def forward(self, x):       # x.shape(batch_size, C, chunk_size, H, W)
        x = self.linear(x)
        return x