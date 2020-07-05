import torch
import torch.nn as nn
from torchvision import models

class FC_AC(nn.Module):
    def __init__(self, args):
        super(FC_AC, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(int(args.feat_vect_dim), 2)
        )

    def forward(self, x):       # x.shape(batch_size, C, chunk_size, H, W)
        x = self.linear(x)
        return x