
import torch.nn as nn

from lib.models.i3d.i3d import Unit3D

class OnlyAct(nn.Module):
    def __init__(self, args):
        super(OnlyAct, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, args.num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x