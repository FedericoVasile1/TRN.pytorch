
import torch.nn as nn

from lib.models.i3d.i3d import Unit3D

class OnlyAct(nn.Module):
    def __init__(self, args):
        super(OnlyAct, self).__init__()
        self.classifier = Unit3D(in_channels=1024,
                                 output_channels=args.num_classes,
                                 kernel_shape=[1, 1, 1],
                                 padding=0,
                                 activation_fn=None,
                                 use_batch_norm=False,
                                 use_bias=True,
                                 name=self._name + 'Logits')

    def forward(self, x):
        x = self.classifier(x)
        return x