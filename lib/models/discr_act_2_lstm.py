import torch
import torch.nn as nn
import numpy as np

from .feature_extractor import build_feature_extractor

'''
Actually this model is only for evaluation. The two submodules(discrimiantor and action) must be trained separately
'''
class DiscrActLSTM2(nn.Module):
    def __init__(self, args):
        super(DiscrActLSTM2, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps
        self.act_h_n = None
        self.act_c_n = None

        self.is_first = True

        self.backgr_vect = torch.zeros(1, self.num_classes)   # (batch_size, num_classes)
        self.backgr_vect[0, 0] = 100.0

        self.feature_extractor = build_feature_extractor(args)

        # The discriminator lstm discriminate between background and action
        self.discr = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.discr_classifier = nn.Linear(self.hidden_size, 2)  # 2 because background and action

    def step(self, camera_input, discr_h_n, discr_c_n, target):
        feat_vect = self.feature_extractor(camera_input, torch.zeros(1))

        discr_h_n, discr_c_n = self.discr(feat_vect, (discr_h_n, discr_c_n))
        out = self.discr_classifier(discr_h_n)   # out.shape == (batch_size, num_classes) == (1, 2)
        assert out.shape == torch.Size([1, 2]), 'size mismatch, wrong input to step function'

        if out[0, 1].item() > 0.5:
            out = torch.zeros_like(self.backgr_vect)
            out[0, np.argmax(target)] = 100.0
        else:
            self.is_first = True
            out = self.backgr_vect

        return out, discr_h_n, discr_c_n