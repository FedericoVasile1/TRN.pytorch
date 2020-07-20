import torch
import torch.nn as nn
import numpy as np

from .feature_extractor import build_feature_extractor

'''
Actually this model is only for evaluation. The two submodules(discrimiantor and action) must be trained separately
'''
class DiscrActLSTM(nn.Module):
    def __init__(self, args):
        super(DiscrActLSTM, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps

        self.feature_extractor = build_feature_extractor(args)

        # The discriminator lstm discriminate between background and action
        self.discr = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.discr_classifier = nn.Linear(self.hidden_size, 2)  # 2 because background and action
        # The action lstm predicts only when the discriminator has predicted action, so now it's up to the
        #  action lstm to predict the class of the action
        self.act = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.act_classifier = nn.Linear(self.hidden_size, self.num_classes)

    def step(self, camera_input, discr_h_n, discr_c_n, act_h_n, act_c_n):
        # camera_input.shape == (batch_size, feat_vect_dim)
        feat_vect = self.feature_extractor(camera_input, torch.zeros(1))

        discr_h_n, discr_c_n = self.discr(feat_vect, (discr_h_n, discr_c_n))
        out = self.discr_classifier(discr_h_n)   # out.shape == (batch_size, num_classes) == (1, 2)
        #assert out.shape == torch.Size([1, 2]), 'size mismatch, wrong input to step function'

        act_h_n, act_c_n = self.act(feat_vect, (act_h_n, act_c_n))

        if out.argmax().item() == 1:
            out = self.act_classifier(act_h_n)
            out[:, 0] = out.min(dim=1)[0] - 100.0       # suppress background because here the action lstm is forced to predict an action class
        else:
            out = torch.cat([out, torch.zeros(out.shape[0], self.num_classes - 2).to(out.device)], 1)

        return out, discr_h_n, discr_c_n, act_h_n, act_c_n