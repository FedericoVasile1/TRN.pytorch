import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .feature_extractor import build_feature_extractor

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

        # The input to the action lstm is the discriminator lstm prediction, and the initial hidden state is
        self.act = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.act_classifier = nn.Linear(self.hidden_size, self.num_classes)

        self.classifier = nn.Linear(self.num_classes + 2, self.num_classes)

    def step(self, camera_input, discr_h_n, discr_c_n, act_h_n, act_c_n):
        # camera_input.shape == (batch_size, feat_vect_dim)
        feat_vect = self.feature_extractor(camera_input, torch.zeros(1))

        discr_h_n, discr_c_n = self.discr(feat_vect, (discr_h_n, discr_c_n))
        out_discr = self.discr_classifier(discr_h_n)   # out.shape == (batch_size, num_classes) == (1, 2)

        act_h_n, act_c_n = self.act(feat_vect, (act_h_n, act_c_n))
        out_act = self.act_classifier(act_h_n)

        out = self.discr_to_classes(out_discr)
        out += out_act
        out = self.classifier(out)

        return out, discr_h_n, discr_c_n, act_h_n, act_c_n

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        discr_h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        discr_c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        act_h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        act_c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device,dtype=x.dtype)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        scores_discr = torch.zeros(x.shape[0], x.shape[1], 2, dtype=x.dtype)
        scores_act = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)

        for step in range(self.enc_steps):
            x_t = x[:, step]
            x_t = self.feature_extractor(x_t, torch.zeros(1))
            discr_h_n, discr_c_n = self.discr(x_t, (discr_h_n, discr_c_n))
            out_discr = self.discr_classifier(discr_h_n)

            act_h_n, act_c_n = self.act(x_t, (act_h_n, act_c_n))
            out_act = self.act_classifier(act_h_n)

            out = torch.cat((out_discr, out_act), dim=1)
            out = self.classifier(out)

            scores_discr[:, step, :] = out_discr
            scores_act[:, step, :] = out_act
            scores[:, step, :] = out

        return scores, scores_discr, scores_act


