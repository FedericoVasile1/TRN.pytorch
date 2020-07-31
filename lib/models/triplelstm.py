import torch
import torch.nn as nn

import _init_paths
import utils as utl
from .feature_extractor import build_feature_extractor

class TripleLSTM(nn.Module):
    def __init__(self, args):
        super(TripleLSTM, self).__init__()
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.num_classes_actback = 2                        # action and background
        self.num_classes_acts = args.num_classes            # all actions
        self.num_classes_startend = args.num_classes * 2    # all actions divided in start and end

        self.feature_extractor = build_feature_extractor(args)

        self.actback = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.actback_classifier = nn.Linear(self.hidden_size, self.num_classes_actback)

        self.acts = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.acts_classifier = nn.Linear(self.hidden_size, self.num_classes_acts)

        self.startend = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.startend_classifier = nn.Linear(self.hidden_size, self.num_classes_startend)

        self.final_classifier = nn.Linear(4, self.num_classes_acts)

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        actback_h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        actback_c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        acts_h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        acts_c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        startend_h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        startend_c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)

        actback_scores = torch.zeros(x.shape[0], self.enc_steps, self.num_classes_actback, dtype=x.dtype)
        acts_scores = torch.zeros(x.shape[0], self.enc_steps, self.num_classes_acts, dtype=x.dtype)
        startend_scores = torch.zeros(x.shape[0], self.enc_steps, self.num_classes_startend, dtype=x.dtype)
        final_scores = torch.zeros(x.shape[0], self.enc_steps, self.num_classes_acts, dtype=x.dtype)

        for step in range(self.enc_steps):
            x_t = x[:, step]
            out = self.feature_extractor(x_t, torch.zeros(1))  # second input is optical flow, in our case will not be used

            actback_h_n, actback_c_n = self.actback(out, (actback_h_n, actback_c_n))
            acts_h_n, acts_c_n = self.acts(out, (acts_h_n, acts_c_n))
            startend_h_n, startend_c_n = self.startend(out, (startend_h_n, startend_c_n))

            actback_score = self.actback_classifier(actback_h_n)
            acts_score = self.acts_classifier(acts_h_n)
            startend_score = self.startend_classifier(startend_h_n)

            argmax_actions = utl.soft_argmax(acts_score).unsqueeze(1)
            argmax_startend = utl.soft_argmax(startend_score).unsqueeze(1)

            fusion_vect = torch.cat((actback_score, argmax_actions, argmax_startend), dim=1)

            out = self.final_classifier(fusion_vect)  # out.shape == (batch_size, num_classes_acts)

            actback_scores[:, step] = actback_score
            acts_scores[:, step] = actback_score
            startend_scores[:, step] = startend_score
            final_scores[:, step] = out
        return final_scores, actback_scores, acts_scores, startend_scores


    def step(self, camera_input, h_n, c_n):
        raise NotImplementedError