import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class RULSTM(nn.Module):
    def __init__(self, args):
        super(RULSTM, self).__init__()
        self.r_hidden_size = args.hidden_size     # must be the same as the feature vector dim
        self.u_hidden_size = args.hidden_size_dec
        self.num_classes = args.num_classes
        self.r_steps = args.enc_steps
        self.u_steps = args.dec_steps

        self.feature_extractor = build_feature_extractor(args)

        self.r_lstm = nn.LSTMCell(self.feature_extractor.fusion_size, self.r_hidden_size)
        self.r_classifier = nn.Linear(self.r_hidden_size, self.num_classes)

        self.h_lin_trasf = nn.Sequential(
            nn.Linear(self.r_hidden_size, self.u_hidden_size),
            nn.ReLU(),
        )
        self.c_lin_trasf = nn.Sequential(
            nn.Linear(self.r_hidden_size, self.u_hidden_size),
            nn.ReLU(),
        )
        self.u_lstm = nn.LSTMCell(self.feature_extractor.fusion_size, self.u_hidden_size)
        self.out_lin_transf = nn.Sequential(
            nn.Linear(self.u_hidden_size, self.feature_extractor.fusion_size),
            nn.ReLU()
        )
        self.u_classifier = nn.Linear(self.feature_extractor.fusion_size, self.num_classes)

        self.final_classifier = nn.Linear(self.num_classes * 2, self.num_classes)

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        r_h_n = torch.zeros(x.shape[0], self.r_hidden_size, device=x.device, dtype=x.dtype)
        r_c_n = torch.zeros(x.shape[0], self.r_hidden_size, device=x.device, dtype=x.dtype)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        t_scores = torch.zeros(x.shape[0], self.u_steps, self.feature_extractor.fusion_size, dtype=x.dtype)
        for step in range(self.r_steps):
            x_t = x[:, step]
            feat_vects = self.feature_extractor(x_t, torch.zeros(1))  # second input is optical flow, in our case will not be used
            r_h_n, r_c_n = self.r_lstm(feat_vects, (r_h_n, r_c_n))

        u_h_n = self.h_lin_trasf(r_h_n)
        u_c_n = self.c_lin_trasf(r_c_n)
        next_in = feat_vects
        for step in range(self.u_steps):
            u_h_n, u_c_n = self.u_lstm(next_in, (u_h_n, u_c_n))
            next_in = self.out_lin_transf(u_h_n)        # TODO: SUPERVISION HERE
            t_scores[:, step] = next_in

        out = self.u_classifier(next_in)
        for step in range(self.r_steps):
            scores[:, step] = out

        return scores, t_scores

    def step(self, camera_input, h_n, c_n):
        raise NotImplementedError