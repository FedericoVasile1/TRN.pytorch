import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class FutureLSTM(nn.Module):
    def __init__(self, args):
        super(FutureLSTM, self).__init__()
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps

        self.feature_extractor = build_feature_extractor(args)

        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.feature_extractor.fusion_size)

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        scores = torch.zeros(x.shape[0], x.shape[1], self.dec_steps, x.shape[2])
        for step_e in range(self.enc_steps):
            h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
            c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)

            x_t = x[:, step_e]
            out = self.feature_extractor(x_t,
                                         torch.zeros(1))  # second input is optical flow, in our case will not be used
            for step_d in range(self.dec_steps):
                h_n, c_n = self.lstm(self.drop(out), (h_n, c_n))
                out = self.classifier(h_n)  # self.classifier(h_n).shape == (batch_size, feat_vect_dim)

                scores[:, step_e, step_d, :] = out
        return scores

