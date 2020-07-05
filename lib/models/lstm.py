import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class LSTMmodel(nn.Module):
    def __init__(self, args):
        super(LSTMmodel, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps

        self.feature_extractor = build_feature_extractor(args)

        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        if 'ORACLE' in args.model:
            self.forward = self.oracle_forward
            self.step = self.oracle_step
        else:
            self.forward = self.base_forward
            self.step = self.base_step

    def base_forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        for step in range(self.enc_steps):
            x_t = x[:, step]
            out = self.feature_extractor(x_t, torch.zeros(1))  # second input is optical flow, in our case will not be used
            h_n, c_n = self.lstm(self.drop(out), (h_n, c_n))
            out = self.classifier(h_n)  # self.classifier(h_n).shape == (batch_size, num_classes)

            scores[:, step, :] = out
        return scores

    def base_step(self, camera_input, h_n, c_n):
        out = self.feature_extractor(camera_input, torch.zeros(1))
        h_n, c_n = self.lstm(out, (h_n, c_n))
        out = self.classifier(h_n)
        return out, h_n, c_n

    def oracle_forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        scores = torch.zeros(x.shape[0], self.num_classes, dtype=x.dtype)
        for step in range(self.enc_steps):
            x_t = x[:, step]
            out = self.feature_extractor(x_t, torch.zeros(1))  # second input is optical flow, in our case will not be used
            h_n, c_n = self.lstm(self.drop(out), (h_n, c_n))

        scores = self.classifier(h_n)  # scores.shape == (batch_size, num_classes)
        return scores

    def oracle_step(self, x, h_n, c_n):
        score = torch.zeros(x.shape[0], self.num_classes, dtype=x.dtype)
        out = self.feature_extractor(x, torch.zeros(1))
        h_n, c_n = self.lstm(self.drop(out), (h_n, c_n))
        score = self.classifier(h_n)
        return score, h_n, c_n