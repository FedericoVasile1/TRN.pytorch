import torch
import torch.nn as nn
from torchvision import models

from .feature_extractor import build_feature_extractor

def fc_relu(in_features, out_features, inplace=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(inplace=inplace),
    )

class DiscriminatorLSTM(nn.Module):
    def __init__(self, args):
        super(DiscriminatorLSTM, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes   # == 2, i.e. action and background
        self.enc_steps = args.enc_steps
        self.two_layers_lstm = args.two_layers_lstm

        self.feature_extractor = build_feature_extractor(args)

        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        if self.two_layers_lstm:
            self.drop2 = nn.Dropout(args.dropout)
            self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        if self.two_layers_lstm:
            h_n_2 = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
            c_n_2 = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        for step in range(self.enc_steps):
            x_t = x[:, step]
            out = self.feature_extractor(x_t, torch.zeros(1))  # second input is optical flow, in our case will not be used
            h_n, c_n = self.lstm(self.drop(out), (h_n, c_n))
            if self.two_layers_lstm:
                h_n_2, c_n_2 = self.lstm2(self.drop2(h_n), (h_n_2, c_n_2))
            out = self.classifier(h_n_2 if self.two_layers_lstm else h_n)  # out.shape == (batch_size, num_classes)

            scores[:, step, :] = out
        return scores

    def step(self, x, h_n, c_n):
        out = self.feature_extractor(x, torch.zeros(1))
        h_n, c_n = self.lstm(out, (h_n, c_n))
        out = self.classifier(h_n)
        return out, h_n, c_n