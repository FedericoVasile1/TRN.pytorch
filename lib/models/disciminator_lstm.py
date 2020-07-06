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
        self.num_classes = 2   # action and background
        self.enc_steps = args.enc_steps

        self.feature_extractor = build_feature_extractor(args)

        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
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