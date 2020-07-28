import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor


def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
    pad = (kernel_size - 1) * dilation
    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)

class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
        self.conv1 = nn.Sequential(
            CausalConv1d(1, 1, 2, 1),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )
        self.conv2 = nn.Sequential(
            CausalConv1d(1, 1, 2, 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )
        self.conv3 = nn.Sequential(
            CausalConv1d(1, 1, 2, 4),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        out1 = self.conv1(x)
        x = torch.cat((x, torch.ones(out1.shape[0], 1, 1).to(out1.device)), dim=2)
        out1 += x
        out2 = self.conv2(out1)
        out1 = torch.cat((out1, torch.ones(out2.shape[0], 1, 2).to(out2.device)), dim=2)
        out2 += out1
        out3 = self.conv3(out2)
        out2 = torch.cat((out2, torch.ones(out3.shape[0], 1, 4).to(out3.device)), dim=2)
        out3 += out2
        return out3.squeeze(1)

class LSTMmodel(nn.Module):
    def __init__(self, args):
        super(LSTMmodel, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps

        self.feature_extractor = build_feature_extractor(args)

        self.tcn = TCN(args)

        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTMCell(self.feature_extractor.fusion_size + 7, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        for step in range(self.enc_steps):
            x_t = x[:, step]
            out = self.feature_extractor(x_t, torch.zeros(1))  # second input is optical flow, in our case will not be used
            out = self.tcn(out)

            h_n, c_n = self.lstm(out, (h_n, c_n))
            out = self.classifier(self.drop(h_n))  # out.shape == (batch_size, num_classes)

            scores[:, step, :] = out
        return scores

    def step(self, camera_input, h_n, c_n):
        out = self.feature_extractor(camera_input, torch.zeros(1))
        h_n, c_n = self.lstm(out, (h_n, c_n))
        out = self.classifier(h_n)
        return out, h_n, c_n