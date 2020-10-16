import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class TWOLAYERSGRU(nn.Module):
    def __init__(self, args):
        super(TWOLAYERSGRU, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps

        self.feature_extractor = build_feature_extractor(args)

        self.rnn = nn.GRU(input_size=self.feature_extractor.fusion_size,
                          hidden_size=self.hidden_size,
                          num_layers=2,
                          batch_first=True,
                          dropout=args.dropout,
                          bidirectional=False)

        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, camera_input, motion_input):
        scores = torch.zeros(camera_input.shape[0], camera_input.shape[1], self.num_classes, dtype=camera_input.dtype)

        h_ts, _ = self.rnn(camera_input)        # h_ts.shape == (batch_size, enc_steps, hidden_dim)

        for step in range(self.enc_steps):
            scores[:, step, :] = self.classifier(h_ts[:, step, :])

        return scores