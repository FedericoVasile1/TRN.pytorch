import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class BIDIRECTIONALGRU(nn.Module):
    def __init__(self, args):
        super(BIDIRECTIONALGRU, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.steps = args.steps

        self.feature_extractor = build_feature_extractor(args)

        self.rnn = nn.GRU(input_size=self.feature_extractor.fusion_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=True,
                          dropout=args.dropout,
                          bidirectional=True)

        self.classifier = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x):
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)

        for step in range(self.steps):
            x[:, step, :] = self.feature_extractor(x[:, step])

        h_ts, _ = self.rnn(x)        # h_ts.shape == (batch_size, enc_steps, hidden_dim)

        for step in range(self.steps):
            scores[:, step, :] = self.classifier(h_ts[:, step, :])

        return scores