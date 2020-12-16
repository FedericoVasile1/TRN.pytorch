import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class BIDIRECTIONALRNN(nn.Module):
    def __init__(self, args):
        super(BIDIRECTIONALRNN, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.steps = args.steps
        self.use_heatmaps = args.use_heatmaps

        self.feature_extractor = build_feature_extractor(args)

        if args.model == 'BIDIRECTIONALLSTM':
            self.rnn = nn.LSTM(input_size=self.feature_extractor.fusion_size,
                               hidden_size=self.hidden_size,
                               num_layers=1,
                               batch_first=True,
                               dropout=args.dropout,
                               bidirectional=True)
        elif args.model == 'BIDIRECTIONALGRU':
            self.rnn = nn.GRU(input_size=self.feature_extractor.fusion_size,
                              hidden_size=self.hidden_size,
                              num_layers=1,
                              batch_first=True,
                              dropout=args.dropout,
                              bidirectional=True)
        else:
            raise Exception('Model ' + args.model + ' here is not supported')

        self.classifier = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x, heatmaps):
        # x.shape == heatmaps.shape == (batch_size, steps, 1024)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        transf_x = torch.zeros(x.shape[0],
                               x.shape[1],
                               (x.shape[2]*2) if self.use_heatmaps else x.shape[2]).to(dtype=x.dtype, device=x.device)

        for step in range(self.steps):
            transf_x[:, step, :] = self.feature_extractor(x[:, step], heatmaps[:, step])

        h_ts, _ = self.rnn(transf_x)        # h_ts.shape == (batch_size, enc_steps, 2*hidden_size)

        for step in range(self.steps):
            scores[:, step, :] = self.classifier(h_ts[:, step, :])

        return scores