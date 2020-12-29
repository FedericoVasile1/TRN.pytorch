import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class _MyGRUCell(nn.GRUCell):
    """
    This model has been created for the only purpose of having a forward signature equal to the one
    of the LSTMCell model, by doing this we only need a single code pipeline for both lstm and gru models.
    """
    def forward(self, x, states):
        h_n, _ = states
        return super(_MyGRUCell, self).forward(x, h_n), torch.zeros(1)

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.steps = args.steps

        self.feature_extractor = build_feature_extractor(args)

        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        if 'LSTM' in args.model:
            self.rnn = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
            self.model = 'LSTM'
        elif 'GRU' in args.model:
            self.rnn = _MyGRUCell(self.feature_extractor.fusion_size, self.hidden_size)
            self.model = 'GRU'
        else:
            raise Exception('Model ' + args.model + ' here is not supported')
        self.drop_before = nn.Dropout(args.dropout)
        self.drop_after = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, heatmaps):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        h_n = torch.zeros(x.shape[0],
                          self.hidden_size,
                          device=x.device,
                          dtype=x.dtype)
        c_n = torch.zeros(x.shape[0],
                          self.hidden_size,
                          device=x.device,
                          dtype=x.dtype) if self.model == 'LSTM' else torch.zeros(1)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)

        for step in range(self.steps):
            x[:, step] = self.feature_extractor(x[:, step], heatmaps[:, step])

        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)

        for step in range(self.steps):
            h_n, c_n = self.rnn(x[:, step], (h_n, c_n))
            out = self.classifier(h_n)  # out.shape == (batch_size, num_classes)
            scores[:, step, :] = out

        return scores