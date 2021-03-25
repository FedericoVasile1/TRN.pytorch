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

class Reduction(nn.Module):
    def __init__(self):
        super(Reduction, self).__init__()

    def forward(self, x):
        # x.shape == batch_size, numfeatmaps, TT, HH, WW
        return x.mean(dim=2)

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        # x.shape == batch_size, numfeatmaps, HH, WW
        return x.mean(dim=(2, 3))

class RNNmodel(nn.Module):
    def __init__(self, args):
        super(RNNmodel, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.steps = args.steps

        if 'mixed5c' in args.model_input or 'mixed4f' in args.model_input:
            self.reduction = nn.Sequential(
                Reduction(),
                GlobalAvgPool(),
            )
        else:
            self.reduction = nn.Identity()

        self.feature_extractor = build_feature_extractor(args)

        if args.model == 'LSTM':
            self.rnn = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
            self.model = 'LSTM'
        elif args.model == 'GRU':
            self.rnn = _MyGRUCell(self.feature_extractor.fusion_size, self.hidden_size)
            self.model = 'GRU'
        else:
            raise Exception('Model ' + args.model + ' here is not supported')
        self.drop_before = nn.Dropout(args.dropout)
        self.drop_after = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
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
            x_t = x[:, step]

            x_t = self.reduction(x_t)
            transf_x_t = self.feature_extractor(x_t)
            if step == 0:
                transf_x_t = self.drop_before(transf_x_t)

            h_n, c_n = self.rnn(transf_x_t, (h_n, c_n))
            out = self.classifier(self.drop_after(h_n))  # out.shape == (batch_size, num_classes)

            scores[:, step] = out
        return scores

    def step(self, x_t, h_n, c_n):
        x_t = self.reduction(x_t)
        out = self.feature_extractor(x_t)

        # to check if we are at the first timestep of the sequence, we exploit the fact that at the first
        #  timestep the hidden state is all zeros.
        appo = torch.zeros_like(out)
        if torch.all(torch.eq(appo, out)):
            out = self.drop_before(out)

        h_n, c_n = self.rnn(out, (h_n, c_n))
        out = self.classifier(self.drop_after(h_n))
        return out, h_n, c_n