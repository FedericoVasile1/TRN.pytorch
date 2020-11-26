import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import build_feature_extractor

class _MyGRUCell(nn.GRUCell):
    """
    This model has been created for the only purpose of having a forward signature equal to the one
    of the LSTMCell model, by doing this we only need a single code pipeline for both lstm and gru models.
    """
    def forward(self, x, states):
        h_n, _ = states
        return super(_MyGRUCell, self).forward(x, h_n), torch.zeros(1)

class EncDec(nn.Module):
    def __init__(self, args):
        super(EncDec, self).__init__()
        self.enc_hidden_size = args.hidden_size
        self.dec_hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.steps = args.steps
        self.model = args.model

        self.feature_extractor = build_feature_extractor(args)

        # ONLY THE ENCODER is bidirectional
        self.is_bidirectional = 'BIDIRECTIONAL' in args.model
        if args.model == 'ENCDECLSTM' or args.model == 'ENCDECBIDIRECTIONALLSTM':
            self.enc = nn.LSTM(input_size=self.feature_extractor.fusion_size,
                               hidden_size=self.enc_hidden_size,
                               num_layers=1,
                               batch_first=True,
                               dropout=args.dropout,
                               bidirectional=self.is_bidirectional)
            self.dec = nn.LSTMCell(input_size=self.enc_hidden_size*(2 if self.is_bidirectional else 1)+self.feature_extractor.fusion_size,
                                   hidden_size=self.dec_hidden_size)
        elif args.model == 'ENCDECGRU' or args.model == 'ENCDECBIDIRECTIONALGRU':
            self.enc = nn.GRU(input_size=self.feature_extractor.fusion_size,
                              hidden_size=self.enc_hidden_size,
                              num_layers=1,
                              batch_first=True,
                              dropout=args.dropout,
                              bidirectional=self.is_bidirectional)
            self.dec = _MyGRUCell(input_size=self.enc_hidden_size*(2 if self.is_bidirectional else 1)+self.feature_extractor.fusion_size,
                                  hidden_size=self.dec_hidden_size)
        else:
            raise Exception('Model ' + args.model + ' here is not supported')

        self.lin_proj_e2d = nn.Linear(self.enc_hidden_size*(2 if self.is_bidirectional else 1), self.dec_hidden_size)
        self.lin_proj_d2e = nn.Linear(self.dec_hidden_size, self.enc_hidden_size*(2 if self.is_bidirectional else 1))

        self.classifier = nn.Linear(self.dec_hidden_size, self.num_classes)

    def forward(self, x):
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)

        for step in range(self.steps):
            x[:, step, :] = self.feature_extractor(x[:, step])
        h_ts, _ = self.enc(x)        # h_ts.shape == (batch_size, steps, enc_hidden_size)
        proj_h_ts = self.lin_proj_e2d(h_ts)     # proj_h_ts.shape == (batch_size, steps, dec_hidden_size)

        dec_h_n = torch.zeros(proj_h_ts.shape[0],proj_h_ts.shape[2]).to(dtype=proj_h_ts.dtype,
                                                                        device=proj_h_ts.device)
        dec_c_n = torch.zeros(proj_h_ts.shape[0], proj_h_ts.shape[2]).to(dtype=proj_h_ts.dtype,
                                                                         device=proj_h_ts.device) if 'LSTM' in self.model else torch.zeros(1)
        for step in range(self.steps):
            x_t = x[:, step]

            # perform scaled dot product attention
            attn_weights = dec_h_n.unsqueeze(1).bmm(proj_h_ts.permute(0, 2, 1))     # attn_weights.shape == (batch_size, 1, steps)
            attn_weights = attn_weights.div(math.sqrt(self.dec_hidden_size))
            attn_weights = F.softmax(attn_weights, dim=2)

            attn = attn_weights.bmm(proj_h_ts)      # attn.shape == (batch_size, 1, dec_hidden_size)
            attn = attn.squeeze(1)

            attn = self.lin_proj_d2e(attn)

            x_t = torch.cat((x_t, attn), dim=1)       # x_t.shape == (batch_size, input_size + enc_hidden_size)

            dec_h_n, dec_c_n = self.dec(x_t, (dec_h_n, dec_c_n))

            scores[:, step, :] = self.classifier(dec_h_n)

        return scores, attn_weights.squeeze(1)