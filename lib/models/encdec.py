import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import build_feature_extractor

class EncDec(nn.Module):
    def __init__(self, args):
        super(EncDec, self).__init__()
        self.enc_hidden_size = args.hidden_size
        self.dec_hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.steps = 240 // 9

        self.feature_extractor = build_feature_extractor(args)

        if args.model == 'ENCDECLSTM':
            self.enc = nn.LSTM(input_size=self.feature_extractor.fusion_size,
                               hidden_size=self.enc_hidden_size,
                               num_layers=1,
                               batch_first=True,
                               dropout=args.dropout,
                               bidirectional=False)
            self.dec = nn.LSTMCell(input_size=self.enc_hidden_size+self.feature_extractor.fusion_size,
                                   hidden_size=self.dec_hidden_size)
        elif args.model == 'ENCDECGRU':
            raise NotImplementedError()
        else:
            raise Exception('Model ' + args.model + ' here is not supported')

        self.lin_proj = nn.Linear(self.enc_hidden_size, self.dec_hidden_size)

        self.classifier = nn.Linear(self.dec_hidden_size, self.num_classes)

    def forward(self, x):
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)

        for step in range(self.steps):
            x[:, step, :] = self.feature_extractor(x[:, step])

        h_ts, _ = self.enc(x)        # h_ts.shape == (batch_size, steps, enc_hidden_size)

        proj_h_ts = self.lin_proj(h_ts)     # proj_h_ts.shape == (batch_size, steps, dec_hidden_size)

        dec_h_n = torch.zeros(proj_h_ts.shape[0], proj_h_ts.shape[2]).to(proj_h_ts.dtype, proj_h_ts.device)
        dec_c_n = torch.zeros(proj_h_ts.shape[0], proj_h_ts.shape[2]).to(proj_h_ts.dtype, proj_h_ts.device)
        for step in range(self.steps):
            x_t = x[:, step]

            # perform scaled dot product attention
            attn_weights = dec_h_n.unsqueeze(1).bmm(proj_h_ts.permute(0, 2, 1))     # attn_weights.shape == (batch_size, 1, steps)
            attn_weights = attn_weights.div(math.sqrt(self.dec_hidden_size))
            attn_weights = F.softmax(attn_weights, dim=2)

            attn = attn_weights.bmm(proj_h_ts)      # attn.shape == (batch_size, 1, dec_hidden_size)
            attn = attn.squeeze(1)

            x_t = torch.cat(x_t, attn, dim=1)       # x_t.shape == (batch_size, input_size + dec_hidden_size)

            dec_h_n, dec_c_n = self.dec(x_t, (dec_h_n, dec_c_n))

            scores[:, step, :] = self.classifier(dec_h_n)

        return scores