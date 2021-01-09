import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from .feature_extractor import build_feature_extractor


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, num_features)
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        z = Variable(self.pe[:, :seq_len], requires_grad=False)
        x = x + z
        return x


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        self.num_classes = args.num_classes
        self.steps = args.steps

        self.feature_extractor = build_feature_extractor(args)
        self.d_model = self.feature_extractor.fusion_size

        self.pe = PositionalEncoder(self.d_model, self.steps)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.nhead,
                                                   dropout=args.dropout,
                                                   dim_feedforward=self.d_model * 4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        transf_x = torch.zeros(x.shape[0], x.shape[1], self.feature_extractor.fusion_size,
                               dtype=x.dtype, device=x.device)

        for step in range(self.steps):
            transf_x[:, step] = self.feature_extractor(x[:, step])

        transf_x = self.pe(transf_x)
        transf_x = self.transformer(transf_x.transpose(0, 1)).transpose(0, 1)

        scores = self.classifier(transf_x)

        return scores