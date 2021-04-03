import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.nhead = args.nhead
        self.num_layers = args.num_layers_transformer
        self.num_classes = args.num_classes
        self.steps = args.steps

        self.feature_extractor = build_feature_extractor(args)
        self.d_model = self.feature_extractor.fusion_size

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.nhead,
                                                   dropout=args.dropout,
                                                   dim_feedforward=self.d_model * 4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        transf_x = torch.zeros(x.shape[0], x.shape[1], self.feature_extractor.fusion_size).to(dtype=x.dtype,
                                                                                              device=x.device)

        for step in range(self.steps):
            transf_x[:, step] = self.feature_extractor(x[:, step])

        transf_x = self.transformer(transf_x.transpose(0, 1)).transpose(0, 1)

        for step in range(self.steps):
            scores[:, step] = self.classifier(transf_x[:, step])

        return scores