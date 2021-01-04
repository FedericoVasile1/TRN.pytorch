import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.d_model = self.feature_extractor.fusion_size
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        self.num_classes = args.num_classes
        self.steps = args.steps

        self.feature_extractor = build_feature_extractor(args)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.classifiers = {}
        for step in range(args.steps):
            self.classifiers[step] = nn.Linear(self.d_model, self.num_classes).cuda()

    def forward(self, x, heatmaps):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        transf_x = torch.zeros(x.shape[0], x.shape[1], self.feature_extractor.fusion_size,
                               dtype=x.dtype, device=x.device)

        for step in range(self.steps):
            transf_x[:, step] = self.feature_extractor(x[:, step], heatmaps[:, step])

        transf_x = self.transformer(transf_x.transpose(0, 1)).transpose(0, 1)

        for step in range(self.steps):
            out = self.classifiers[step](transf_x[:, step])  # out.shape == (batch_size, num_classes)
            scores[:, step, :] = out

        return scores