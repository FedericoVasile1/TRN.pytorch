import torch
import torch.nn as nn
from torchvision import models

from .feature_extractor import build_feature_extractor

def fc_relu(in_features, out_features, inplace=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(inplace=inplace),
    )

class MyTRN(nn.Module):
    def __init__(self, args):
        super(MyTRN, self).__init__()
        self.hidden_size_dec = args.hidden_size_dec
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps

        self.feature_extractor = build_feature_extractor(args)

        self.enc_drop = nn.Dropout(args.dropout)
        self.enc = nn.LSTMCell(self.feature_extractor.fusion_size, self.feature_extractor.fusion_size)
        self.enc_classifier = nn.Linear(self.feature_extractor.fusion_size, self.num_classes)

        self.dec_drop = nn.Dropout(args.dropout)
        self.dec = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size_dec)
        self.dec_classifier = nn.Linear(self.hidden_size_dec, self.feature_extractor.fusion_size)

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        batch_size = x.shape[0]
        enc_scores = torch.zeros(batch_size, x.shape[1], self.num_classes, dtype=x.dtype)
        dec_scores = torch.zeros(batch_size, x.shape[1], self.dec_steps, self.feature_extractor.fusion_size, dtype=x.dtype)

        for enc_step in range(self.enc_steps):
            x_step = x[:, enc_step]
            feat_vects = self.feature_extractor(x_step, torch.zeros(1))

            dec_h_n = torch.zeros(batch_size, self.hidden_size_dec, device=x.device, dtype=x.dtype)
            dec_c_n = torch.zeros(batch_size, self.hidden_size_dec, device=x.device, dtype=x.dtype)
            input = feat_vects
            out_buf = torch.zeros(batch_size, self.feature_extractor.fusion_size, dtype=x.dtype, device=x.device)
            for dec_step in range(self.dec_steps):
                dec_h_n, dec_c_n = self.dec(input, (dec_h_n, dec_c_n))
                out = self.dec_classifier(self.drop(dec_h_n))

                dec_scores[:, enc_step, dec_step, :] = out
                input = out
                out_buf += out

            enc_h_n = out / self.dec_steps
            enc_c_n = torch.zeros(batch_size, self.feature_extractor.fusion_size, device=x.device, dtype=x.dtype)
            enc_h_n, enc_c_n = self.enc(feat_vects, (enc_h_n, enc_c_n))
            preds = self.enc_classifier(self.enc_drop(enc_h_n))
            enc_scores[:, enc_step, :] = preds

        return enc_scores, dec_scores

    # TODO
    def step(self, camera_input, enc_h_n, enc_c_n):
        # camera_input.shape == (batch_size, feat_vect_dim) == (1, feat_vect_dim)
        enc_score = torch.zeros(1, self.num_classes, dtype=camera_input.dtype)
        dec_scores = torch.zeros(self.dec_steps, 1, self.future_size, dtype=camera_input.dtype)

        dec_h_n = torch.zeros(camera_input.shape[0], self.hidden_size_dec, device=camera_input.device, dtype=camera_input.dtype)
        dec_c_n = torch.zeros(camera_input.shape[0], self.hidden_size_dec, device=camera_input.device, dtype=camera_input.dtype)
        feat_vect = self.feature_extractor(camera_input, torch.zeros(1))
        future_input = feat_vect
        for dec_step in range(self.dec_steps):
            dec_h_n, dec_c_n = self.dec(future_input, (dec_h_n, dec_c_n))
            future_input = self.dec_transf(dec_h_n)

            dec_scores[dec_step] = future_input

        feat_vect_plus_future = torch.cat((feat_vect, future_input), dim=1)  # shape == (1, fusion_size)
        enc_h_n, enc_c_n = self.enc(feat_vect_plus_future, (enc_h_n, enc_c_n))
        enc_score = self.classifier(enc_h_n)

        return enc_score, dec_scores, enc_h_n, enc_c_n