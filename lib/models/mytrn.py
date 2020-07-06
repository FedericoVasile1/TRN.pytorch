import torch
import torch.nn as nn
from torchvision import models

from .feature_extractor import build_feature_extractor

class MyTRN(nn.Module):
    def __init__(self, args):
        super(MyTRN, self).__init__()
        self.hidden_size_enc = args.hidden_size
        self.hidden_size_dec = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps

        self.feature_extractor = build_feature_extractor(args)
        self.future_size = self.feature_extractor.fusion_size
        self.fusion_size = self.future_size * 2

        self.enc_drop = nn.Dropout(args.dropout)
        self.enc = nn.LSTMCell(self.fusion_size, self.hidden_size_enc)
        self.enc_classifier = nn.Linear(self.hidden_size_enc, self.num_classes)

        self.dec_drop = nn.Dropout(args.dropout)
        self.dec = nn.LSTMCell(self.future_size, self.hidden_size_dec)
        self.dec_classifier = nn.Linear(self.hidden_size_dec, self.num_classes)
        self.hidden_to_future = nn.Linear(self.hidden_size_dec, self.future_size)

    def forward(self, x): # x.shape == (batch_size, enc_steps, feat_vect_dim)
        batch_size = x.shape[0]
        enc_scores = torch.zeros(batch_size, x.shape[1], self.num_classes, dtype=x.dtype)
        dec_scores = torch.zeros(batch_size, x.shape[1], self.dec_steps, self.num_classes, dtype=x.dtype)

        enc_h_n = torch.zeros(batch_size, self.hidden_size_enc, device=x.device, dtype=x.dtype)
        enc_c_n = torch.zeros(batch_size, self.hidden_size_enc, device=x.device, dtype=x.dtype)
        future_input = torch.zeros(batch_size, self.future_size)
        for enc_step in range(self.enc_steps):
            feat_vects = self.feature_extractor(x[:, enc_step], torch.zeros(1))
            feat_vects_plus_future = torch.cat((feat_vects, future_input), dim=1)
            enc_h_n, enc_c_n = self.enc(self.enc_drop(feat_vects_plus_future), (enc_h_n, enc_c_n))
            out = self.enc_classifier(enc_h_n)
            enc_scores[:, enc_step, :] = out

            dec_h_n = enc_h_n
            dec_c_n = enc_c_n
            future_input = torch.zeros(batch_size, self.future_size)
            for dec_step in range(self.dec_steps):
                dec_h_n, dec_c_n = self.dec(self.dec_drop(future_input), (dec_h_n, dec_c_n))
                out = self.dec_classifier(dec_h_n)
                dec_scores[:, enc_step, dec_step, :] = out

                future_input = future_input + self.hidden_to_future(dec_h_n)
            future_input /= self.dec_steps
        return enc_scores, dec_scores

    def forward2(self, x):  # x.shape == (batch_size, enc_steps, C, chunk_size, H, W)
        enc_scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        dec_scores = torch.zeros(x.shape[0], x.shape[1], self.dec_steps, self.future_size, dtype=x.dtype)

        enc_h_n = torch.zeros(x.shape[0], self.hidden_size_enc, device=x.device, dtype=x.dtype)
        enc_c_n = torch.zeros(x.shape[0], self.hidden_size_enc, device=x.device, dtype=x.dtype)
        for enc_step in range(self.enc_steps):
            # decoder pass
            dec_h_n = torch.zeros(x.shape[0], self.hidden_size_dec, device=x.device, dtype=x.dtype)
            dec_c_n = torch.zeros(x.shape[0], self.hidden_size_dec, device=x.device, dtype=x.dtype)
            feat_vects = self.feature_extractor(x[:, enc_step], torch.zeros(1))
            future_input = feat_vects
            for dec_step in range(self.dec_steps):
                dec_h_n, dec_c_n = self.dec(self.dec_drop(future_input), (dec_h_n, dec_c_n))
                future_input = self.dec_transf(dec_h_n)

                dec_scores[:, enc_step, dec_step, :] = future_input

            # encoder pass
            feat_vects_plus_future = torch.cat((feat_vects, future_input), dim=1)   # shape == (batch_size, fusion_size)
            enc_h_n, enc_c_n = self.enc(self.enc_drop(feat_vects_plus_future), (enc_h_n, enc_c_n))
            out = self.classifier(enc_h_n)

            enc_scores[:, enc_step, :] = out

        return enc_scores, dec_scores

    def step(self, camera_input, enc_h_n, enc_c_n):     # camera_input.shape == (1, C, chunk_size, H, W)
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