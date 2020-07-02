import torch
import torch.nn as nn
from torchvision import models

from .feature_extractor import build_feature_extractor

class MyTRN(nn.Module):
    def __init__(self, args):
        super(MyTRN, self).__init__()
        self.hidden_size_enc = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.future_size = self.num_classes

        self.feature_extractor = build_feature_extractor(args)
        self.fusion_size = self.feature_extractor.fusion_size + self.num_classes

        self.dec_drop = nn.Dropout(args.dropout)
        self.dec = nn.LSTMCell(self.num_classes, self.feature_extractor.fusion_size)
        self.dec_transf = nn.Linear(self.feature_extractor.fusion_size, self.num_classes)

        self.enc_drop = nn.Dropout(args.dropout)
        self.enc = nn.LSTMCell(self.fusion_size, self.hidden_size_enc)
        self.classifier = nn.Linear(self.hidden_size_enc, self.num_classes)

    def forward(self, x):  # x.shape == (batch_size, enc_steps, C, chunk_size, H, W)
        enc_scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        dec_scores = torch.zeros(x.shape[0], x.shape[1], self.dec_steps, self.num_classes, dtype=x.dtype)

        enc_h_n = torch.zeros(x.shape[0], self.hidden_size_enc, device=x.device, dtype=x.dtype)
        enc_c_n = torch.zeros(x.shape[0], self.hidden_size_enc, device=x.device, dtype=x.dtype)
        for enc_step in range(self.enc_steps):
            # decoder pass
            future_input = torch.zeros(x.shape[0], self.num_classes, device=x.device, dtype=x.dtype)
            feat_vects = self.feature_extractor(x[:, enc_step], torch.zeros(1))
            dec_h_n = feat_vects       # the feature vector is the initial hidden state of the decoder
            dec_c_n = torch.zeros_like(dec_h_n).to(device=x.device, dtype=x.dtype)
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
        dec_scores = torch.zeros(self.dec_steps, 1, self.num_classes, dtype=camera_input.dtype)

        future_input = torch.zeros(camera_input.shape[0], self.num_classes, device=camera_input.device, dtype=camera_input.dtype)
        feat_vect = self.feature_extractor(camera_input, torch.zeros(1))
        dec_h_n = feat_vect
        dec_c_n = torch.zeros_like(dec_h_n).to(device=camera_input.device, dtype=camera_input.dtype)
        for dec_step in range(self.dec_steps):
            dec_h_n, dec_c_n = self.dec(future_input, (dec_h_n, dec_c_n))
            future_input = self.dec_transf(dec_h_n)

            dec_scores[dec_step] = future_input

        feat_vect_plus_future = torch.cat((feat_vect, future_input), dim=1)  # shape == (1, fusion_size)
        enc_h_n, enc_c_n = self.enc(feat_vect_plus_future, (enc_h_n, enc_c_n))
        enc_score = self.classifier(enc_h_n)

        return enc_score, dec_scores, enc_h_n, enc_c_n