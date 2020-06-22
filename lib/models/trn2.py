import torch
import torch.nn as nn
from torchvision import models

class TRN2V2(nn.Module):
    def __init__(self, args):
        super(TRN2V2, self).__init__()
        self.hidden_size_enc = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.future_size = self.num_classes
        FEAT_VECT_DIM = args.feat_vect_dim
        self.fusion_size = FEAT_VECT_DIM + self.num_classes

        self.dec_drop = nn.Dropout(args.dropout)
        self.dec = nn.LSTMCell(self.num_classes, FEAT_VECT_DIM)
        self.dec_transf = nn.Linear(FEAT_VECT_DIM, self.num_classes)

        self.enc_drop = nn.Dropout(args.dropout)
        self.enc = nn.LSTMCell(self.fusion_size, self.hidden_size_enc)
        self.classifier = nn.Linear(self.hidden_size_enc, self.num_classes)

    def forward(self, x):  # x.shape == (batch_size, enc_steps, feat_vect_dim)
        enc_scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        dec_scores = torch.zeros(x.shape[0], x.shape[1], self.dec_steps, self.num_classes)

        enc_h_n = torch.zeros(x.shape[0], self.hidden_size_enc, device=x.device, dtype=x.dtype)
        enc_c_n = torch.zeros(x.shape[0], self.hidden_size_enc, device=x.device, dtype=x.dtype)
        for enc_step in range(self.enc_steps):
            # decoder pass
            future_input = torch.zeros(x.shape[0], self.num_classes, device=x.device, dtype=x.dtype)
            dec_h_n = x[:, enc_step, :]   # the feature vector is the initial hidden state of the decoder
            dec_c_n = torch.zeros_like(dec_h_n).to(device=x.device, dtype=x.dtype)
            for dec_step in range(self.dec_steps):
                dec_h_n, dec_c_n = self.dec(self.dec_drop(future_input), (dec_h_n, dec_c_n))
                future_input = self.dec_transf(dec_h_n)

                dec_scores[:, enc_step, dec_step, :] = future_input

            # encoder pass
            feat_vect = x[:, enc_step, :]
            feat_vect_plus_future = torch.cat((feat_vect, future_input), dim=1)   # shape == (batch_size, fusion_size)
            enc_h_n, enc_c_n = self.enc(self.enc_drop(feat_vect_plus_future), (enc_h_n, enc_c_n))
            out = self.classifier(enc_h_n)

            enc_scores[:, enc_step, :] = out

        return enc_scores, dec_scores

    def step(self, camera_input, enc_h_n, enc_c_n):
        enc_score = torch.zeros(1, self.num_classes, dtype=camera_input.dtype)
        dec_scores = torch.zeros(self.dec_steps, 1, self.num_classes, dtype=camera_input.dtype)

        future_input = torch.zeros(camera_input.shape[0], self.num_classes, device=camera_input.device, dtype=camera_input.dtype)
        dec_h_n = camera_input
        dec_c_n = torch.zeros_like(dec_h_n).to(device=camera_input.device, dtype=camera_input.dtype)
        for dec_step in range(self.dec_steps):
            dec_h_n, dec_c_n = self.dec(future_input, (dec_h_n, dec_c_n))
            future_input = self.dec_transf(dec_h_n)

            dec_scores[dec_step] = future_input

        feat_vect_plus_future = torch.cat((camera_input, future_input), dim=1)  # shape == (batch_size, fusion_size)
        enc_h_n, enc_c_n = self.enc(feat_vect_plus_future, (enc_h_n, enc_c_n))
        enc_score = self.classifier(enc_h_n)

        return enc_score, dec_scores, enc_h_n, enc_c_n

class TRN2V2E2E(nn.Module):
    def __init__(self, args):
        super(TRN2V2E2E, self).__init__()
        self.hidden_size_enc = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.future_size = self.num_classes

        self.feat_extr = models.video.r2plus1d_18(pretrained=True)
        FEAT_VECT_DIM = self.feat_extr.fc.in_features
        self.feat_extr.fc = nn.Identity()
        self.fusion_size = FEAT_VECT_DIM + self.num_classes

        for param in self.feat_extr.parameters():
            param.requires_grad = False

        self.dec_drop = nn.Dropout(args.dropout)
        self.dec = nn.LSTMCell(self.num_classes, FEAT_VECT_DIM)
        self.dec_transf = nn.Linear(FEAT_VECT_DIM, self.num_classes)

        self.enc_drop = nn.Dropout(args.dropout)
        self.enc = nn.LSTMCell(self.fusion_size, self.hidden_size_enc)
        self.classifier = nn.Linear(self.hidden_size_enc, self.num_classes)

    def forward(self, x):  # x.shape == (batch_size, enc_steps, C, chunk_size, H, W)
        enc_scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)
        dec_scores = torch.zeros(x.shape[0], x.shape[1], self.dec_steps, self.num_classes, dtype=x.dtype)

        enc_h_n = torch.zeros(x.shape[0], self.hidden_size_enc, device=x.device, dtype=x.dtype)
        enc_c_n = torch.zeros(x.shape[0], self.hidden_size_enc, device=x.device, dtype=x.dtype)
        for enc_step in range(x.shape[1]):
            # decoder pass
            future_input = torch.zeros(x.shape[0], self.num_classes, device=x.device, dtype=x.dtype)
            feat_vects = self.feat_extr(x[:, enc_step])
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
        feat_vect = self.feat_extr(camera_input)
        dec_h_n = feat_vect
        dec_c_n = torch.zeros_like(dec_h_n).to(device=camera_input.device, dtype=camera_input.dtype)
        for dec_step in range(self.dec_steps):
            dec_h_n, dec_c_n = self.dec(future_input, (dec_h_n, dec_c_n))
            future_input = self.dec_transf(dec_h_n)

            dec_scores[dec_step] = future_input

        feat_vect_plus_future = torch.cat((camera_input, future_input), dim=1)  # shape == (batch_size, fusion_size)
        enc_h_n, enc_c_n = self.enc(feat_vect_plus_future, (enc_h_n, enc_c_n))
        enc_score = self.classifier(enc_h_n)

        return enc_score, dec_scores, enc_h_n, enc_c_n