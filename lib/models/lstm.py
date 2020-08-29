import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class LSTMmodel(nn.Module):
    def __init__(self, args):
        super(LSTMmodel, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps

        self.feature_extractor = build_feature_extractor(args)

        self.lstm = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.drop = nn.Identity()#nn.Dropout(args.dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, camera_input, motion_input):
        # camera_input.shape == (batch_size, enc_steps, feat_vect_dim)
        h_n = torch.zeros(camera_input.shape[0], self.hidden_size, device=camera_input.device, dtype=camera_input.dtype)
        c_n = torch.zeros(camera_input.shape[0], self.hidden_size, device=camera_input.device, dtype=camera_input.dtype)
        scores = torch.zeros(camera_input.shape[0], camera_input.shape[1], self.num_classes, dtype=camera_input.dtype)
        for step in range(self.enc_steps):
            camera_input_t = camera_input[:, step]
            motion_input_t = motion_input[:, step]
            out = self.feature_extractor(camera_input_t, motion_input_t)

            h_n, c_n = self.lstm(out, (h_n, c_n))
            out = self.classifier(self.drop(h_n))  # out.shape == (batch_size, num_classes)

            scores[:, step, :] = out
        return scores

    def step(self, camera_input_t, motion_input_t, h_n, c_n):
        out = self.feature_extractor(camera_input_t, motion_input_t)
        h_n, c_n = self.lstm(out, (h_n, c_n))
        out = self.classifier(h_n)
        return out, h_n, c_n