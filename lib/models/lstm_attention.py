import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .feature_extractor import build_feature_extractor

'''
Actually, this model can only work in end to end mode(i.e. starting from the frames), since 
we do not have the extracted featrure maps
'''
class LSTMAttention(nn.Module):
    def __init__(self, args, dtype=torch.float32):
        super(LSTMAttention, self).__init__()
        if args.feature_extractor != 'RESNET2+1D':
            raise Exception('wrong feature_extractor option')

        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps


        # The RESNET2+1D outputs feature maps of shape (batch_size, 512, 1, 7, 7)
        self.feature_extractor = build_feature_extractor(args)
        self.numfeatmaps = 512       # HARD-CODED; DO NOT MODIFY
        self.input_size = 7 * 7  # HARD-CODED; DO NOT MODIFY

        self.numfeatmaps_to_hidden = nn.Linear(self.numfeatmaps, self.hidden_size)

        # i.e. the input to the lstm is the concatenation of the input vector and context vector
        self.lstm = nn.LSTMCell(self.feature_extractor.fusion_size + self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        # Attention weights

    def forward(self, chunks):
        # feat_maps.shape == (batch_size, enc_steps, 3, 6, 112, 112)
        h_n = torch.zeros(chunks.shape[0], self.hidden_size, device=chunks.device, dtype=chunks.dtype)
        c_n = torch.zeros(chunks.shape[0], self.hidden_size, device=chunks.device, dtype=chunks.dtype)
        scores = torch.zeros(chunks.shape[0], chunks.shape[1], self.num_classes, dtype=chunks.dtype)

        for step in range(self.enc_steps):
            chunks_t = chunks[:, step]
            feat_maps = self.feature_extractor(chunks_t, torch.zeros(1))  # second input is optical flow, in our case will not be used

            # feat_maps.shape == (batch_size, 512, 1, 7, 7)
            feat_maps = feat_maps.squeeze(2)
            feat_maps = feat_maps.flatten(start_dim=2)      # flatten feature maps to feature vectors

            # project the number of channels down to hidden_size, i.e. from 512 to hidden_size
            feat_maps_projected = self.numfeatmaps_to_hidden(feat_maps.permute(0, 2, 1))
            # feat_maps_projected.shape == (batch_size, 7 * 7, hidden_size)
            # Permute back to original axis
            feat_maps_projected = feat_maps_projected.permute(0, 2, 1)

            # perform **scaled** dot-product attention
            attn_weights = h_n.unsqueeze(1).bmm(feat_maps_projected)  # attn_weights.shape == (batch_size, 1, 7 * 7)
            attn_weights = attn_weights.div(math.sqrt(self.hidden_size))

            # get probability-normalized weights
            attn_weights = F.softmax(attn_weights, dim=2)

            # compute context vector
            attn = feat_maps_projected.bmm(attn_weights.permute(0, 2, 1))   # attn.shape == (batch_size, hidden_size, 1)
            attn = attn.squeeze(2)

            # TODO: for each timestep return also the attention weights; we will need them for the visualization
            attn_weights = attn_weights.squeeze(1).view(attn_weights.shape[0], 7, 7)    # attn_weights.shape == (batch_size, 7, 7)

            input = feat_maps.mean(dim=2)       # perform global average pooling. input.shape == (batch_size, 512)
            input = torch.cat((input, attn), dim=1)
            h_n, c_n = self.lstm(input, (h_n, c_n))
            out = self.classifier(h_n)  # out.shape == (batch_size, num_classes)

            scores[:, step, :] = out
        return scores

    def step(self, camera_input, h_n, c_n):
        raise NotImplementedError