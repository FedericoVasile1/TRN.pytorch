import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .feature_extractor import build_feature_extractor

class SelfAttention(nn.Module):
    def __init__(self, feat_maps_channels, num_filters):
        super(SelfAttention, self).__init__()

        self.queries_conv = nn.Conv2d(feat_maps_channels, num_filters, 1)
        self.keys_conv = nn.Conv2d(feat_maps_channels, num_filters, 1)
        self.values_conv = nn.Conv2d(feat_maps_channels, num_filters, 1)
        self.final_conv = nn.Conv2d(num_filters, feat_maps_channels, 1)

    def forward(self, x):
        '''
        Check https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture13.pdf at slide 81 for
        details about computations
        :param x: the output of the feature extractor; feature maps of shape (batch_size, C, H, W)
        :return attn: tensor of shape (batch_size, C, H, W)
                attn_weights: tensor of shape(batch_size, H*W, H*W)
        '''
        _, _, H, W = x.shape

        queries = self.queries_conv(x).flatten(start_dim=2)
        keys = self.keys_conv(x).flatten(start_dim=2)

        queries = queries.permute(0, 2, 1)  # transpose matrices
        attn_weights = F.softmax(queries.bmm(keys), dim=2)   # attn_weigths.shape == (batch_size, H*W, H*W)

        values = self.values_conv(x).flatten(start_dim=2)   # values.shape == (batch_size, C', H*W)
        attn = values.bmm(attn_weights)
        attn = attn.view(attn.shape[0], attn.shape[1], H, W)

        attn = self.final_conv(attn)
        attn += x

        return attn, attn_weights

class ScaledDotProductAttention(nn.Module):
    def __init__(self, rnn_hidden_size):
        super(ScaledDotProductAttention, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size

    def forward(self, prev_h, feat_maps_projected):
        # perform **scaled** dot-product attention
        attn_weights = prev_h.unsqueeze(1).bmm(feat_maps_projected)  # attn_weights.shape == (batch_size, 1, 7 * 7)
        attn_weights = attn_weights.div(math.sqrt(self.rnn_hidden_size))

        # get probability-normalized weights
        attn_weights = F.softmax(attn_weights, dim=2)

        # compute context vector
        attn = feat_maps_projected.bmm(attn_weights.permute(0, 2, 1))  # attn.shape == (batch_size, hidden_size, 1)
        attn = attn.squeeze(2)

        return attn, attn_weights.squeeze(1).view(attn_weights.shape[0], 7, 7)   # attn_weights.shape == (batch_size, 7, 7)

class LSTMAttention(nn.Module):
    '''
    Actually, this model can only work in end to end mode(i.e. starting from the frames), since
    we do not have the extracted featrure maps. Hence camera_feature == video_frames_24fps is the only one supported
    '''
    def __init__(self, args, dtype=torch.float32):
        super(LSTMAttention, self).__init__()

        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps

        # The feature_extractor outputs feature maps of shape (batch_size, 512, 7, 7)
        self.feature_extractor = build_feature_extractor(args)
        self.numfeatmaps = self.feature_extractor.feat_vect_dim

        self.numfeatmaps_to_hidden = nn.Linear(self.numfeatmaps, self.hidden_size)
        self.attention = ScaledDotProductAttention(self.hidden_size)

        # The input to the lstm is the concatenation of the input vector and context vector.
        #  The input vector is the usual feature vector, i.e. the globalavgpooling along the feature maps
        #  The context vector is the one returned by the attention layer
        self.lstm = nn.LSTMCell(self.feature_extractor.fusion_size + self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, 3, chunk_size, 112, 112) || (batch_size, enc_steps, 3, 224, 224)
        h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)

        for step in range(self.enc_steps):
            x_t = x[:, step]
            feat_maps = self.feature_extractor(x_t, torch.zeros(1))  # second input is optical flow, in our case will not be used

            # feat_maps.shape == (batch_size, 512, 7, 7)
            feat_maps = feat_maps.flatten(start_dim=2)      # flatten feature maps to feature vectors

            # project the number of channels down to hidden_size, i.e. from 512 to hidden_size
            feat_maps_projected = self.numfeatmaps_to_hidden(feat_maps.permute(0, 2, 1))
            # feat_maps_projected.shape == (batch_size, 7 * 7, hidden_size)
            # Permute back to original axis
            feat_maps_projected = feat_maps_projected.permute(0, 2, 1)

            # Compute attention weights and context vector(attn)
            attn, _ = self.attention(h_n, feat_maps_projected)

            input = feat_maps.mean(dim=2)       # perform global average pooling. input.shape == (batch_size, 512)
            input = torch.cat((input, attn), dim=1)
            h_n, c_n = self.lstm(input, (h_n, c_n))
            out = self.classifier(h_n)  # out.shape == (batch_size, num_classes)

            scores[:, step, :] = out
        return scores

    def step(self, x, h_n, c_n):
        # x.shape == (1, 3, 6, 112, 112) || (1, 3, 224, 224)
        feat_maps = self.feature_extractor(x, torch.zeros(1))
        feat_maps = feat_maps.flatten(start_dim=2)

        feat_maps_projected = self.numfeatmaps_to_hidden(feat_maps.permute(0, 2, 1))
        feat_maps_projected = feat_maps_projected.permute(0, 2, 1)

        attn, attn_weights = self.attention(h_n, feat_maps_projected)

        input = feat_maps.mean(dim=2)
        input = torch.cat((input, attn), dim=1)
        h_n, c_n = self.lstm(input, (h_n, c_n))
        out = self.classifier(h_n)

        return out, h_n, c_n, attn_weights