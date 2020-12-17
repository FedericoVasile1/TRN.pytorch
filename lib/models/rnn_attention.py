import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .feature_extractor import build_feature_extractor
from .rnn import _MyGRUCell

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        # x.shape == batch_size, numfeatmaps, 1, H, W
        return x.squeeze(dim=2)

class Reduction(nn.Module):
    def __init__(self):
        super(Reduction, self).__init__()

    def forward(self, x):
        # x.shape == batch_size, numfeatmaps, T, H, W
        return x.mean(dim=2)

class ScaledDotProductAttention(nn.Module):
    '''
    The scaled dot product attention conceptually takes as input at each timestep the feature maps and the
    previous hidden state and performs a dot product between them in order to obtain the attention weights.
    The attention weights tell us how much each spatial position of the feature maps is important at this timestep.
    The attention weights will be fed as input at this timestep to the lstm together with feature maps.
    '''
    def __init__(self, rnn_hidden_size):
        super(ScaledDotProductAttention, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size

    def forward(self, prev_h, feat_maps_projected):
        """
        - Perform the dot product above discussed.
        - In order to perform dot product(i.e. the batched matrix multiply below) the two inputs must have the
            following shapes: input1: (batch_size, R, C)  -  input2: (batch_size, C, C2) and the output will
            have the following shape:  output: (batch_size, R, C2)
        - (*)Notice that til now, we are only working with few feature extractors, and all of them output
            feature maps of shape (batch_size, 512, 7, 7) hence HH == WW == 7.
        :param prev_h: torch tensor of shape (batch_size, rnn_hidden_size)
        :param feat_maps_projected: torch tensor of shape (batch_size, rnn_hidden_size, HH * WW)
        :return attn: torch tensor of shape (batch_size, rnn_hidden_size)
            This will be the input to the lstm, together with the usual input(i.e. the feature vector in our case)
        :return attn_weights: torch tensor of shape (batch_size, HH, WW)
            We use the weights in evaluation mode; where we project back the (HH, WW) 'image' to its original
            size (H, W) in order to visualize how much each spatial part of the image contributes to the prediction
        """
        #  WARNING: we are assuming feature maps of squared dimension, i.e. HH == WW
        HH, WW = (int(math.sqrt(feat_maps_projected.shape[2])), int(math.sqrt(feat_maps_projected.shape[2])))

        attn_weights = prev_h.unsqueeze(1).bmm(feat_maps_projected)  # attn_weights.shape == (batch_size, 1, HH * WW)
        attn_weights = attn_weights.div(math.sqrt(self.rnn_hidden_size))

        # get probability-normalized weights
        attn_weights = F.softmax(attn_weights, dim=2)

        # compute context vector attn
        attn = feat_maps_projected.bmm(attn_weights.permute(0, 2, 1))  # attn.shape == (batch_size, rnn_hidden_size, 1)
        attn = attn.squeeze(2)

        return attn, attn_weights.squeeze(1).view(attn_weights.shape[0], HH, WW)

class RNNAttention(nn.Module):
    """
    Notice that here we work only with RGB feature maps; neither optical flow fusion nor feature vectors are allowed.
    """
    def __init__(self, args):
        super(RNNAttention, self).__init__()
        if 'featuremaps' not in args.model_input:
            raise Exception('Wrong --model_input option. It must be a folder containing featuremaps')

        self.dtype = torch.float32

        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.steps

        # The feature_extractor outputs feature maps of shape (batch_size, 512, 7, 7)
        self.feature_extractor = build_feature_extractor(args)
        self.numfeatmaps = self.feature_extractor.feat_vect_dim     # e.g. 512
        if self.numfeatmaps not in (832, 1024):
            raise Exception('i3d_mixed4f requires 832.\ni3d_mixed5c requires 1024')

        # WARNING: we are assuming input is 3d, i.e. feature maps of shape (batch_size, CC, TT, HH, WW)
        #  if input will be 2d (i.e. feature maps of shape (batch_size, CC, HH, WW) remove this attribute below)
        # - Option 1: learnable downsampling
        #self.lin_reduction = nn.Sequential(
        #    nn.Conv3d(self.numfeatmaps,
        #              self.numfeatmaps,
        #              (3, 1, 1) if 'mixed4f' in args.model_input else (2, 1, 1)),
        #    Squeeze(),
        #)
        # - Option 2: NOT learnable downsampling
        self.lin_reduction = Reduction()

        self.numfeatmaps_to_hidden = nn.Linear(self.numfeatmaps, self.hidden_size)
        self.attention = ScaledDotProductAttention(self.hidden_size)

        # The input to the lstm is the concatenation of the input vector and context vector.
        #  The input vector is the usual feature vector, i.e. the globalavgpooling along the feature maps
        #  The context vector is the one returned by the attention layer
        if args.model == 'LSTMATTENTION':
            self.rnn = nn.LSTMCell(self.feature_extractor.fusion_size + self.hidden_size, self.hidden_size)
            self.model = 'LSTMATTENTION'
        elif args.model == 'GRUATTENTION':
            self.rnn = _MyGRUCell(self.feature_extractor.fusion_size + self.hidden_size, self.hidden_size)
            self.model = 'GRUATTENTION'
        else:
            raise Exception('Model ' + args.model + ' here is not supported')

        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, C, chunk_size, 112, 112) || (batch_size, enc_steps, 3, 224, 224)
        h_n = torch.zeros(x.shape[0],
                          self.hidden_size,
                          device=x.device,
                          dtype=x.dtype)
        c_n = torch.zeros(x.shape[0],
                          self.hidden_size,
                          device=x.device,
                          dtype=x.dtype) if self.model == 'LSTMATTENTION' else torch.zeros(1).cpu()
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)

        for step in range(self.enc_steps):
            x_t = x[:, step]
            feat_maps = self.feature_extractor(x_t, torch.zeros(1).cpu())  # second input is optical flow, in our case will not be used
            feat_maps = self.lin_reduction(feat_maps)

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
            h_n, c_n = self.rnn(input, (h_n, c_n))
            out = self.classifier(h_n)  # out.shape == (batch_size, num_classes)

            scores[:, step, :] = out
        return scores

    def step(self, x_t, h_n, c_n):
        feat_maps = self.feature_extractor(x_t)
        feat_maps = self.lin_reduction(feat_maps)
        feat_maps = feat_maps.flatten(start_dim=2)

        feat_maps_projected = self.numfeatmaps_to_hidden(feat_maps.permute(0, 2, 1))
        feat_maps_projected = feat_maps_projected.permute(0, 2, 1)

        attn, attn_weights = self.attention(h_n, feat_maps_projected)

        input = feat_maps.mean(dim=2)
        input = torch.cat((input, attn), dim=1)
        h_n, c_n = self.rnn(input, (h_n, c_n))
        out = self.classifier(h_n)

        return out, h_n, c_n, attn_weights