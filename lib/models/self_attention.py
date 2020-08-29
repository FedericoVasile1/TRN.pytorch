import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import build_feature_extractor

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        # The shape of x must be (batch_size, enc_steps, num_channels, height, width)
        return x.mean(3, 4)

class SelfAttentionLayer(nn.Module):
    """
    Pay attention to not make confusion between lstm with attention and self-attention; they are two totally different
    models, both able to process sequences!!
    """
    def __init__(self, in_channels, out_channels):
        super(SelfAttentionLayer, self).__init__()

        self.queries_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.keys_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.values_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.final_conv = nn.Conv2d(out_channels, in_channels, 1)

    def forward(self, x):
        '''
        Check https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture13.pdf at slide 81 for
        details about computations
        :param x: torch tensor of shape (batch_size, CC, HH, WW)
        :return attn: torch tensor of shape (batch_size, CC, HH, WW)
                attn_weights: torch tensor of shape(batch_size, HH*WW, HH*WW)
        '''
        _, _, HH, WW = x.shape

        queries = self.queries_conv(x).flatten(start_dim=2)
        keys = self.keys_conv(x).flatten(start_dim=2)

        queries = queries.permute(0, 2, 1)  # transpose matrices
        attn_weights = F.softmax(queries.bmm(keys), dim=2)   # attn_weigths.shape == (batch_size, HH*WW, HH*WW)

        values = self.values_conv(x).flatten(start_dim=2)   # values.shape == (batch_size, C', HH*WW)
        attn = values.bmm(attn_weights)
        attn = attn.view(attn.shape[0], attn.shape[1], HH, WW)

        attn = self.final_conv(attn)
        attn += x

        return attn, attn_weights

class SelfAttention(nn.Module):
    '''
    This is the full SelfAttention model; it is made up of: feature_extractor > self-attention layers > classifier
    '''
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.camera_feature != 'resnet3d_featuremaps' and args.camera_feature != 'video_frames_24fps':
            raise Exception('Wrong camera_feature option, this model supports only feature maps. '
                            'Change this option to \'resnet3d_featuremaps\' or switch to end to end training with the '
                            'following options: --camera_feature video_frames_24fps --feature_extractor RESNET2+1D')

        filters = [64, 32, args.num_classes]
        num_layers = 3
        assert num_layers ==  len(filters)
        assert args.num_classes == filters[-1]
        self.enc_steps = 8      # it corrensponds to 2 seconds
        self.num_classes = args.num_classes

        self.feature_extractor = build_feature_extractor(args)
        self.input_channels_dim = self.feature_extractor.fusion_size

        # TODO: think whether to project down or not the channels dim
        self.lin_proj = nn.Identity()

        self.selfattention_layers = []
        for i in range(num_layers):
            input_dim = self.input_channels_dim if i == 0 else self.filters[i-1]
            layer = SelfAttentionLayer(input_dim, filters[i])
            self.selfattention_layers.append(layer)

        self.globalavgpool = GlobalAveragePooling()

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, 512, 7, 7) || (batch_size, enc_steps, 3, chunk_size, 112, 112)
        scores = torch.zeros(x.shape[0], x.shape[1], self.num_classes, dtype=x.dtype)

        feat_maps = []
        for step in range(self.enc_steps):
            out = self.feature_extractor(x, torch.zeros(1).cpu())
            feat_maps.append(out)
        # TODO: CHECK IF THIS OPERATION BELOW(ESPECIALLY THE PERMUTE) IS CORRECT
        out = torch.stack(feat_maps).permute(1, 0, 2, 3, 4)     # swap batch_size and enc_steps
        # out.shape == (batch_size, enc_steps, 512, 7, 7)

        for l in range(len(self.selfattention_layers)):
            out = self.selfattention_layers[l](out)
        # out.shape == (batch_size, enc_steps, num_classes, 7, 7)

        scores = self.globalavgpool(out)

        return scores

    def step(self, x):
        pass