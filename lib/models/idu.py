import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import math

from .feature_extractor import build_feature_extractor

class IDUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, steps, device, dtype=torch.float32):
        super(IDUCell, self).__init__()
        self.num_classes = num_classes
        self.steps = steps

        self.Wxe = Parameter(torch.randn(input_size, hidden_size, dtype=dtype, device=device).div(math.sqrt(input_size)))
        self.bxe = Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.Wep = Parameter(torch.randn(hidden_size, num_classes, dtype=dtype, device=device).div(math.sqrt(hidden_size)))
        self.bep = Parameter(torch.zeros(num_classes, dtype=dtype, device=device))

        self.Whr = Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype, device=device).div(math.sqrt(hidden_size)))
        self.bhr = Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.Wx0r = Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype, device=device).div(math.sqrt(hidden_size)))
        self.bx0r = Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))

        self.Wxtz = Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype, device=device).div(math.sqrt(hidden_size)))
        self.bxtz = Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.Wx0z = Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype, device=device).div(math.sqrt(hidden_size)))
        self.bx0z = Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))

        self.Wxth1 = Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype, device=device).div(math.sqrt(hidden_size)))
        self.bxth1 = Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.Wh1h1 = Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype, device=device).div(math.sqrt(hidden_size)))
        self.bh1h1 = Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))

    def forward(self, xt, x0, prev_h):
        """
        Do a forward pass for a single timestep
        :param xt: Feature vector of the current timestep, of shape (batch_size, feat_vect_dim)
        :param x0: Current information, of shape (batch_size, feat_vect_dim)
        :param prev_h: The previous hidden state, of shape (batch_size, hidden_size)
        :return ht: the next hidden state, of shape (batch_size, hidden_size)
                pte, p0e: probability distribution over action classes, for current timestep and current
                            information, both of shape (batch_size, num_classes)
                xte, xt0: the outputs of the early embedding module, for current timestep and current
                            information, both of shape (batch_size, hidden_size)
        """
        xte = F.relu(xt.mm(self.Wxe) + self.bxe)
        x0e = F.relu(x0.mm(self.Wxe) + self.bxe)

        pte = xte.mm(self.Wep) + self.bep
        p0e = x0e.mm(self.Wep) + self.bep

        rt = torch.sigmoid(prev_h.mm(self.Whr) + self.bhr + x0e.mm(self.Wx0r) + self.bx0r)
        zt = torch.sigmoid(xte.mm(self.Wxtz) + self.bxtz + x0e.mm(self.Wx0z) + self.bx0z)

        h1_t_1 = prev_h.mm(self.Wh1h1) + self.bh1h1
        ht1 = torch.tanh(xte.mm(self.Wxth1) + self.bxth1 + rt * h1_t_1)
        ht = (1 - zt) * ht1 + zt * prev_h

        return ht, pte, p0e, xte, x0e

class IDU(nn.Module):
    def __init__(self, args):
        super(IDU, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.steps = args.enc_steps

        self.feature_extractor = build_feature_extractor(args)
        self.drop = nn.Identity()
        self.iducell = IDUCell(self.feature_extractor.fusion_size,
                               args.hidden_size,
                               args.num_classes,
                               args.enc_steps,
                               args.device)
        self.classifier = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, x):
        """
        It executes a complete unroll of the network for enc_steps. Furthermore, we need also the current
        information: for a given sequence of data(it is enc_steps long) the current information is the feature
        vector of the last timestep of the sequence
        :param x: Input tensor containing feature vectors for each
                    timestep, of shape (batch_size, enc_steps, feat_vect_dim)
        :return scores: tensor containing score classes for each
                        timestep, of shape (batch_size, enc_steps, num_classes)
                ptes, p0es: probability distributions, of shape (batch_size, enc_steps, num_classes)
                xtes, x0es: early embedding features, of shape (batch_size, enc_steps, hidden_size)
        """
        scores = []
        ptes = []
        p0es = []
        xtes = []
        x0es = []

        x0 = x[:, -1]       # last step features is the current information
        ht = torch.zeros(x.shape[0], self.hidden_size, dtype=x.dtype, device=x.device)
        for step in range(self.steps):
            xt = x[:, step]
            xt = self.feature_extractor(xt, torch.zeros(1))
            ht, pte, p0e, xte, x0e = self.iducell(xt, x0, ht)
            out = self.classifier(ht)

            scores.append(out)
            ptes.append(pte)
            p0es.append(p0e)
            xtes.append(xte)
            x0es.append(x0e)
        scores = torch.stack(scores, dim=1)
        ptes = torch.stack(ptes, dim=1)
        p0es = torch.stack(p0es, dim=1)
        xtes = torch.stack(xtes, dim=1)
        x0es = torch.stack(x0es, dim=1)

        return scores, ptes, p0es, xtes, x0es

    def step(self, x_t, x_0, prev_h):
        raise NotImplementedError