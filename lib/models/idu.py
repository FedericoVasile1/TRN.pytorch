import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from models.gru import GRU

import math

class IDUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(IDUCell, self).__init__()
        self.num_classes = num_classes

        self.Wxe = Parameter(torch.randn(input_size, hidden_size)).div(math.sqrt(input_size))
        self.bxe = Parameter(torch.zeros(hidden_size))
        self.Wep = Parameter(torch.randn(hidden_size, num_classes)).div(math.sqrt(hidden_size))
        self.bep = Parameter(torch.zeros(num_classes))

        self.Whr = Parameter(torch.randn(hidden_size, hidden_size)).div(math.sqrt(hidden_size))
        self.bhr = Parameter(torch.zeros(hidden_size))
        self.Wx0r = Parameter(torch.randn(hidden_size, hidden_size)).div(math.sqrt(hidden_size))
        self.bx0r = Parameter(torch.zeros(hidden_size))

        self.Wxtz = Parameter(torch.randn(hidden_size, hidden_size)).div(math.sqrt(hidden_size))
        self.bxtz = Parameter(torch.zeros(hidden_size))
        self.Wx0z = Parameter(torch.randn(hidden_size, hidden_size)).div(math.sqrt(hidden_size))
        self.bx0z = Parameter(torch.zeros(hidden_size))

        self.Wxth1 = Parameter(torch.randn(hidden_size, hidden_size)).div(math.sqrt(hidden_size))
        self.bxth1 = Parameter(torch.zeros(hidden_size))
        self.Wh1h1 = Parameter(torch.randn(hidden_size, hidden_size)).div(math.sqrt(hidden_size))
        self.bh1h1 = Parameter(torch.zeros(hidden_size))

    def step(self, xt, x0, prev_h):
        '''
        Do a forward for a single timestep
        :param xt: Feature vector of the current timestep, of shape (batch_size, feat_vect_dim)
        :param x0: Current information, of shape (batch_size, feat_vect_dim)
        :param prev_h: The previous hidden state, of shape  (batch_size, hidden_size)
        :return ht: the next hidden  state
                pte, p0e: probability distribution over action classes, for current timestep and current
                            information, both of shape (batch_size, num_classes)
                xte, xt0: the outputs of the early embedding module, for current timestep and current
                            information, both of shape (batch_size, hidden_size)
        '''

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

    def forward(self, x, h0):
        '''
        It executes a complete unroll of the network for enc_steps. Furthermore, we need also the current
        information: for a given sequence of data(it is enc_steps long) the current information is the feature
        vector of the last timestep of the sequence
        :param x: Input tensor containing feature vectors for each
                    timestep, of shape (batch_size, enc_steps, feat_vect_dim)
        :param h0: The initial hidden state, of shape (batch_size, hidden_size)
        :return hts: hidden states for each timestep, of shape (batch_size, enc_steps, hidden_size)
                ptes, p0es: prpbability distributions, of shape (batch_size, enc_steps, num_classes)
                xtes, x0es: early embedding features, of shape (batch_size, enc_steps, hidden_size)
        '''
        batch_size, steps, _ = x.shape
        hidden_size = h0.shape[1]
        hts = torch.zeros(batch_size, steps, hidden_size).to(dtype=x.dtype)
        ptes = torch.zeros(batch_size, steps, self.num_classes).to(dtype=x.dtype)
        p0es = torch.zeros(batch_size, steps, self.num_classes).to(dtype=x.dtype)
        xtes = torch.zeros(batch_size, steps, hidden_size).to(dtype=x.dtype)
        x0es = torch.zeros(batch_size, steps, hidden_size).to(dtype=x.dtype)

        x0 = x[:, -1]        # last step features is the current information
        h_t = h0
        for step_t in range(steps):
            x_t = x[:, step_t]
            h_t, pte, p0e, xte, x0e = self.step(x_t, x0, h_t)

            hts[:, step_t] = h_t
            ptes[:, step_t] = pte
            p0es[:, step_t] = p0e
            xtes[:, step_t] = xte
            x0es[:, step_t] = x0e

        return hts, ptes, p0es, xtes, x0es

class IDU(nn.Module):
    def __init__(self, args):
        super(IDU, self).__init__()
        self.hidden_size = args.hidden_size

        self.iducell = IDUCell(args.feat_vect_dim, args.hidden_size, args.num_classes)
        self.classifier = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, x):
        # x.shape == (batch_size, steps, feat_vect_dim)
        h0 = torch.zeros(x.shape[0], self.hidden_size, dtype=x.dtype, device=x.device)
        hts, ptes, p0es, xtes, x0es = self.iducell(x, h0)
        scores = self.classifier(hts[:, -1])
        return scores, ptes, p0es, xtes, x0es