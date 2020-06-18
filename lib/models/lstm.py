import torch
import torch.nn as nn

class LSTMmodel(nn.Module):
    def __init__(self, args):
        super(LSTMmodel, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.future_size = args.neurons

        FEAT_VECT_DIM = args.feat_vect_dim
        self.lin_transf = nn.Sequential(
            nn.Linear(FEAT_VECT_DIM, args.neurons),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.LSTMCell(args.neurons, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, junk):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        h_n = torch.zeros(x.shape[0], self.hidden_size, dtype=x.dtype, device=x.device)
        c_n = torch.zeros(x.shape[0], self.hidden_size, dtype=x.dtype, device=x.device)
        score_stack = []
        for step in range(x.shape[1]):
            x_t = x[:, step]
            out = self.lin_transf(x_t)
            h_n, c_n = self.lstm(out, (h_n, c_n))
            out = self.classifier(h_n)  # self.classifier(h_n).shape == (batch_size, num_classes)
            score_stack.append(out)
        junk = torch.zeros(x.shape[0], self.enc_steps*self.dec_steps, self.num_classes).view(-1, self.num_classes)
        scores = torch.stack(score_stack, dim=1).view(-1, self.num_classes)
        return scores, junk

    def step(self, camera_input, junk, junk2, h_n, c_n):
        out = self.lin_transf(camera_input)
        h_n, c_n = self.lstm(out, (h_n, c_n))
        out = self.classifier(h_n)

        junk = camera_input.new_zeros((camera_input.shape[0], self.future_size))
        junk2 = [torch.zeros(1, 22) for dec_step in range(self.dec_steps)]
        return junk, h_n, c_n, out, junk2