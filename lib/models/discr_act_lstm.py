import torch
import torch.nn as nn

from .feature_extractor import build_feature_extractor

class DiscrActLSTM(nn.Module):
    def __init__(self, args):
        super(DiscrActLSTM, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes     # leave background
        self.enc_steps = args.enc_steps
        self.act_h_n = None
        self.act_c_n = None

        self.is_first = True
        self.backgr_vect = torch.zeros(1, self.num_classes)   # (batch_size, num_classes)
        self.backgr_vect[0, 0] = 1

        self.feature_extractor = build_feature_extractor(args)

        self.discr = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.discr_classifier = nn.Linear(self.hidden_size, 2)  # 2 because background and action
        self.act = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.act_classifier = nn.Linear(self.hidden_size, self.num_classes)

    def step(self, camera_input, discr_h_n, discr_c_n):
        feat_vect = self.feature_extractor(camera_input, torch.zeros(1))

        discr_h_n, discr_c_n = self.discr(feat_vect, (discr_h_n, discr_c_n))
        out = self.discr_classifier(discr_h_n)   # out.shape == (batch_size, num_classes) == (1, 2)

        assert out.shape == torch.Size([1, 2])
        if out[0, 1].item() > 0.5:
            if self.is_first:
                self.act_h_n = torch.zeros(1, self.hidden_size, device=camera_input.device, dtype=camera_input.dtype)
                self.act_c_n = torch.zeros(1, self.hidden_size, device=camera_input.device, dtype=camera_input.dtype)
                self.is_first = False
            self.act_h_n, self.act_c_n = self.act(feat_vect, (self.act_h_n, self.act_c_n))
            out = self.act_classifier(self.act_h_n)
        else:
            self.is_first = True
            out = self.backgr_vect

        return out, discr_h_n, discr_c_n