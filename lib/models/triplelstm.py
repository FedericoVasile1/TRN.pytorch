import torch
import torch.nn as nn

import _init_paths
import utils as utl
from .feature_extractor import build_feature_extractor

class TripleLSTM(nn.Module):
    def __init__(self, args):
        super(TripleLSTM, self).__init__()
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.num_classes_actback = 2                        # action and background
        self.num_classes_acts = args.num_classes            # all actions
        self.num_classes_startend = args.num_classes * 2    # all actions divided in start and end

        self.feature_extractor = build_feature_extractor(args)

        self.actback = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.actback_classifier = nn.Linear(self.hidden_size, 2)

        self.acts = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.acts_classifier = nn.Linear(self.hidden_size, self.num_classes_acts)

        self.startend = nn.LSTMCell(self.feature_extractor.fusion_size, self.hidden_size)
        self.startend_classifier = nn.Linear(self.hidden_size, self.num_classes_startend)

    def forward(self, x):
        # x.shape == (batch_size, enc_steps, feat_vect_dim)
        actback_h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        actback_c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        acts_h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        acts_c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        startend_h_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        startend_c_n = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)

        actback_scores = torch.zeros(x.shape[0], self.enc_steps, self.num_classes_actback, dtype=x.dtype)
        acts_scores = torch.zeros(x.shape[0], self.enc_steps, self.num_classes_acts, dtype=x.dtype)
        startend_scores = torch.zeros(x.shape[0], self.enc_steps, self.num_classes_startend, dtype=x.dtype)
        final_scores = torch.zeros(x.shape[0], self.enc_steps, self.num_classes_acts, dtype=x.dtype)

        for step in range(self.enc_steps):
            x_t = x[:, step]
            out = self.feature_extractor(x_t, torch.zeros(1))  # second input is optical flow, in our case will not be used

            actback_h_n, actback_c_n = self.actback(out, (actback_h_n, actback_c_n))
            acts_h_n, acts_c_n = self.acts(out, (acts_h_n, acts_c_n))
            startend_h_n, startend_c_n = self.startend(out, (startend_h_n, startend_c_n))

            actback_score = self.actback_classifier(actback_h_n)
            acts_score = self.acts_classifier(acts_h_n)
            startend_score = self.startend_classifier(startend_h_n)

            actback_scores[:, step] = actback_score
            acts_scores[:, step] = acts_score
            startend_scores[:, step] = startend_score

        # do post-processing for the final prediction
        with torch.set_grad_enabled(False):
            for step in range(self.enc_steps):
                for sample in range(x.shape[0]):
                    if actback_scores[sample, step].argmax().detach().item() == 0:
                        backgr_vect = torch.zeros(self.num_classes_acts)
                        backgr_vect[0] = 1
                        final_scores[sample, step] = backgr_vect
                    else:
                        cls_acts = acts_scores[sample, step].argmax()
                        cls_startend = startend_scores[sample, step].argmax()
                        if cls_startend.detach().item() > 21:
                            cls_startend -= 22

                        if cls_acts.detach().item() == cls_startend.detach().item():
                            # they predict the same action for the current step, so we are very surely about
                            #  action predicted
                            action_vect = torch.zeros(self.num_classes_acts)
                            action_vect[cls_startend] = 1
                            final_scores[sample, step] = action_vect
                        else:
                            if torch.tensor([cls_startend]).to(x.device) in startend_scores[sample].argmax(dim=1)   \
                            and torch.tensor([cls_startend+22]).to(x.device) in startend_scores[sample].argmax(dim=1):
                                # during the sequence the model individuate both start and end of an action, so
                                #  for this timestep the action is surely this one
                                action_vect = torch.zeros(self.num_classes_acts)
                                action_vect[cls_startend] = 1
                                final_scores[sample, step] = action_vect
                            else:
                                # we have not enough information to be sure that the initially predicted action is
                                #  the real one, so predict background
                                backgr_vect = torch.zeros(self.num_classes_acts)
                                backgr_vect[0] = 1
                                final_scores[sample, step] = backgr_vect

        return final_scores, actback_scores, acts_scores, startend_scores

    def step(self, camera_input, h_n, c_n):
        raise NotImplementedError