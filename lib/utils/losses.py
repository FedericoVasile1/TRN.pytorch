import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MultiCrossEntropyLoss', 'ContrastiveLoss']

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-500):
        super(MultiCrossEntropyLoss, self).__init__()

        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), 1)
            return torch.mean(output[target[:, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * logsoftmax(input), 1)
            if self.size_average:
                return torch.mean(output)
            else:
                return torch.sum(output)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, x1, x2, y):
        """
        :param x1: torch tensor of shape (batch_size, num_features)
        :param x2: torch tensor of shape (batch_size, num_features)
        :param y: torch tensor of shape (batch_size,) containing a flag {+1, -1} indicating if the pair of samples
                    at position "i" in the batch are condiderated similar or dissimilar
        :return: torch scalar(or tensor, depending on self.reduction) representing the loss over the batch
        """
        dist = F.mse_loss(x1, x2, reduction='none').mean(dim=1)      # dist.shape == (batch_size,)

        mask = y == 1
        out1 = (dist) * mask     # out1.shape == (batch_size,)

        mask = y != 1
        out2 = (torch.max(torch.zeros_like(dist), self.margin - dist)) * mask       # out2.shape == (batch_size)

        out = out1 + out2

        if self.reduction == 'none':
            return out
        elif self.reduction == 'sum':
            return out.sum()
        elif self.reduction == 'mean':
            return out.mean()
        else:
            raise Exception('Reduction ' + self.reduction + ' is not supported')