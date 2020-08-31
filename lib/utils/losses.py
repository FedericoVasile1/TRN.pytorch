import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MultiCrossEntropyLoss', 'ContrastiveLoss', 'MarginMSELoss']

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

class MarginMSELoss(nn.Module):
    def __init__(self, margin, reduction='mean'):
        super(MarginMSELoss, self).__init__()
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

class ContrastiveLoss(nn.Module):
    def __init__(self, ignore_index=21):
        super(ContrastiveLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, xtes, x0es, yts, m=1):
        batch_size, steps, num_features = xtes.shape            # in our case num_features == idu.hidden_size
        _, _, num_classes = yts.shape

        loss = 0.0
        for step in range(steps):
            for sample in range(batch_size):
                sample_xte, sample_x0e = (xtes[sample, step], x0es[sample, step])
                target_xte, target_x0e = (yts[sample, step], yts[sample, -1])

                if target_xte.argmax() == self.ignore_index or target_x0e.argmax() == self.ignore_index:
                    continue

                d = torch.dist(sample_xte, sample_x0e).pow(2)       # calculate the squared euclidean distance
                if target_xte.argmax() == target_x0e.argmax():
                    loss += d
                else:
                    temp = m - d
                    if temp > 0:
                        loss += temp
        loss /= (steps * batch_size)
        return loss

if __name__ == '__main__':
    a = torch.randn(3, 2)
    b = torch.randn(3, 2)
    y = torch.randint(0, 2, (3,))
    loss = MarginMSELoss(1, reduction='none')
    print(loss(a, b, y))