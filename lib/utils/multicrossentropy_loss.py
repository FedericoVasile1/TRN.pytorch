import torch
import torch.nn as nn

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