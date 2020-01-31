import torch
import torch.nn as nn


__all__ = ('linear',)


class Linear(nn.Linear):

    def set_weight(self, weight):
        with torch.no_grad():
            self.weight = torch.nn.Parameter(weight)


def linear(input_dim=10, num_classes=10):
    return Linear(input_dim, num_classes, bias=False)
