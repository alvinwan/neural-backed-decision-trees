import torch


class TopK:

    def __init__(self, k=1):
        self.k = k

    def forward(self, outputs, targets):
        _, pred = torch.topk(outputs, self.k)
        labels = targets.repeat((self.k, 1)).T
        return (pred == labels).sum().item()
