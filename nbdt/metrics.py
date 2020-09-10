import torch


__all__ = names = ('Top1', 'Top2', 'Top5')


class TopK:

    def __init__(self, k=1):
        self.k = k
        self.clear()

    def clear(self):
        self.correct = 0
        self.total = 0

    def forward(self, outputs, targets):
        _, pred = torch.topk(outputs, self.k)
        labels = targets.repeat((self.k, 1)).T
        self.correct += (pred == labels).sum().item()
        self.total += targets.size(0)

    def report(self):
        return self.correct / self.total

    def __repr__(self):
        return f'Top{self.k}: {self.report()}'

    def __str__(self):
        return repr(self)


Top1 = lambda: TopK(1)
Top2 = lambda: TopK(2)
Top5 = lambda: TopK(5)
