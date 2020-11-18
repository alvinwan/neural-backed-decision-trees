import torch


__all__ = names = ("top1", "top2", "top5", "top10")


class TopK:
    def __init__(self, k=1):
        self.k = k
        self.clear()

    def clear(self):
        self.correct = 0
        self.total = 0

    def forward(self, outputs, targets):
        _, preds = torch.topk(outputs, self.k)
        results = [(pred == target).any() for pred, target in zip(preds, targets)]
        self.correct += sum(results).item()
        self.total += targets.size(0)

    def report(self):
        return self.correct / (self.total or 1)

    def __repr__(self):
        return f"Top{self.k}: {self.report()}"

    def __str__(self):
        return repr(self)


top1 = lambda: TopK(1)
top2 = lambda: TopK(2)
top5 = lambda: TopK(5)
top10 = lambda: TopK(10)
