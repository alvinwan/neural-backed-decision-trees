import torch


class InverseNormalize:

    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)[None, :, None, None]
        self.std = torch.Tensor(std)[None, :, None, None]

    def __call__(self, sample):
        return (sample * self.std) + self.mean
