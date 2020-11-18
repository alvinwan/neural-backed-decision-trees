"""Wrappers around CIFAR datasets"""

from torchvision import datasets, transforms
from . import transforms as transforms_custom

__all__ = names = ("CIFAR10", "CIFAR100")


class CIFAR:
    @staticmethod
    def transform_train():
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    @staticmethod
    def transform_val():
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    @staticmethod
    def transform_val_inverse():
        return transforms_custom.InverseNormalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )


class CIFAR10(datasets.CIFAR10, CIFAR):
    pass


class CIFAR100(datasets.CIFAR100, CIFAR):
    pass
