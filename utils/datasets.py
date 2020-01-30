import torchvision.datasets as datasets
import torchvision.transforms as transforms


class TinyImagenetDataset(datasets.ImageFolder):
    """Tiny imagenet dataloader"""

    def __init__(self, path='data/tiny-imagenet-200/train', *args,
            transform=transforms.ToTensor(), **kwargs):
        super(path, *args, transform=transform, **kwargs)

    @staticmethod
    def transforms_train():
        return transforms.ToTensor()

    @staticmethod
    def transforms_val():
        return transforms.ToTensor()


class CIFAR10NodeDataset(datasets.CIFAR10):
    """Creates dataset for a specific node in the CIFAR10 wordnet tree"""

    def __init__(self, wnid, path_tree):
        pass

    def __getitem__(self, i):
        sample, label = super()[i]
        return sample, label
