import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset



__all__ = names = ('TinyImagenet200',)


class TinyImagenet200(Dataset):
    """Tiny imagenet dataloader"""

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    dataset = None

    # TODO: download is just ignoredd
    def __init__(self, root='./data',
            *args, train=True, download=False, **kwargs):
        super().__init__()

        if download:
            print('==> Download not setup. Ignoring...')
        dataset = _TinyImagenet200Train if train else _TinyImagenet200Val
        self.dataset = dataset(root, *args, **kwargs)
        self.classes = self.dataset.classes

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class _TinyImagenet200Train(datasets.ImageFolder):

    def __init__(self, root='./data', *args, **kwargs):
        super().__init__(os.path.join(root, 'tiny-imagenet-200/train'), *args, **kwargs)


class _TinyImagenet200Val(datasets.ImageFolder):

    def __init__(self, root='./data', *args, **kwargs):
        super().__init__(os.path.join(root, 'tiny-imagenet-200/val'), *args, **kwargs)

        self.path_to_class = {}
        with open(os.path.join(self.root, 'val_annotations.txt')) as f:
            for line in f.readlines():
                parts = line.split()
                path = os.path.join(self.root, 'images', parts[0])
                self.path_to_class[path] = parts[1]

        self.classes = list(sorted(set(self.path_to_class.values())))
        self.class_to_idx = {
            label: self.classes.index(label) for label in self.classes
        }

    def __getitem__(self, i):
        sample, _ = super().__getitem__(i)
        path, _ = self.samples[i]
        label = self.path_to_class[path]
        target = self.class_to_idx[label]
        return sample, target

    def __len__(self):
        return super().__len__()
