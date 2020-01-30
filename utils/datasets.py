import torchvision.datasets as datasets
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
from utils.xmlutils import get_leaves, remove


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
    """Creates dataset for a specific node in the CIFAR10 wordnet tree

    wnids.txt is needed to map wnids to class indices
    """

    names = (
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    )

    def __init__(self, wnid, root='./data', *args,
            path_tree='./data/cifar10/tree.xml',
            path_wnids='./data/cifar10/wnids.txt', **kwargs):
        super().__init__(root=root, *args, **kwargs)
        self.wnid = wnid

        with open(path_wnids) as f:
            wnids = [line.strip() for line in f.readlines()]

        tree = ET.parse(path_tree)
        # handle multiple paths issue with hack -- remove other path
        remove(tree, tree.find('.//synset[@wnid="n03791235"]'))

        # generate mapping from wnid to class
        self.mapping = {}
        node = tree.find('.//synset[@wnid="{}"]'.format(wnid))
        children = node.getchildren()
        n = len(children)
        assert n > 0, 'Cannot build dataset for leaf node.'

        for new_index, child in enumerate(children):
            for leaf in get_leaves(child):
                wnid = leaf.get('wnid')
                old_index = wnids.index(wnid)
                self.mapping[old_index] = new_index

        for old_index in range(10):
            if old_index not in self.mapping:
                self.mapping[old_index] = n

    def __getitem__(self, i):
        sample, old_label = super().__getitem__(i)
        return sample, self.mapping[old_label]
