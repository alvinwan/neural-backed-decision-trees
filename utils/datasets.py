import torchvision.datasets as datasets
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
from utils.xmlutils import get_leaves, remove
import torch
import numpy as np


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


class CIFAR10Node:

    original_classes = (
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

    def __init__(self, wnid,
            path_tree='./data/cifar10/tree.xml',
            path_wnids='./data/cifar10/wnids.txt'):
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
        self.num_children = n

        for new_index, child in enumerate(children):
            for leaf in get_leaves(child):
                wnid = leaf.get('wnid')
                old_index = wnids.index(wnid)
                self.mapping[old_index] = new_index

        for old_index in range(10):
            if old_index not in self.mapping:
                self.mapping[old_index] = n

        classes = [[] for _ in range(n + 1)]
        for old_index in range(10):
            original_class = self.original_classes[old_index]
            classes[new_index].append(original_class)
        self.classes = [','.join(names) for names in classes]

    @staticmethod
    def get_wnid_to_node(path_tree, path_wnids):
        tree = ET.parse(path_tree)
        wnid_to_node = {}
        for node in tree.iter():
            wnid = node.get('wnid')
            if wnid is None or len(node.getchildren()) == 0:
                continue
            wnid_to_node[wnid] = CIFAR10Node(node.get('wnid'),
                path_tree=path_tree, path_wnids=path_wnids)
        return wnid_to_node



class CIFAR10NodeDataset(datasets.CIFAR10):
    """Creates dataset for a specific node in the CIFAR10 wordnet tree

    wnids.txt is needed to map wnids to class indices
    """

    def __init__(self, wnid, root='./data', *args,
            path_tree='./data/cifar10/tree.xml',
            path_wnids='./data/cifar10/wnids.txt', **kwargs):
        super().__init__(root=root, *args, **kwargs)
        self.node = CIFAR10Node(wnid, path_tree, path_wnids)
        self.classes = self.node.classes
        self.original_classes = self.node.original_classes

    def __getitem__(self, i):
        sample, old_label = super().__getitem__(i)
        return sample, self.node.mapping[old_label]


class CIFAR10PathSanityDataset(datasets.CIFAR10):
    """returns samples that assume all node classifiers are perfect"""

    def __init__(self, root='./data', *args,
            path_tree='./data/cifar10/tree.xml',
            path_wnids='./data/cifar10/wnids.txt', **kwargs):
        super().__init__(root=root, *args, **kwargs)
        wnid_to_node = CIFAR10Node.get_wnid_to_node(path_tree, path_wnids)
        wnids = sorted(wnid_to_node)
        self.nodes = [wnid_to_node[wnid] for wnid in wnids]

    def get_sample(self, node, old_label):
        new_label = node.mapping[old_label]
        sample = [0] * len(node.classes)
        sample[new_label] = 1
        return sample

    def _get_node_weights(self, node):
        n = len(node.classes)
        k = 10

        A = np.zeros((n, k))
        for new_index, cls in enumerate(node.classes):
            if ',' not in cls and cls:  # if class is leaf
                old_index = node.original_classes.index(cls)
                A[new_index, old_index] = 1
        return A

    def get_weights(self):
        """get perfect fully-connected layer weights"""
        weights = []
        for node in self.nodes:
            weights.append(self._get_node_weights(node))
        weights = np.concatenate(weights, axis=0).T
        return torch.Tensor(weights)

    def get_input_dim(self):
        return sum([len(dataset.classes) for dataset in self.nodes])

    def __getitem__(self, i):
        _, old_label = super().__getitem__(i)

        sample = []
        for dataset in self.nodes:
            sample.extend(self.get_sample(dataset, old_label))
        sample = torch.Tensor(sample)

        return sample, old_label


class CIFAR10PathDataset(datasets.CIFAR10):
    """returns samples from all node classifiers"""

    def __init__(self, root='./data', *args,
            path_tree='./data/cifar10/tree.xml',
            path_wnids='./data/cifar10/wnids.txt',
            pretrained=True,
            **kwargs):
        super().__init__(root=root, *args, **kwargs)

        wnid_to_node = CIFAR10Node.get_wnid_to_node(path_tree, path_wnids)
        wnids = sorted(wnid_to_node)
        self.nodes = [wnid_to_node[wnid] for wnid in wnids]
        self.nets = [self.get_net_for_node(node, pretrained) for node in self.nodes]

    def get_net_for_node(self, node, pretrained):
        import models
        # TODO: WARNING: the model and paths are hardcoded
        net = models.ResNet10(num_classes=len(node.classes))
        if pretrained:
            checkpoint = torch.load(f'./checkpoint/ckpt-CIFAR10node-ResNet10-{node.wnid}.pth')
            # remove module. prefix from all keys 0.o hack
            state_dict = {key.replace('module.', '', 1): value for key, value in checkpoint['net'].items()}
            net.load_state_dict(state_dict)
        return net

    # WARNING: copy-pasta from above
    def get_input_dim(self):
        return sum([len(dataset.classes) for dataset in self.nodes])

    def __getitem__(self, i):
        old_sample, old_label = super().__getitem__(i)

        sample = []
        for net in self.nets:
            sample.extend(net(old_sample[None]))
        sample = torch.cat(sample, 0)

        return sample, old_label
