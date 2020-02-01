import torchvision.datasets as datasets
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
from utils.xmlutils import get_leaves, remove
import torch
import numpy as np


__all__ = names = ('CIFAR10Node', 'CIFAR10JointNodes', 'CIFAR10PathSanity')


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


class Node:

    def __init__(self, wnid,
            path_tree='./data/cifar10/tree.xml',
            path_wnids='./data/cifar10/wnids.txt',
            classes=()):
        self.wnid = wnid
        self.original_classes = classes

        with open(path_wnids) as f:
            wnids = [line.strip() for line in f.readlines()]
        self.num_original_classes = len(wnids)

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
        self.num_classes = self.num_children + 1

        for new_index, child in enumerate(children):
            for leaf in get_leaves(child):
                wnid = leaf.get('wnid')
                old_index = wnids.index(wnid)
                self.mapping[old_index] = new_index

        for old_index in range(self.num_original_classes):
            if old_index not in self.mapping:
                self.mapping[old_index] = n

        self.classes = []
        if self.original_classes:
            classes = [[] for _ in range(n + 1)]
            for old_index in range(self.num_original_classes):
                original_class = self.original_classes[old_index]
                new_index = self.mapping[old_index]
                classes[new_index].append(original_class)

            self.classes = [','.join(names) for names in classes if names]

    @staticmethod
    def get_wnid_to_node(path_tree, path_wnids, classes):
        tree = ET.parse(path_tree)
        wnid_to_node = {}
        for node in tree.iter():
            wnid = node.get('wnid')
            if wnid is None or len(node.getchildren()) == 0:
                continue
            wnid_to_node[wnid] = Node(node.get('wnid'),
                path_tree=path_tree, path_wnids=path_wnids, classes=classes)
        return wnid_to_node

    @staticmethod
    def get_nodes(path_tree, path_wnids, classes):
        wnid_to_node = Node.get_wnid_to_node(path_tree, path_wnids, classes)
        wnids = sorted(wnid_to_node)
        nodes = [wnid_to_node[wnid] for wnid in wnids]
        return nodes

    @staticmethod
    def dim(nodes):
        return sum([node.num_classes for node in nodes])


class CIFAR10Node(datasets.CIFAR10):
    """Creates dataset for a specific node in the CIFAR10 wordnet tree

    wnids.txt is needed to map wnids to class indices
    """

    def __init__(self, wnid, root='./data', *args,
            path_tree='./data/cifar10/tree.xml',
            path_wnids='./data/cifar10/wnids.txt', **kwargs):
        super().__init__(root=root, *args, **kwargs)
        self.node = Node(wnid, path_tree, path_wnids, self.classes)
        self.original_classes = self.classes
        self.classes = self.node.classes

    def __getitem__(self, i):
        sample, old_label = super().__getitem__(i)
        return sample, self.node.mapping[old_label]


class CIFAR10JointNodes(datasets.CIFAR10):

    def __init__(self, root='./data', *args,
            path_tree='./data/cifar10/tree.xml',
            path_wnids='./data/cifar10/wnids.txt', **kwargs):
        super().__init__(root=root, *args, **kwargs)
        self.nodes = Node.get_nodes(path_tree, path_wnids, self.classes)

        # NOTE: the below is used for computing num_classes, which is ignored
        # anyways. Also, this will break the confusion matrix code
        self.original_classes = self.classes
        self.classes = self.nodes[0].classes

    def __getitem__(self, i):
        sample, old_label = super().__getitem__(i)
        new_label = torch.Tensor([
            node.mapping[old_label] for node in self.nodes
        ]).long()
        return sample, new_label


class CIFAR10PathSanity(datasets.CIFAR10):
    """returns samples that assume all node classifiers are perfect"""

    def __init__(self, root='./data', *args,
            path_tree='./data/cifar10/tree.xml',
            path_wnids='./data/cifar10/wnids.txt', **kwargs):
        super().__init__(root=root, *args, **kwargs)
        wnid_to_node = Node.get_wnid_to_node(path_tree, path_wnids, self.classes)
        wnids = sorted(wnid_to_node)
        self.nodes = [wnid_to_node[wnid] for wnid in wnids]

    def get_sample(self, node, old_label):
        new_label = node.mapping[old_label]
        sample = [0] * node.num_classes
        sample[new_label] = 1
        return sample

    def _get_node_weights(self, node):
        n = node.num_classes
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
        return Node.dim(self.nodes)

    def __getitem__(self, i):
        _, old_label = super().__getitem__(i)

        sample = []
        for dataset in self.nodes:
            sample.extend(self.get_sample(dataset, old_label))
        sample = torch.Tensor(sample)

        return sample, old_label
