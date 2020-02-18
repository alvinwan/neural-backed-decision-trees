import torchvision.datasets as datasets
import xml.etree.ElementTree as ET
from utils.xmlutils import get_leaves, remove
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from utils.utils import (
    DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS, DEFAULT_CIFAR100_TREE,
    DEFAULT_CIFAR100_WNIDS, DEFAULT_TINYIMAGENET200_TREE,
    DEFAULT_TINYIMAGENET200_WNIDS
)
from . import custom_datasets


__all__ = names = ('CIFAR10Node', 'CIFAR10JointNodes', 'CIFAR10PathSanity',
                   'CIFAR100Node', 'CIFAR100JointNodes',
                   'TinyImagenet200JointNodes', 'CIFAR100PathSanity',
                   'TinyImagenet200PathSanity')


class Node:

    def __init__(self, wnid,
            path_tree=DEFAULT_CIFAR10_TREE,
            path_wnids=DEFAULT_CIFAR10_WNIDS,
            classes=()):
        self.wnid = wnid
        self.original_classes = classes

        with open(path_wnids) as f:
            wnids = [line.strip() for line in f.readlines()]
        self.num_original_classes = len(wnids)

        tree = ET.parse(path_tree)

        # generate mapping from wnid to class
        self.mapping = {}
        node = tree.find('.//synset[@wnid="{}"]'.format(wnid))
        assert node is not None, f'Failed to find node with wnid {wnid}'
        children = node.getchildren()
        n = len(children)
        assert n > 0, 'Cannot build dataset for leaf node.'
        self.num_children = n
        self.num_classes = n

        for new_index, child in enumerate(children):
            for leaf in get_leaves(child):
                wnid = leaf.get('wnid')
                old_index = wnids.index(wnid)
                self.mapping[old_index] = new_index
        if len(self.mapping) < self.num_original_classes:
            self.num_classes += 1

        for old_index in range(self.num_original_classes):
            if old_index not in self.mapping:
                self.mapping[old_index] = n

        self.new_to_old = [[] for _ in range(self.num_classes)]
        for old_index in range(self.num_original_classes):
            new_index = self.mapping[old_index]
            self.new_to_old[new_index].append(old_index)

        self.classes = []
        if self.original_classes:
            self.classes = [','.join(
                [self.original_classes[old_index] for old_index in old_indices])
                for old_indices in self.new_to_old
            ]
        self._probabilities = None

        self.children_wnids = [child.get('wnid') for child in children]

    @property
    def class_counts(self):
        """Number of old classes in each new class"""
        return [len(old_indices) for old_indices in self.new_to_old]

    @property
    def probabilities(self):
        """Calculates probability of training on the ith class.

        If the class contains more than `resample_threshold` samples, the
        probability is lower, as it is likely to cause severe class imbalance
        issues.
        """
        if self._probabilities is None:
            reference = min(self.class_counts)
            self._probabilities = torch.Tensor([
                min(1, reference / len(old_indices))
                for old_indices in self.new_to_old
            ])
        return self._probabilities

    @probabilities.setter
    def probabilities(self, probabilities):
        self._probabilities = probabilities

    @staticmethod
    def get_wnid_to_node(path_tree, path_wnids, classes=()):
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
    def get_nodes(path_tree, path_wnids, classes=()):
        wnid_to_node = Node.get_wnid_to_node(path_tree, path_wnids, classes)
        wnids = sorted(wnid_to_node)
        nodes = [wnid_to_node[wnid] for wnid in wnids]
        return nodes

    @staticmethod
    def get_root_node_wnid(path_tree):
        tree = ET.parse(path_tree)
        for node in tree.iter():
            wnid = node.get('wnid')
            if wnid is not None:
                return wnid
        return None

    @staticmethod
    def dim(nodes):
        return sum([node.num_classes for node in nodes])


class NodeDataset(Dataset):
    """Creates dataset for a specific node in the CIFAR10 wordnet tree

    wnids.txt is needed to map wnids to class indices
    """

    needs_wnid = True

    def __init__(self, wnid, path_tree, path_wnids, dataset):
        super().__init__()

        self.dataset = dataset
        self.node = Node(wnid, path_tree, path_wnids, dataset.classes)
        self.original_classes = dataset.classes
        self.classes = self.node.classes

    def __getitem__(self, i):
        sample, old_label = self.dataset[i]
        return sample, self.node.mapping[old_label]

    def __len__(self):
        return len(self.dataset)


class CIFAR10Node(NodeDataset):

    def __init__(self, wnid, *args, root='./data', **kwargs):
        super().__init__(wnid, DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS,
            dataset=datasets.CIFAR10(*args, root=root, **kwargs))


class CIFAR100Node(NodeDataset):

    def __init__(self, wnid, *args, root='./data', **kwargs):
        super().__init__(wnid, DEFAULT_CIFAR100_TREE, DEFAULT_CIFAR100_WNIDS,
            dataset=datasets.CIFAR100(*args, root=root, **kwargs))


class JointNodesDataset(Dataset):

    accepts_path_tree = True

    def __init__(self, path_tree, path_wnids, dataset):
        super().__init__()
        self.nodes = Node.get_nodes(path_tree, path_wnids, dataset.classes)
        self.dataset = dataset
        # NOTE: the below is used for computing num_classes, which is ignored
        # anyways. Also, this will break the confusion matrix code
        self.original_classes = dataset.classes
        self.classes = self.nodes[0].classes

    def __getitem__(self, i):
        sample, old_label = self.dataset[i]
        new_label = torch.Tensor([
            node.mapping[old_label] for node in self.nodes
        ]).long()
        return sample, new_label

    def __len__(self):
        return len(self.dataset)


class CIFAR10JointNodes(JointNodesDataset):

    def __init__(self,
            *args,
            path_tree=DEFAULT_CIFAR10_TREE,
            path_wnids=DEFAULT_CIFAR10_WNIDS,
            root='./data',
            **kwargs):
        super().__init__(path_tree, path_wnids,
            dataset=datasets.CIFAR10(*args, root=root, **kwargs))


class CIFAR100JointNodes(JointNodesDataset):

    def __init__(self,
            *args,
            path_tree=DEFAULT_CIFAR100_TREE,
            path_wnids=DEFAULT_CIFAR100_WNIDS,
            root='./data',
            **kwargs):
        super().__init__(path_tree, path_wnids,
            dataset=datasets.CIFAR100(*args, root=root, **kwargs))


class TinyImagenet200JointNodes(JointNodesDataset):

    def __init__(self,
            *args,
            path_tree=DEFAULT_TINYIMAGENET200_TREE,
            path_wnids=DEFAULT_TINYIMAGENET200_WNIDS,
            root='./data',
            **kwargs):
        super().__init__(path_tree, path_wnids,
            dataset=custom_datasets.TinyImagenet200(*args, root=root, **kwargs))


class PathSanityDataset(Dataset):
    """returns samples that assume all node classifiers are perfect"""

    def __init__(self, path_tree, path_wnids, dataset):
        super().__init__()
        self.nodes = Node.get_nodes(path_tree, path_wnids, dataset.classes)
        self.dataset = dataset
        self.classes = dataset.classes

    def get_sample(self, node, old_label):
        new_label = node.mapping[old_label]
        sample = [0] * node.num_classes
        sample[new_label] = 1
        return sample

    def _get_node_weights(self, node):
        n = node.num_classes
        k = len(self.dataset.classes)

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
        _, old_label = self.dataset[i]

        sample = []
        for dataset in self.nodes:
            sample.extend(self.get_sample(dataset, old_label))
        sample = torch.Tensor(sample)

        return sample, old_label

    def __len__(self):
        return len(self.dataset)


class CIFAR10PathSanity(PathSanityDataset):

    def __init__(self, *args, root='./data', **kwargs):
        super().__init__(DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS,
            dataset=datasets.CIFAR10(*args, root=root, **kwargs))


class CIFAR100PathSanity(PathSanityDataset):

    def __init__(self, *args, root='./data', **kwargs):
        super().__init__(DEFAULT_CIFAR100_TREE, DEFAULT_CIFAR100_WNIDS,
            dataset=datasets.CIFAR100(*args, root=root, **kwargs))


class TinyImagenet200PathSanity(PathSanityDataset):

    def __init__(self, *args, root='./data', **kwargs):
        super().__init__(DEFAULT_TINYIMAGENET200_TREE, DEFAULT_TINYIMAGENET200_WNIDS,
            dataset=custom_datasets.TinyImagenet200(*args, root=root, **kwargs))
