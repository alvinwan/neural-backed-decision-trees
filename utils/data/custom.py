import torchvision.datasets as datasets
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from utils.utils import (
    DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS, DEFAULT_CIFAR100_TREE,
    DEFAULT_CIFAR100_WNIDS, DEFAULT_TINYIMAGENET200_TREE,
    DEFAULT_TINYIMAGENET200_WNIDS
)
from collections import defaultdict
from utils.graph import get_wnids, read_graph, get_leaves, get_non_leaves
from . import imagenet
import torch.nn as nn
import random


__all__ = names = ('CIFAR10Node', 'CIFAR10JointNodes', 'CIFAR10PathSanity',
                   'CIFAR100Node', 'CIFAR100JointNodes',
                   'TinyImagenet200JointNodes', 'CIFAR100PathSanity',
                   'TinyImagenet200PathSanity', 'CIFAR10IncludeLabels',
                   'CIFAR100IncludeLabels', 'TinyImagenet200IncludeLabels',
                   'CIFAR10ExcludeLabels', 'CIFAR100ExcludeLabels',
                   'TinyImagenet200ExcludeLabels',
                   'CIFAR10ResampleLabels', 'CIFAR100ResampleLabels',
                   'TinyImagenet200ResampleLabels',
                   'CIFAR10JointNodesSingle', 'CIFAR100JointNodesSingle',
                   'TinyImagenet200JointNodesSingle')


class Node:

    def __init__(self, wnid, classes,
            path_graph=DEFAULT_CIFAR10_TREE,
            path_wnids=DEFAULT_CIFAR10_WNIDS):
        self.wnid = wnid
        self.wnids = get_wnids(path_wnids)
        self.G = read_graph(path_graph)

        self.original_classes = classes
        self.num_original_classes = len(self.wnids)

        assert not self.is_leaf(), 'Cannot build dataset for leaf'
        self.num_children = len(self.get_children())
        self.num_classes = self.num_children

        self.old_to_new_classes, self.new_to_old_classes = \
            self.build_class_mappings()
        self.classes = self.build_classes()

        assert len(self.classes) == self.num_classes, (
            self.classes, self.num_classes)

        self._probabilities = None
        self._class_weights = None

    def get_parents(self):
        return self.G.pred[self.wnid]

    def get_children(self):
        return self.G.succ[self.wnid]

    def is_leaf(self):
        return len(self.get_children()) == 0

    def is_root(self):
        return len(self.get_parents()) == 0

    def build_class_mappings(self):
        old_to_new = defaultdict(lambda: [])
        new_to_old = defaultdict(lambda: [])
        for new_index, child in enumerate(self.get_children()):
            for leaf in get_leaves(self.G, child, include_root=True):
                old_index = self.wnids.index(leaf)
                old_to_new[old_index].append(new_index)
                new_to_old[new_index].append(old_index)
        return old_to_new, new_to_old

    def build_classes(self):
        return [
            ','.join([self.original_classes[old_index] for old_index in old_indices])
            for old_indices in self.new_to_old_classes.values()
        ]

    @property
    def class_counts(self):
        """Number of old classes in each new class"""
        return [len(old_indices) for old_indices in self.new_to_old_classes]

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
                for old_indices in self.new_to_old_classes
            ])
        return self._probabilities

    @probabilities.setter
    def probabilities(self, probabilities):
        self._probabilities = probabilities

    @property
    def class_weights(self):
        if self._class_weights is None:
            self._class_weights = self.probabilities
        return self._class_weights

    @class_weights.setter
    def class_weights(self, class_weights):
        self._class_weights = class_weights

    @staticmethod
    def get_wnid_to_node(path_graph, path_wnids, classes):
        wnid_to_node = {}
        G = read_graph(path_graph)
        for wnid in get_non_leaves(G):
            wnid_to_node[wnid] = Node(
                wnid, classes, path_graph=path_graph, path_wnids=path_wnids)
        return wnid_to_node

    @staticmethod
    def get_nodes(path_graph, path_wnids, classes):
        wnid_to_node = Node.get_wnid_to_node(path_graph, path_wnids, classes)
        wnids = sorted(wnid_to_node)
        nodes = [wnid_to_node[wnid] for wnid in wnids]
        return nodes

    @staticmethod
    def get_root_node_wnid(path_graph):
        raise UserWarning('Root node may have wnid now')
        tree = ET.parse(path_graph)
        for node in tree.iter():
            wnid = node.get('wnid')
            if wnid is not None:
                return wnid
        return None

    @staticmethod
    def dim(nodes):
        return sum([node.num_classes for node in nodes])


class NodeDataset(Dataset):
    """Creates dataset for a specific node in the wordnet tree

    wnids.txt is needed to map wnids to class indices
    """

    needs_wnid = True

    def __init__(self, wnid, path_graph, path_wnids, dataset, node=None):
        super().__init__()

        self.dataset = dataset
        self.node = node or Node(wnid, dataset.classes, path_graph, path_wnids)
        self.original_classes = dataset.classes
        self.classes = self.node.classes

    @staticmethod
    def multi_label_to_k_hot(node, labels):
        k_hot = [0] * node.num_classes
        for label in labels:
            k_hot[label] = 1
        return torch.Tensor(k_hot)

    def __getitem__(self, i):
        sample, old_label = self.dataset[i]
        new_labels = self.node.old_to_new_classes[old_label]
        new_label = NodeDataset.multi_label_to_k_hot(self.node, new_labels)
        return sample, new_label

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

    accepts_path_graph = True
    criterion = nn.BCEWithLogitsLoss

    def __init__(self, path_graph, path_wnids, dataset):
        super().__init__()
        self.nodes = Node.get_nodes(path_graph, path_wnids, dataset.classes)
        self.dataset = dataset
        self.path_graph = path_graph
        # NOTE: the below is used for computing num_classes, which is ignored
        # anyways. Also, this will break the confusion matrix code
        self.original_classes = dataset.classes
        self.classes = self.nodes[0].classes

    def get_label(self, old_label):
        return torch.cat([
            NodeDataset.multi_label_to_k_hot(node, node.old_to_new_classes[old_label])
            for node in self.nodes
        ], dim=0)

    def __getitem__(self, i):
        sample, old_label = self.dataset[i]
        new_label = self.get_label(old_label)
        return sample, new_label

    def __len__(self):
        return len(self.dataset)


class JointNodesSingleDataset(JointNodesDataset):

    criterion = nn.CrossEntropyLoss

    def get_label(self, old_label):
        path_length_per_leaf = [
            len(node.old_to_new_classes[old_label])
            for node in self.nodes
        ]
        assert all([length <= 1 for length in path_length_per_leaf]), (
            f'Dataset asks for single_path=True but tree {self.path_graph}'
            f' has leaves with multiple paths: {path_length_per_leaf}. '
            'Did you mean to use --path-graph to pass a new graph?'
        )
        return torch.Tensor([
            (node.old_to_new_classes[old_label] or [-1])[0]
            for node in self.nodes]).long()


class CIFAR10JointNodes(JointNodesDataset):

    def __init__(self,
            *args,
            path_graph=DEFAULT_CIFAR10_TREE,
            path_wnids=DEFAULT_CIFAR10_WNIDS,
            root='./data',
            **kwargs):
        super().__init__(path_graph, path_wnids,
            dataset=datasets.CIFAR10(*args, root=root, **kwargs))


class CIFAR100JointNodes(JointNodesDataset):

    def __init__(self,
            *args,
            path_graph=DEFAULT_CIFAR100_TREE,
            path_wnids=DEFAULT_CIFAR100_WNIDS,
            root='./data',
            **kwargs):
        super().__init__(path_graph, path_wnids,
            dataset=datasets.CIFAR100(*args, root=root, **kwargs))


class TinyImagenet200JointNodes(JointNodesDataset):

    def __init__(self,
            *args,
            path_graph=DEFAULT_TINYIMAGENET200_TREE,
            path_wnids=DEFAULT_TINYIMAGENET200_WNIDS,
            root='./data',
            **kwargs):
        super().__init__(path_graph, path_wnids,
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs))


class CIFAR10JointNodesSingle(JointNodesSingleDataset):

    def __init__(self,
            *args,
            path_graph=DEFAULT_CIFAR10_TREE,
            path_wnids=DEFAULT_CIFAR10_WNIDS,
            root='./data',
            **kwargs):
        super().__init__(path_graph, path_wnids,
            dataset=datasets.CIFAR10(*args, root=root, **kwargs))


class CIFAR100JointNodesSingle(JointNodesSingleDataset):

    def __init__(self,
            *args,
            path_graph=DEFAULT_CIFAR100_TREE,
            path_wnids=DEFAULT_CIFAR100_WNIDS,
            root='./data',
            **kwargs):
        super().__init__(path_graph, path_wnids,
            dataset=datasets.CIFAR100(*args, root=root, **kwargs))


class TinyImagenet200JointNodesSingle(JointNodesSingleDataset):

    def __init__(self,
            *args,
            path_graph=DEFAULT_TINYIMAGENET200_TREE,
            path_wnids=DEFAULT_TINYIMAGENET200_WNIDS,
            root='./data',
            **kwargs):
        super().__init__(path_graph, path_wnids,
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs))


class PathSanityDataset(Dataset):
    """returns samples that assume all node classifiers are perfect"""

    def __init__(self, path_graph, path_wnids, dataset):
        super().__init__()
        self.path_graph = path_graph
        self.path_wnids = path_wnids

        self.nodes = Node.get_nodes(path_graph, path_wnids, dataset.classes)
        self.dataset = dataset
        self.classes = dataset.classes

    def get_sample(self, node, old_label):
        new_labels = node.old_to_new_classes[old_label]
        sample = [0] * node.num_classes
        for new_label in new_labels:
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

    def __init__(self,
            *args,
            path_graph=DEFAULT_CIFAR10_TREE,
            path_wnids=DEFAULT_CIFAR10_WNIDS,
            root='./data',
            **kwargs):
        super().__init__(path_graph, path_wnids,
            dataset=datasets.CIFAR10(*args, root=root, **kwargs))


class CIFAR100PathSanity(PathSanityDataset):

    def __init__(self,
            *args,
            path_graph=DEFAULT_CIFAR100_TREE,
            path_wnids=DEFAULT_CIFAR100_WNIDS,
            root='./data',
            **kwargs):
        super().__init__(path_graph, path_wnids,
            dataset=datasets.CIFAR100(*args, root=root, **kwargs))


class TinyImagenet200PathSanity(PathSanityDataset):

    def __init__(self,
            *args,
            path_graph=DEFAULT_TINYIMAGENET200_TREE,
            path_wnids=DEFAULT_TINYIMAGENET200_WNIDS,
            root='./data',
            **kwargs):
        super().__init__(path_graph, path_wnids,
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs))


class ResampleLabelsDataset(Dataset):
    """
    Dataset that includes only the labels provided, with a limited number of
    samples. Note that labels are integers in [0, k) for a k-class dataset.

    :drop_classes bool: Modifies the dataset so that it is only a m-way
                        classification where m of k classes are kept. Otherwise,
                        the problem is still k-way.
    """

    accepts_probability_labels = True

    def __init__(self, dataset, probability_labels=1, drop_classes=False, seed=0):
        self.dataset = dataset
        self.classes = dataset.classes
        self.labels = list(range(len(self.classes)))
        self.probability_labels = self.get_probability_labels(dataset, probability_labels)

        self.drop_classes = drop_classes
        if self.drop_classes:
            self.classes, self.labels = self.get_classes_after_drop(
                dataset, probability_labels)

        assert self.labels, 'No labels are included in `include_labels`'

        self.new_to_old = self.build_index_mapping(seed=seed)

    def get_probability_labels(self, dataset, ps):
        if not isinstance(ps, (tuple, list)):
            return [ps] * len(dataset.classes)
        if len(ps) == 1:
            return ps * len(dataset.classes)
        assert len(ps) == len(dataset.classes), (
            f'Length of probabilities vector {len(ps)} must equal that of the '
            f'dataset classes {len(dataset.classes)}.'
        )
        return ps

    def apply_drop(self, dataset, ps):
        classes = [
            cls for p, cls in zip(ps, dataset.classes)
            if p > 0
        ]
        labels = [i for p, i in zip(ps, range(len(dataset.classes))) if p > 0]
        return classes, labels

    def build_index_mapping(self, seed=0):
        """Iterates over all samples in dataset.

        Remaps all to-be-included samples to [0, n) where n is the number of
        samples with a class in the whitelist.

        Additionally, the outputted list is truncated to match the number of
        desired samples.
        """
        random.seed(seed)

        new_to_old = []
        for old, (_, label) in enumerate(self.dataset):
            if random.random() < self.probability_labels[label]:
                new_to_old.append(old)
        return new_to_old

    def __getitem__(self, index_new):
        index_old = self.new_to_old[index_new]
        sample, label_old = self.dataset[index_old]

        label_new = label_old
        if self.drop_classes:
            label_new = self.include_labels.index(label_old)

        return sample, label_new

    def __len__(self):
        return len(self.new_to_old)


class IncludeLabelsDataset(ResampleLabelsDataset):

    accepts_include_labels = True
    accepts_probability_labels = False

    def __init__(self, dataset, include_labels=(0,)):
        super().__init__(dataset, probability_labels=[
            int(cls in include_labels) for cls in range(len(dataset.classes))
        ])


class CIFAR10ResampleLabels(ResampleLabelsDataset):

    def __init__(self, *args, root='./data', probability_labels=1, **kwargs):
        super().__init__(
            dataset=datasets.CIFAR10(*args, root=root, **kwargs),
            probability_labels=probability_labels)


class CIFAR100ResampleLabels(ResampleLabelsDataset):

    def __init__(self, *args, root='./data', probability_labels=1, **kwargs):
        super().__init__(
            dataset=datasets.CIFAR100(*args, root=root, **kwargs),
            probability_labels=probability_labels)


class TinyImagenet200ResampleLabels(ResampleLabelsDataset):

    def __init__(self, *args, root='./data', probability_labels=1, **kwargs):
        super().__init__(
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs),
            probability_labels=probability_labels)


class IncludeClassesDataset(IncludeLabelsDataset):
    """
    Dataset that includes only the labels provided, with a limited number of
    samples. Note that classes are strings, like 'cat' or 'dog'.
    """

    accepts_include_labels = False
    accepts_include_classes = True

    def __init__(self, dataset, include_classes=()):
        super().__init__(dataset, include_labels=[
                dataset.classes.index(cls) for cls in include_classes
            ])


class CIFAR10IncludeLabels(IncludeLabelsDataset):

    def __init__(self, *args, root='./data', include_labels=(0,), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR10(*args, root=root, **kwargs),
            include_labels=include_labels)


class CIFAR100IncludeLabels(IncludeLabelsDataset):

    def __init__(self, *args, root='./data', include_labels=(0,), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR100(*args, root=root, **kwargs),
            include_labels=include_labels)


class TinyImagenet200IncludeLabels(IncludeLabelsDataset):

    def __init__(self, *args, root='./data', include_labels=(0,), **kwargs):
        super().__init__(
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs),
            include_labels=include_labels)


class ExcludeLabelsDataset(IncludeLabelsDataset):

    accepts_include_labels = False
    accepts_exclude_labels = True

    def __init__(self, dataset, exclude_labels=(0,)):
        k = len(dataset.classes)
        include_labels = set(range(k)) - set(exclude_labels)
        super().__init__(
            dataset=dataset,
            include_labels=include_labels)


class CIFAR10ExcludeLabels(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', exclude_labels=(0,), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR10(*args, root=root, **kwargs),
            exclude_labels=exclude_labels)


class CIFAR100ExcludeLabels(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', exclude_labels=(0,), **kwargs):
        super().__init__(
            dataset=datasets.CIFAR100(*args, root=root, **kwargs),
            exclude_labels=exclude_labels)


class TinyImagenet200ExcludeLabels(ExcludeLabelsDataset):

    def __init__(self, *args, root='./data', exclude_labels=(0,), **kwargs):
        super().__init__(
            dataset=imagenet.TinyImagenet200(*args, root=root, **kwargs),
            exclude_labels=exclude_labels)
