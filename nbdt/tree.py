"""Tree and node utilities for navigating the NBDT hierarchy"""
import torchvision.datasets as datasets
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from nbdt.utils import DATASET_TO_NUM_CLASSES, DATASETS
from collections import defaultdict
from nbdt.graph import get_wnids, read_graph, get_leaves, \
    FakeSynset, get_leaf_to_path, wnid_to_synset, wnid_to_name
from nbdt.utils import (
    dataset_to_default_path_graph,
    dataset_to_default_path_wnids,
    hierarchy_to_path_graph)
from nbdt.data import imagenet
import torch.nn as nn
import random


def dataset_to_dummy_classes(dataset):
    assert dataset in DATASETS
    num_classes = DATASET_TO_NUM_CLASSES[dataset]
    return [FakeSynset.create_from_offset(i).wnid for i in range(num_classes)]


class Node:

    def __init__(self, tree, wnid, other_class=False):
        self.tree = tree

        self.wnid = wnid
        self.name = wnid_to_name(wnid)
        self.synset = wnid_to_synset(wnid)

        self.original_classes = tree.classes
        self.num_original_classes = len(self.tree.wnids_leaves)

        self.has_other = other_class and not (self.is_root() or self.is_leaf())
        self.num_children = len(self.succ)

        self.num_classes = self.num_children + int(self.has_other)

        self.class_index_to_child_index, self.child_index_to_class_index = self.build_class_mappings()
        self.classes = self.build_classes()

        assert len(self.classes) == self.num_classes, (
            f'Number of classes {self.num_classes} does not equal number of '
            f'class names found ({len(self.classes)}): {self.classes}'
        )

        self.leaves = list(self.get_leaves())
        self.num_leaves = len(self.leaves)

    def wnid_to_class_index(self, wnid):
        return self.tree.wnids_leaves.index(wnid)

    @property
    def parent(self):
        if not self.parents:
            return None
        return self.parents[0]

    @property
    def pred(self):
        return self.tree.G.pred[self.wnid]

    @property
    def parents(self):
        return [self.tree.wnid_to_node[wnid] for wnid in self.pred]

    @property
    def succ(self):
        return self.tree.G.succ[self.wnid]

    @property
    def children(self):
        return [self.tree.wnid_to_node[wnid] for wnid in self.succ]

    def get_leaves(self):
        return get_leaves(self.tree.G, self.wnid)

    def is_leaf(self):
        return len(self.succ) == 0

    def is_root(self):
        return len(self.pred) == 0

    def build_class_mappings(self):
        if self.is_leaf():
            return {}, {}

        old_to_new = defaultdict(lambda: [])
        new_to_old = defaultdict(lambda: [])
        for new_index, child in enumerate(self.succ):
            for leaf in get_leaves(self.tree.G, child):
                old_index = self.wnid_to_class_index(leaf)
                old_to_new[old_index].append(new_index)
                new_to_old[new_index].append(old_index)

        if not self.has_other:
            return old_to_new, new_to_old

        new_index = self.num_children
        for old in range(self.num_original_classes):
            if old not in old_to_new:
                old_to_new[old].append(new_index)
                new_to_old[new_index].append(old)
        return old_to_new, new_to_old

    def build_classes(self):
        return [
            ','.join([self.original_classes[old] for old in old_indices])
            for new_index, old_indices in sorted(
                self.child_index_to_class_index.items(), key=lambda t: t[0])
        ]

    @property
    def class_counts(self):
        """Number of old classes in each new class"""
        return [len(old_indices) for old_indices in self.child_index_to_class_index]

    @staticmethod
    def dim(nodes):
        return sum([node.num_classes for node in nodes])


class Tree:

    def __init__(
            self, dataset, path_graph=None, path_wnids=None, classes=None,
            hierarchy=None):
        if dataset and hierarchy and not path_graph:
            path_graph = hierarchy_to_path_graph(dataset, hierarchy)
        if dataset and not path_graph:
            path_graph = dataset_to_default_path_graph(dataset)
        if dataset and not path_wnids:
            path_wnids = dataset_to_default_path_wnids(dataset)
        if dataset and not classes:
            classes = dataset_to_dummy_classes(dataset)

        self.dataset = dataset
        self.path_graph = path_graph
        self.path_wnids = path_wnids
        self.classes = classes
        self.G = read_graph(path_graph)
        self.wnids_leaves = get_wnids(path_wnids)
        self.wnid_to_class = {wnid: cls for wnid, cls in zip(self.wnids_leaves, self.classes)}
        self.wnid_to_class_index = {wnid: i for i, wnid in enumerate(self.wnids_leaves)}
        self.wnid_to_node = self.get_wnid_to_node()
        self.nodes = [self.wnid_to_node[wnid] for wnid in sorted(self.wnid_to_node)]
        self.inodes = [node for node in self.nodes if not node.is_leaf()]
        self.leaves = [self.wnid_to_node[wnid] for wnid in self.wnids_leaves]

    @property
    def root(self):
        for node in self.inodes:
            if node.is_root():
                return node
        raise UserWarning('Should not be reachable. Tree should always have root')

    def get_wnid_to_node(self):
        wnid_to_node = {}
        for wnid in self.G:
            wnid_to_node[wnid] = Node(self, wnid)
        return wnid_to_node

    def get_leaf_to_steps(self):
        node = self.inodes[0]
        leaf_to_path = get_leaf_to_path(self.G)
        leaf_to_steps = {}
        for leaf in self.wnids_leaves:
            next_indices = [index for index, _ in leaf_to_path[leaf][1:]] + [-1]
            leaf_to_steps[leaf] = [
                {
                    'node': self.wnid_to_node[wnid],
                    'name': self.wnid_to_node[wnid].name,
                    'next_index': next_index,  # curr node's next child index to traverse
                }
                for next_index, (_, wnid) in zip(next_indices, leaf_to_path[leaf])
            ]
        return leaf_to_steps
