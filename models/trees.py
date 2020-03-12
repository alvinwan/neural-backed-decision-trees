from utils.data.custom import Node
from utils.utils import generate_fname
from contextlib import contextmanager
import torch
import torch.nn as nn
import random
import os
import csv
import numpy as np
from collections import defaultdict
from utils.loss import NBDTHardLoss, NBDTSoftLoss

import torch.nn.functional as F
from utils import data
import torchvision.datasets as datasets
from utils.utils import (
    DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS, DEFAULT_CIFAR100_TREE,
    DEFAULT_CIFAR100_WNIDS, DEFAULT_TINYIMAGENET200_TREE,
    DEFAULT_TINYIMAGENET200_WNIDS, DEFAULT_IMAGENET1000_TREE,
    DEFAULT_IMAGENET1000_WNIDS,
)

__all__ = ('CIFAR10TreeSup', 'CIFAR100TreeSup', 'TinyImagenet200TreeSup',
           'Imagenet1000TreeSup',
           'CIFAR10TreeBayesianSup', 'CIFAR100TreeBayesianSup',
           'TinyImagenet200TreeBayesianSup', 'Imagenet1000TreeBayesianSup')


@contextmanager
def noop():
    yield None


def load_checkpoint(net, path):
    if not os.path.exists(path):
        print(f' * Failed to load model. No such path found: {path}')
        return
    checkpoint = torch.load(path)
    # hacky fix lol
    state_dict = {key.replace('module.', '', 1): value for key, value in checkpoint['net'].items()}
    net.load_state_dict(state_dict)


def get_featurizer(net):
    if hasattr(net, 'featurize'):
        return net.featurize
    elif hasattr(net, 'features'):  # for imgclsmob models
        def featurize(x):
            x = net.features(x)
            x = x.view(x.size(0), -1)
            return x
        return featurize
    raise UserWarning('Model needs either a `.features` or a `.featurize` method')


def get_linear(net):
    if hasattr(net, 'linear'):
        return net.linear
    elif hasattr(net, 'output'):   # for imgclsmob models
        return net.output
    raise UserWarning('Model needs either a `.linear` or a `.output` method')


class TreeSup(nn.Module):

    accepts_path_graph = True
    accepts_max_leaves_supervised = True
    accepts_min_leaves_supervised = True
    accepts_tree_supervision_weight = True
    accepts_weighted_average = True
    accepts_fine_tune = True
    accepts_backbone = True

    def __init__(self, path_graph, path_wnids, dataset, backbone='ResNet10',
            num_classes=10, max_leaves_supervised=-1, min_leaves_supervised=-1,
            tree_supervision_weight=1., weighted_average=False,
            fine_tune=False):
        super().__init__()
        import models

        assert hasattr(models, backbone), (
            f'The specified backbone {backbone} is not a valid backbone name. '
            'Did you mean to use --path-backbone instead?'
        )
        self.net = getattr(models, backbone)(num_classes=num_classes)
        self.loss = HardTreeSupLoss(path_graph, path_wnids, dataset.classes,
            max_leaves_supervised, min_leaves_supervised,
            tree_supervision_weight, weighted_average, fine_tune)

        self.fine_tune = fine_tune
        if self.fine_tune:
            has_featurizer = hasattr(self.net, 'featurize') or hasattr(self.net, 'features')
            has_linear = hasattr(self.net, 'linear') or hasattr(self.net, 'output')
            assert has_featurizer and has_linear, (
                f'Network {self.net} does not have both .featurize and .linear '
                'methods/operations, which are needed to only fine-tune the '
                'model.'
            )

        self._loaded_backbone = False

    def load_backbone(self, path):
        self._loaded_backbone = True

        kwargs = {}
        if not torch.cuda.is_available():
            kwargs['map_location'] = torch.device('cpu')

        checkpoint = torch.load(path, **kwargs)
        if 'net' in checkpoint.keys():
            state_dict = {
                key.replace('module.', '', 1): value
                for key, value in checkpoint['net'].items()
            }
        else:
            state_dict = checkpoint
        self.net.load_state_dict(state_dict, strict=False)

    def custom_loss(self, criterion, outputs, targets):
        return criterion(outputs, targets) + self.loss(outputs, targets)

    def forward(self, x):
        if self.fine_tune:
            assert self._loaded_backbone, (
                'Model is being fine-tuned but no backbone weights loaded. '
                'Please pass the --backbone flag'
            )
            with torch.no_grad():
                x = get_featurizer(self.net)(x)
            return get_linear(self.net)(x)
        return self.net(x)


class CIFAR10TreeSup(TreeSup):

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, backbone='ResNet10',
            num_classes=10, max_leaves_supervised=-1, min_leaves_supervised=-1,
            tree_supervision_weight=1., weighted_average=False,
            fine_tune=False):
        super().__init__(path_graph, DEFAULT_CIFAR10_WNIDS,
            dataset=datasets.CIFAR10(root='./data'),
            backbone=backbone,
            num_classes=num_classes,
            max_leaves_supervised=max_leaves_supervised,
            min_leaves_supervised=min_leaves_supervised,
            tree_supervision_weight=tree_supervision_weight,
            weighted_average=weighted_average, fine_tune=fine_tune)


class CIFAR100TreeSup(TreeSup):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, backbone='ResNet10',
            num_classes=100, max_leaves_supervised=-1, min_leaves_supervised=-1,
            tree_supervision_weight=1., weighted_average=False,
            fine_tune=False):
        super().__init__(path_graph, DEFAULT_CIFAR100_WNIDS,
            dataset=datasets.CIFAR100(root='./data'),
            backbone=backbone,
            num_classes=num_classes,
            max_leaves_supervised=max_leaves_supervised,
            min_leaves_supervised=min_leaves_supervised,
            tree_supervision_weight=tree_supervision_weight,
            weighted_average=weighted_average, fine_tune=fine_tune)


class TinyImagenet200TreeSup(TreeSup):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, backbone='ResNet10',
            num_classes=200, max_leaves_supervised=-1, min_leaves_supervised=-1,
            tree_supervision_weight=1., weighted_average=False,
            fine_tune=False):
        super().__init__(path_graph, DEFAULT_TINYIMAGENET200_WNIDS,
            dataset=data.TinyImagenet200(root='./data'),
            backbone=backbone,
            num_classes=num_classes,
            max_leaves_supervised=max_leaves_supervised,
            min_leaves_supervised=min_leaves_supervised,
            tree_supervision_weight=tree_supervision_weight,
            weighted_average=weighted_average, fine_tune=fine_tune)


class Imagenet1000TreeSup(TreeSup):

    def __init__(self, path_graph=DEFAULT_IMAGENET1000_TREE, backbone='ResNet10',
            num_classes=1000, max_leaves_supervised=-1, min_leaves_supervised=-1,
            tree_supervision_weight=1., weighted_average=False,
            fine_tune=False):
        super().__init__(path_graph, DEFAULT_IMAGENET1000_WNIDS,
            dataset=data.Imagenet1000(root='./data'),
            backbone=backbone,
            num_classes=num_classes,
            max_leaves_supervised=max_leaves_supervised,
            min_leaves_supervised=min_leaves_supervised,
            tree_supervision_weight=tree_supervision_weight,
            weighted_average=weighted_average, fine_tune=fine_tune)


class TreeBayesianSup(TreeSup):

    def __init__(self, path_graph, path_wnids, dataset, backbone='ResNet10',
            num_classes=10, max_leaves_supervised=-1, min_leaves_supervised=-1,
            tree_supervision_weight=1., weighted_average=False,
            fine_tune=False):
        super().__init__(path_graph, path_wnids, dataset, backbone,
            num_classes, max_leaves_supervised, min_leaves_supervised,
            tree_supervision_weight, weighted_average, fine_tune)
        self.loss = SoftTreeSupLoss(path_graph, path_wnids, dataset.classes,
            max_leaves_supervised, min_leaves_supervised,
            tree_supervision_weight, weighted_average, fine_tune)
        self.num_classes = len(self.dataset.classes)

    def custom_loss(self, criterion, outputs, targets):
        return criterion(outputs, targets) + self.loss(outputs, targets)


class CIFAR10TreeBayesianSup(TreeBayesianSup):

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, backbone='ResNet10',
            num_classes=10, max_leaves_supervised=-1, min_leaves_supervised=-1,
            tree_supervision_weight=1., weighted_average=False,
            fine_tune=False):
        super().__init__(path_graph, DEFAULT_CIFAR10_WNIDS,
            dataset=datasets.CIFAR10(root='./data'),
            backbone=backbone,
            num_classes=num_classes,
            max_leaves_supervised=max_leaves_supervised,
            min_leaves_supervised=min_leaves_supervised,
            tree_supervision_weight=tree_supervision_weight,
            weighted_average=weighted_average, fine_tune=fine_tune)


class CIFAR100TreeBayesianSup(TreeBayesianSup):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, backbone='ResNet10',
            num_classes=100, max_leaves_supervised=-1, min_leaves_supervised=-1,
            tree_supervision_weight=1., weighted_average=False,
            fine_tune=False):
        super().__init__(path_graph, DEFAULT_CIFAR100_WNIDS,
            dataset=datasets.CIFAR100(root='./data'),
            backbone=backbone,
            num_classes=num_classes,
            max_leaves_supervised=max_leaves_supervised,
            min_leaves_supervised=min_leaves_supervised,
            tree_supervision_weight=tree_supervision_weight,
            weighted_average=weighted_average, fine_tune=fine_tune)


class TinyImagenet200TreeBayesianSup(TreeBayesianSup):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, backbone='ResNet10',
            num_classes=200, max_leaves_supervised=-1, min_leaves_supervised=-1,
            tree_supervision_weight=1., weighted_average=False,
            fine_tune=False):
        super().__init__(path_graph, DEFAULT_TINYIMAGENET200_WNIDS,
            dataset=data.TinyImagenet200(root='./data'),
            backbone=backbone,
            num_classes=num_classes,
            max_leaves_supervised=max_leaves_supervised,
            min_leaves_supervised=min_leaves_supervised,
            tree_supervision_weight=tree_supervision_weight,
            weighted_average=weighted_average, fine_tune=fine_tune)


class Imagenet1000TreeBayesianSup(TreeBayesianSup):

    def __init__(self, path_graph=DEFAULT_IMAGENET1000_TREE, backbone='ResNet10',
            num_classes=1000, max_leaves_supervised=-1, min_leaves_supervised=-1,
            tree_supervision_weight=1., weighted_average=False,
            fine_tune=False):
        super().__init__(path_graph, DEFAULT_IMAGENET1000_WNIDS,
            dataset=data.Imagenet1000(root='./data'),
            backbone=backbone,
            num_classes=num_classes,
            max_leaves_supervised=max_leaves_supervised,
            min_leaves_supervised=min_leaves_supervised,
            tree_supervision_weight=tree_supervision_weight,
            weighted_average=weighted_average, fine_tune=fine_tune)
