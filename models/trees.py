from utils.nmn_datasets import Node
from utils.utils import generate_fname
from contextlib import contextmanager
import torch
import torch.nn as nn
import random
import os

from utils.utils import (
    DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS, DEFAULT_CIFAR100_TREE,
    DEFAULT_CIFAR100_WNIDS, DEFAULT_TINYIMAGENET200_TREE,
    DEFAULT_TINYIMAGENET200_WNIDS
)

__all__ = ('CIFAR10Tree', 'CIFAR10JointNodes', 'CIFAR10JointTree',
           'CIFAR100Tree', 'CIFAR100JointNodes', 'CIFAR100JointTree',
           'CIFAR10BalancedJointNodes', 'CIFAR100BalancedJointNodes',
           'CIFAR10BalancedJointTree', 'CIFAR100BalancedJointTree',
           'TinyImagenet200JointNodes', 'TinyImagenet200BalancedJointNodes',
           'TinyImagenet200JointTree', 'TinyImagenet200BalancedJointTree',
           'CIFAR10FreezeJointNodes', 'CIFAR100FreezeJointNodes',
           'TinyImagenet200FreezeJointNodes', 'CIFAR10FreezeJointTree',
           'CIFAR100FreezeJointTree', 'TinyImagenet200FreezeJointTree',
           'CIFAR100BalancedFreezeJointNodes',
           'CIFAR100BalancedFreezeJointTree')


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


class Tree(nn.Module):
    """returns samples from all node classifiers"""

    def __init__(self,
            dataset,
            path_tree,
            path_wnids,
            pretrained=True,
            num_classes=10):
        super().__init__()

        self.nodes = Node.get_nodes(path_tree, path_wnids)
        self.nets = nn.ModuleList([
            self.get_net_for_node(dataset, node, pretrained) for node in self.nodes])
        self.linear = nn.Linear(self.get_input_dim(), num_classes)

    def get_net_for_node(self, dataset, node, pretrained):
        import models
        # TODO: WARNING: the model and paths are hardcoded
        net = models.ResNet10(num_classes=node.num_classes)

        if pretrained:
            load_checkpoint(net, f'./checkpoint/ckpt-{dataset}Node-ResNet10-{node.wnid}.pth')
        return net

    def get_input_dim(self):
        return Node.dim(self.nodes)

    def forward(self, old_sample):
        with torch.no_grad():
            sample = []
            for net in self.nets:
                feature = net(old_sample)
                sample.append(feature)
            sample = torch.cat(sample, 1)
        return self.linear(sample)


class CIFAR10Tree(Tree):

    def __init__(self, *args, pretrained=True, num_classes=10, **kwargs):
        super().__init__('CIFAR10', DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS,
            pretrained=pretrained, num_classes=num_classes, **kwargs)


class CIFAR100Tree(Tree):

    def __init__(self, *args, pretrained=True, num_classes=100, **kwargs):
        super().__init__('CIFAR100', DEFAULT_CIFAR100_TREE, DEFAULT_CIFAR100_WNIDS,
            pretrained=pretrained, num_classes=num_classes, **kwargs)


class JointNodes(nn.Module):
    """
    Requires that model have a featurize method. Like training individual nodes,
    except all nodes share convolutions. Thus, all nodes are trained jointly.
    """

    accepts_path_tree = True

    def __init__(self, path_tree, path_wnids, balance_classes=False,
            freeze_backbone=False):
        super().__init__()

        import models
        # hardcoded for ResNet10
        self.net = models.ResNet10()
        self.nodes = Node.get_nodes(path_tree, path_wnids)
        self.heads = nn.ModuleList([
            # hardcoded for ResNet10
            nn.Linear(512, node.num_classes)
            for node in self.nodes
        ])

        self.balance_classes = balance_classes
        self.freeze_backbone = freeze_backbone

    def custom_loss(self, criterion, outputs, targets):
        """With some probability, drop over-represented classes"""
        loss = 0
        for output, target, node in zip(outputs, targets.T, self.nodes):

            if self.balance_classes:
                random = torch.rand(target.size()).to(target.device)

                if node.probabilities.device != target.device:
                    node.probabilities = node.probabilities.to(target.device)

                selector = (random < node.probabilities[target]).bool()
                if not selector.any():
                    continue
                output = output[selector]
                target = target[selector]
            loss += criterion(output, target)
        return loss

    def custom_prediction(self, outputs):
        preds = []
        for output in outputs:
            _, pred = output.max(dim=1)
            preds.append(pred[:, None])
        predicted = torch.cat(preds, dim=1)
        return predicted

    def load_backbone(self, path):
        checkpoint = torch.load(path)
        state_dict = {
            key.replace('module.', '', 1): value
            for key, value in checkpoint['net'].items()
        }
        state_dict.pop('linear.weight')
        state_dict.pop('linear.bias')
        self.net.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        """Note this returns unconventional output.

        The output is (h, n, k) for h heads (number of trainable nodes in the
        tree), n samples, and k classes.
        """
        assert hasattr(self.net, 'featurize'), \
            'Net needs a `featurize` method to work with CIFAR10JointNodes ' \
            'training'
        context = torch.no_grad() if self.freeze_backbone else noop()
        with context:
            x = self.net.featurize(x)

        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        return outputs


# num_classes is ignored
class CIFAR10JointNodes(JointNodes):

    def __init__(self, path_tree=DEFAULT_CIFAR10_TREE, num_classes=None):
        super().__init__(path_tree, DEFAULT_CIFAR10_WNIDS)


class CIFAR100JointNodes(JointNodes):

    def __init__(self, path_tree=DEFAULT_CIFAR100_TREE, num_classes=None):
        super().__init__(path_tree, DEFAULT_CIFAR100_WNIDS)


class TinyImagenet200JointNodes(JointNodes):

    def __init__(self, path_tree=DEFAULT_TINYIMAGENET200_TREE, num_classes=None):
        super().__init__(path_tree, DEFAULT_TINYIMAGENET200_WNIDS)


class CIFAR10FreezeJointNodes(JointNodes):
    def __init__(self, path_tree=DEFAULT_CIFAR10_TREE, num_classes=None):
        super().__init__(path_tree, DEFAULT_CIFAR10_WNIDS,
            freeze_backbone=True)


class CIFAR100FreezeJointNodes(JointNodes):

    def __init__(self, path_tree=DEFAULT_CIFAR100_TREE, num_classes=None):
        super().__init__(path_tree, DEFAULT_CIFAR100_WNIDS,
            freeze_backbone=True)


class TinyImagenet200FreezeJointNodes(JointNodes):

    def __init__(self, path_tree=DEFAULT_TINYIMAGENET200_TREE, num_classes=None):
        super().__init__(
            path_tree,
            DEFAULT_TINYIMAGENET200_WNIDS,
            freeze_backbone=True)


class CIFAR10BalancedJointNodes(JointNodes):

    def __init__(self, path_tree=DEFAULT_CIFAR10_TREE, num_classes=None):
        super().__init__(path_tree, DEFAULT_CIFAR10_WNIDS,
            balance_classes=True)


class CIFAR100BalancedJointNodes(JointNodes):

    def __init__(self, path_tree=DEFAULT_CIFAR100_TREE, num_classes=None):
        super().__init__(path_tree, DEFAULT_CIFAR100_WNIDS,
            balance_classes=True)


class TinyImagenet200BalancedJointNodes(JointNodes):

    def __init__(self, path_tree=DEFAULT_TINYIMAGENET200_TREE, num_classes=None):
        super().__init__(
            path_tree,
            DEFAULT_TINYIMAGENET200_WNIDS,
            balance_classes=True)


class CIFAR100BalancedFreezeJointNodes(JointNodes):

    def __init__(self, path_tree=DEFAULT_CIFAR100_TREE, num_classes=None):
        super().__init__(path_tree, DEFAULT_CIFAR100_WNIDS,
            balance_classes=True, freeze_backbone=True)


class JointTree(nn.Module):
    """
    Final classifier for the nodes trained jointly above, in the
    JointNodes model
    """

    accepts_path_tree = True

    def __init__(self,
            model_name,
            dataset_name,
            path_tree,
            path_wnids,
            net,
            num_classes=10,
            pretrained=True,
            softmax=False):
        super().__init__()

        self.net = net
        if pretrained:
            # TODO: should use generate_fname
            fname = generate_fname(
                dataset=dataset_name,
                model=model_name,
                path_tree=path_tree
            )
            load_checkpoint(self.net, f'checkpoint/{fname}.pth')
        self.linear = nn.Linear(Node.dim(self.net.nodes), num_classes)

        self.softmax = nn.Softmax(dim=1)
        self._softmax = softmax

    def forward(self, x):
        with torch.no_grad():
            x = self.net(x)
            if self._softmax:
                x = self.softmax(x)
        x = torch.cat(x, dim=1)
        x = self.linear(x)
        return x

    def softmax(self, x):
        # not helpful -- dropped jointTree from 68% to 60%, balancedJointTree
        # from 64% to 31%
        return [self.softmax(xi) for xi in x]


class CIFAR10JointTree(JointTree):

    def __init__(self, path_tree=DEFAULT_CIFAR10_TREE, num_classes=10, pretrained=True):
        super().__init__('CIFAR10JointNodes', 'CIFAR10JointNodes',
            path_tree, DEFAULT_CIFAR10_WNIDS,
            net=CIFAR10JointNodes(path_tree), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR100JointTree(JointTree):

    def __init__(self, path_tree=DEFAULT_CIFAR100_TREE, num_classes=100, pretrained=True):
        super().__init__('CIFAR100JointNodes', 'CIFAR100JointNodes',
            path_tree, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100JointNodes(path_tree), num_classes=num_classes,
            pretrained=pretrained)


class TinyImagenet200JointTree(JointTree):

    def __init__(self, path_tree=DEFAULT_TINYIMAGENET200_TREE, num_classes=200, pretrained=True):
        super().__init__('TinyImagenet200JointNodes', 'TinyImagenet200JointNodes',
            path_tree, DEFAULT_TINYIMAGENET200_WNIDS,
            net=TinyImagenet200JointNodes(path_tree), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR10BalancedJointTree(JointTree):

    def __init__(self, path_tree=DEFAULT_CIFAR10_TREE, num_classes=10, pretrained=True):
        super().__init__('CIFAR10BalancedJointNodes', 'CIFAR10JointNodes',
            path_tree, DEFAULT_CIFAR10_WNIDS,
            net=CIFAR10BalancedJointNodes(path_tree), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR100BalancedJointTree(JointTree):

    def __init__(self, path_tree=DEFAULT_CIFAR100_TREE, num_classes=100, pretrained=True):
        super().__init__('CIFAR100BalancedJointNodes', 'CIFAR100JointNodes',
            path_tree, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100BalancedJointNodes(path_tree), num_classes=num_classes,
            pretrained=pretrained)


class TinyImagenet200BalancedJointTree(JointTree):

    def __init__(self, path_tree=DEFAULT_TINYIMAGENET200_TREE, num_classes=200, pretrained=True):
        super().__init__('TinyImagenet200BalancedJointNodes', 'TinyImagenet200JointNodes',
            path_tree, DEFAULT_TINYIMAGENET200_WNIDS,
            net=TinyImagenet200BalancedJointNodes(path_tree), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR10FreezeJointTree(JointTree):

    def __init__(self, path_tree=DEFAULT_CIFAR10_TREE, num_classes=10, pretrained=True):
        super().__init__('CIFAR10FreezeJointNodes', 'CIFAR10JointNodes',
            path_tree, DEFAULT_CIFAR10_WNIDS,
            net=CIFAR10FreezeJointNodes(path_tree), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR100FreezeJointTree(JointTree):

    def __init__(self, path_tree=DEFAULT_CIFAR100_TREE, num_classes=100, pretrained=True):
        super().__init__('CIFAR100FreezeJointNodes', 'CIFAR100JointNodes',
            path_tree, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100FreezeJointNodes(path_tree), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR100BalancedFreezeJointTree(JointTree):

    def __init__(self, path_tree=DEFAULT_CIFAR100_TREE, num_classes=100, pretrained=True):
        super().__init__('CIFAR100BalancedFreezeJointNodes', 'CIFAR100JointNodes',
            path_tree, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100BalancedFreezeJointNodes(path_tree), num_classes=num_classes,
            pretrained=pretrained)


class TinyImagenet200FreezeJointTree(JointTree):

    def __init__(self, path_tree=DEFAULT_TINYIMAGENET200_TREE, num_classes=200, pretrained=True):
        super().__init__('TinyImagenet200FreezeJointNodes', 'TinyImagenet200JointNodes',
            path_tree, DEFAULT_TINYIMAGENET200_WNIDS,
            net=TinyImagenet200FreezeJointNodes(path_tree), num_classes=num_classes,
            pretrained=pretrained)
