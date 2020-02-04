from utils.datasets import Node
import torch
import torch.nn as nn
import random
import os

from utils.utils import (
    DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS, DEFAULT_CIFAR100_TREE,
    DEFAULT_CIFAR100_WNIDS
)

__all__ = ('CIFAR10Tree', 'CIFAR10JointNodes', 'CIFAR10JointTree',
           'CIFAR100Tree', 'CIFAR100JointNodes', 'CIFAR100JointTree')


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

    def __init__(self, path_tree, path_wnids):
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

    def custom_loss(self, criterion, outputs, targets):
        """With some probability, drop over-represented classes"""
        loss = 0
        for output, target, node in zip(outputs, targets.T, self.nodes):
            random = torch.rand(target.size())

            if node.probabilities.device != target.device:
                probabilities = node.probabilities.to(target.device)

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

    def forward(self, x):
        """Note this returns unconventional output.

        The output is (h, n, k) for h heads (number of trainable nodes in the
        tree), n samples, and k classes.
        """
        assert hasattr(self.net, 'featurize'), \
            'Net needs a `featurize` method to work with CIFAR10JointNodes ' \
            'training'
        x = self.net.featurize(x)

        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        return outputs


# num_classes is ignored
class CIFAR10JointNodes(JointNodes):

    def __init__(self, num_classes=None):
        super().__init__(DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS)


class CIFAR100JointNodes(JointNodes):

    def __init__(self, num_classes=None):
        super().__init__(DEFAULT_CIFAR100_TREE, DEFAULT_CIFAR100_WNIDS)


class JointTree(nn.Module):
    """
    Final classifier for the nodes trained jointly above, in the
    JointNodes model
    """

    def __init__(self,
            dataset,
            path_tree,
            path_wnids,
            net,
            num_classes=10,
            pretrained=True):
        super().__init__()

        self.net = net
        if pretrained:
            # TODO: should use generate_fname
            load_checkpoint(self.net, f'./checkpoint/ckpt-{dataset}JointNodes-{dataset}JointNodes.pth')
        self.linear = nn.Linear(Node.dim(self.net.nodes), num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.net(x)
        x = torch.cat(x, dim=1)
        x = self.linear(x)
        return x


class CIFAR10JointTree(JointTree):

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__('CIFAR10', DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS,
            net=CIFAR10JointNodes(), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR100JointTree(JointTree):

    def __init__(self, num_classes=100, pretrained=True):
        super().__init__('CIFAR100', DEFAULT_CIFAR100_TREE, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100JointNodes(), num_classes=num_classes,
            pretrained=pretrained)
