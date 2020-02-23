from utils.data.custom import Node
from utils.utils import generate_fname
from contextlib import contextmanager
import torch
import torch.nn as nn
import random
import os
import csv

from utils import data
import torchvision.datasets as datasets
from utils.utils import (
    DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS, DEFAULT_CIFAR100_TREE,
    DEFAULT_CIFAR100_WNIDS, DEFAULT_TINYIMAGENET200_TREE,
    DEFAULT_TINYIMAGENET200_WNIDS
)

__all__ = ('CIFAR10Tree', 'CIFAR10JointNodes', 'CIFAR10JointTree',
           'CIFAR100Tree', 'CIFAR100JointNodes', 'CIFAR100JointTree',
           'CIFAR10JointDecisionTree', 'CIFAR100JointDecisionTree',
           'CIFAR10BalancedJointNodes', 'CIFAR100BalancedJointNodes',
           'CIFAR10BalancedJointTree', 'CIFAR100BalancedJointTree',
           'TinyImagenet200JointNodes', 'TinyImagenet200BalancedJointNodes',
           'TinyImagenet200JointTree', 'TinyImagenet200BalancedJointTree',
           'CIFAR10FreezeJointNodes', 'CIFAR100FreezeJointNodes',
           'TinyImagenet200FreezeJointNodes', 'CIFAR10FreezeJointTree',
           'CIFAR100FreezeJointTree', 'TinyImagenet200FreezeJointTree',
           'CIFAR100BalancedFreezeJointNodes',
           'CIFAR100BalancedFreezeJointTree', 'CIFAR10IdInitJointTree',
           'CIFAR100IdInitJointTree', 'TinyImagenet200IdInitJointTree',
           'CIFAR10IdInitFreezeJointTree', 'CIFAR100IdInitFreezeJointTree',
           'TinyImagenet200IdInitFreezeJointTree', 'CIFAR10ReweightedJointNodes',
           'CIFAR100ReweightedJointNodes', 'TinyImagenet200ReweightedJointNodes',
           'CIFAR10ReweightedJointTree', 'CIFAR100ReweightedJointTree',
           'TinyImagenet200ReweightedJointTree',
           'CIFAR10IdInitReweightedJointTree',
           'CIFAR100IdInitReweightedJointTree',
           'TinyImagenet200IdInitReweightedJointTree',
           'CIFAR10TreeSup', 'CIFAR100TreeSup', 'TinyImagenet200TreeSup')


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
            path_graph,
            path_wnids,
            pretrained=True,
            num_classes=10):
        super().__init__()

        self.nodes = Node.get_nodes(path_graph, path_wnids, dataset.classes)
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

    accepts_path_graph = True

    def __init__(self, path_graph, path_wnids, dataset, balance_classes=False,
            freeze_backbone=False, balance_class_weights=False):
        super().__init__()

        import models
        # hardcoded for ResNet10
        self.net = models.ResNet10()
        self.nodes = Node.get_nodes(path_graph, path_wnids, dataset.classes)
        self.heads = nn.ModuleList([
            # hardcoded for ResNet10
            nn.Linear(512, node.num_classes)
            for node in self.nodes
        ])
        self.dataset = dataset

        self.balance_classes = balance_classes
        self.balance_class_weights = balance_class_weights
        self.freeze_backbone = freeze_backbone

    def custom_loss(self, criterion, outputs, targets):
        """With some probability, drop over-represented classes"""
        loss = 0
        for output, node in zip(outputs, self.nodes):
            d = output.size(1)
            target, targets = targets[:, :d], targets[:, d:]

            weights = 1.
            if self.balance_classes:
                output, target, skip = self.resample_by_class(output, target, node)
                if skip:
                    continue
            if self.balance_class_weights:
                weights = self.class_weights(output, target, node)
                # TODO(alvin): hard-coded loss lol
                criterion = nn.BCEWithLogitsLoss(weight=weights)

            loss += criterion(output, target)
        return loss

    def resample_by_class(self, output, target, node):
        random = torch.rand(target.size()).to(target.device)

        if node.probabilities.device != target.device:
            node.probabilities = node.probabilities.to(target.device)

        selector = (random < node.probabilities[target]).bool()
        if not selector.any():
            return None, None, True

        output = output[selector]
        target = target[selector]
        return output, target, False

    def class_weights(self, _, target, node):
        if node.class_weights.device != target.device:
            node.class_weights = node.class_weights.to(target.device)
        return node.class_weights

    def custom_prediction(self, outputs):
        preds = [output > 0.5 for output in outputs]
        predicted = torch.cat(preds, dim=1).float()
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

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_CIFAR10_WNIDS,
            dataset=datasets.CIFAR10(root='./data'))


class CIFAR100JointNodes(JointNodes):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_CIFAR100_WNIDS,
            dataset=datasets.CIFAR100(root='./data'))


class TinyImagenet200JointNodes(JointNodes):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_TINYIMAGENET200_WNIDS,
            dataset=data.TinyImagenet200(root='./data'))


class CIFAR10FreezeJointNodes(JointNodes):
    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_CIFAR10_WNIDS,
            freeze_backbone=True, dataset=datasets.CIFAR10(root='./data'))


class CIFAR100FreezeJointNodes(JointNodes):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_CIFAR100_WNIDS,
            freeze_backbone=True, dataset=datasets.CIFAR100(root='./data'))


class TinyImagenet200FreezeJointNodes(JointNodes):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, num_classes=None):
        super().__init__(
            path_graph,
            DEFAULT_TINYIMAGENET200_WNIDS,
            freeze_backbone=True,
            dataset=data.TinyImagenet200(root='./data'))


class CIFAR10BalancedJointNodes(JointNodes):

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_CIFAR10_WNIDS,
            balance_classes=True, dataset=datasets.CIFAR10(root='./data'))


class CIFAR100BalancedJointNodes(JointNodes):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_CIFAR100_WNIDS,
            balance_classes=True, dataset=datasets.CIFAR100(root='./data'))


class TinyImagenet200BalancedJointNodes(JointNodes):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, num_classes=None):
        super().__init__(
            path_graph,
            DEFAULT_TINYIMAGENET200_WNIDS,
            balance_classes=True,
            dataset=data.TinyImagenet200(root='./data'))


class CIFAR10ReweightedJointNodes(JointNodes):

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_CIFAR10_WNIDS,
            balance_class_weights=True, dataset=datasets.CIFAR10(root='./data'))


class CIFAR100ReweightedJointNodes(JointNodes):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_CIFAR100_WNIDS,
            balance_class_weights=True, dataset=datasets.CIFAR100(root='./data'))


class TinyImagenet200ReweightedJointNodes(JointNodes):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, num_classes=None):
        super().__init__(
            path_graph,
            DEFAULT_TINYIMAGENET200_WNIDS,
            balance_class_weights=True,
            dataset=data.TinyImagenet200(root='./data'))


class CIFAR100BalancedFreezeJointNodes(JointNodes):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_CIFAR100_WNIDS,
            balance_classes=True, freeze_backbone=True,
            dataset=datasets.CIFAR100(root='./data'))


class JointTree(nn.Module):
    """
    Final classifier for the nodes trained jointly above, in the
    JointNodes model
    """

    accepts_path_graph = True

    def __init__(self,
            model_name,
            dataset_name,
            path_graph,
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
                path_graph=path_graph
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

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, num_classes=10, pretrained=True):
        super().__init__('CIFAR10JointNodes', 'CIFAR10JointNodes',
            path_graph, DEFAULT_CIFAR10_WNIDS,
            net=CIFAR10JointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR100JointTree(JointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=100, pretrained=True):
        super().__init__('CIFAR100JointNodes', 'CIFAR100JointNodes',
            path_graph, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100JointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class TinyImagenet200JointTree(JointTree):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, num_classes=200, pretrained=True):
        super().__init__('TinyImagenet200JointNodes', 'TinyImagenet200JointNodes',
            path_graph, DEFAULT_TINYIMAGENET200_WNIDS,
            net=TinyImagenet200JointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR10BalancedJointTree(JointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, num_classes=10, pretrained=True):
        super().__init__('CIFAR10BalancedJointNodes', 'CIFAR10JointNodes',
            path_graph, DEFAULT_CIFAR10_WNIDS,
            net=CIFAR10BalancedJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR100BalancedJointTree(JointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=100, pretrained=True):
        super().__init__('CIFAR100BalancedJointNodes', 'CIFAR100JointNodes',
            path_graph, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100BalancedJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class TinyImagenet200BalancedJointTree(JointTree):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, num_classes=200, pretrained=True):
        super().__init__('TinyImagenet200BalancedJointNodes', 'TinyImagenet200JointNodes',
            path_graph, DEFAULT_TINYIMAGENET200_WNIDS,
            net=TinyImagenet200BalancedJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR10ReweightedJointTree(JointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, num_classes=10, pretrained=True):
        super().__init__('CIFAR10ReweightedJointNodes', 'CIFAR10JointNodes',
            path_graph, DEFAULT_CIFAR10_WNIDS,
            net=CIFAR10ReweightedJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR100ReweightedJointTree(JointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=100, pretrained=True):
        super().__init__('CIFAR100ReweightedJointNodes', 'CIFAR100JointNodes',
            path_graph, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100ReweightedJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class TinyImagenet200ReweightedJointTree(JointTree):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, num_classes=200, pretrained=True):
        super().__init__('TinyImagenet200ReweightedJointNodes', 'TinyImagenet200JointNodes',
            path_graph, DEFAULT_TINYIMAGENET200_WNIDS,
            net=TinyImagenet200ReweightedJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR10FreezeJointTree(JointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, num_classes=10, pretrained=True):
        super().__init__('CIFAR10FreezeJointNodes', 'CIFAR10JointNodes',
            path_graph, DEFAULT_CIFAR10_WNIDS,
            net=CIFAR10FreezeJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR100FreezeJointTree(JointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=100, pretrained=True):
        super().__init__('CIFAR100FreezeJointNodes', 'CIFAR100JointNodes',
            path_graph, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100FreezeJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class CIFAR100BalancedFreezeJointTree(JointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=100, pretrained=True):
        super().__init__('CIFAR100BalancedFreezeJointNodes', 'CIFAR100JointNodes',
            path_graph, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100BalancedFreezeJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class TinyImagenet200FreezeJointTree(JointTree):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, num_classes=200, pretrained=True):
        super().__init__('TinyImagenet200FreezeJointNodes', 'TinyImagenet200JointNodes',
            path_graph, DEFAULT_TINYIMAGENET200_WNIDS,
            net=TinyImagenet200FreezeJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained)


class IdInitJointTree(JointTree):

    def __init__(self, *args, initializer=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.initializer = initializer
        self.initialize_weights(initializer)

    def initialize_weights(self, initializer):
        if initializer is not None:
            self.linear.weight = nn.Parameter(initializer.get_weights())


class CIFAR10IdInitJointTree(IdInitJointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, num_classes=10, pretrained=True):
        super().__init__('CIFAR10JointNodes', 'CIFAR10JointNodes',
            path_graph, DEFAULT_CIFAR10_WNIDS,
            net=CIFAR10JointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained,
            initializer=data.CIFAR10PathSanity(path_graph=path_graph))


class CIFAR100IdInitJointTree(IdInitJointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=100, pretrained=True):
        super().__init__('CIFAR100JointNodes', 'CIFAR100JointNodes',
            path_graph, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100JointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained,
            initializer=data.CIFAR100PathSanity(path_graph=path_graph))


class TinyImagenet200IdInitJointTree(IdInitJointTree):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, num_classes=200, pretrained=True):
        super().__init__('TinyImagenet200JointNodes', 'TinyImagenet200JointNodes',
            path_graph, DEFAULT_TINYIMAGENET200_WNIDS,
            net=TinyImagenet200JointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained,
            initializer=data.TinyImagenet200PathSanity(path_graph=path_graph))


class CIFAR10IdInitFreezeJointTree(IdInitJointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, num_classes=10, pretrained=True):
        super().__init__('CIFAR10FreezeJointNodes', 'CIFAR10JointNodes',
            path_graph, DEFAULT_CIFAR10_WNIDS,
            net=CIFAR10FreezeJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained,
            initializer=data.CIFAR10PathSanity(path_graph=path_graph))


class CIFAR100IdInitFreezeJointTree(IdInitJointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=100, pretrained=True):
        super().__init__('CIFAR100FreezeJointNodes', 'CIFAR100JointNodes',
            path_graph, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100FreezeJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained,
            initializer=data.CIFAR100PathSanity(path_graph=path_graph))


class TinyImagenet200IdInitFreezeJointTree(IdInitJointTree):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, num_classes=200, pretrained=True):
        super().__init__('TinyImagenet200FreezeJointNodes', 'TinyImagenet200JointNodes',
            path_graph, DEFAULT_TINYIMAGENET200_WNIDS,
            net=TinyImagenet200FreezeJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained,
            initializer=data.TinyImagenet200PathSanity(path_graph=path_graph))


class CIFAR10IdInitReweightedJointTree(IdInitJointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, num_classes=10, pretrained=True):
        super().__init__('CIFAR10ReweightedJointNodes', 'CIFAR10JointNodes',
            path_graph, DEFAULT_CIFAR10_WNIDS,
            net=CIFAR10ReweightedJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained,
            initializer=data.CIFAR10PathSanity(path_graph=path_graph))


class CIFAR100IdInitReweightedJointTree(IdInitJointTree):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=100, pretrained=True):
        super().__init__('CIFAR100ReweightedJointNodes', 'CIFAR100JointNodes',
            path_graph, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100ReweightedJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained,
            initializer=data.CIFAR100PathSanity(path_graph=path_graph))


class TinyImagenet200IdInitReweightedJointTree(IdInitJointTree):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, num_classes=200, pretrained=True):
        super().__init__('TinyImagenet200ReweightedJointNodes', 'TinyImagenet200JointNodes',
            path_graph, DEFAULT_TINYIMAGENET200_WNIDS,
            net=TinyImagenet200ReweightedJointNodes(path_graph), num_classes=num_classes,
            pretrained=pretrained,
            initializer=data.TinyImagenet200PathSanity(path_graph=path_graph))


class JointDecisionTree(nn.Module):
    """
    Decision tree based inference method using jointly trained nodes
    """

    def __init__(self,
            model_name,
            dataset_name,
            path_graph,
            path_wnids,
            net,
            num_classes=10,
            pretrained=True,
            backtracking=True):
        super().__init__()

        if pretrained:
            fname = generate_fname(
                dataset=dataset_name,
                model=model_name,
                path_graph=path_graph
            )
            print(fname)
            load_checkpoint(net, f'./checkpoint/{fname}.pth')
        self.net = net.net
        self.nodes = net.nodes
        self.heads = net.heads
        self.wnids = [node.wnid for node in self.nodes]

        root_node_wnid = Node.get_root_node_wnid(path_graph)
        self.root_node = self.nodes[self.wnids.index(root_node_wnid)]

        self.dataset_name = dataset_name.replace('JointNodes', '').lower()
        self.num_classes = num_classes
        self.backtracking = backtracking

        self.metrics = []

    def add_sample_metrics(self, pred_class, path, path_probs,
                           nodes_explored, nodes_backtracked, node_probs):
        self.metrics.append({'pred_class' : pred_class,
                             'path' : path,
                             'path_probs' : [round(prob.item(), 4) for prob in path_probs],
                             'nodes_explored' : nodes_explored,
                             'nodes_backtracked' : nodes_backtracked,
                             'node_probs' : node_probs})

    def save_metrics(self, gt_classes, save_dir='./output'):
        save_path = os.path.join(save_dir, self.dataset_name + '_decision_tree_metrics.tsv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, mode='w') as f:
            metrics_writer = csv.writer(f, delimiter='\t')
            metrics_writer.writerow(['Index', 'GT Class', 'Pred Class', 'Correct', 'Path',
                                     'Path Probs', 'Num Nodes Explored',
                                     'Nodes Backtracked', 'Num Node Backtracks', 'Node Probs'])
            for i in range(len(gt_classes)):
                row = []
                row.append(str(i))
                row.append(str(gt_classes[i]))
                row.append(str(self.metrics[i]['pred_class']))
                row.append(str(self.metrics[i]['pred_class'] == gt_classes[i]))
                row.append(str(self.metrics[i]['path']))
                row.append(str(self.metrics[i]['path_probs']))
                row.append(str(self.metrics[i]['nodes_explored']))
                row.append(str(self.metrics[i]['nodes_backtracked']))
                row.append(str(len(self.metrics[i]['nodes_backtracked'])))
                row.append(str(self.metrics[i]['node_probs']))
                metrics_writer.writerow(row)

    def custom_prediction(self, outputs):
        _, predicted = outputs.max(1)
        ignored_idx = outputs[:,0] == -1
        predicted[ignored_idx] = -1
        return predicted

    def forward(self, x):
        assert hasattr(self.net, 'featurize')
        x = self.net.featurize(x)

        outputs = torch.zeros(x.shape[0], self.num_classes)
        for i in range(len(x)):
            pred_old_index = -1
            curr_node = self.root_node
            # Keep track of current path in decision tree for backtracking
            # and how many children have backtracked for each node in path
            curr_path = [self.root_node]
            global_path = [self.root_node.wnid]
            path_child_backtracks = [0]
            path_probs = []
            global_path_probs = []
            nodes_explored = 1
            nodes_backtracked = []
            node_probs = {}
            while curr_node:
                # If all children have backtracked, ignore sample
                if path_child_backtracks[-1] == curr_node.num_classes:
                    break
                # Else take next highest probability child
                node_index = self.wnids.index(curr_node.wnid)
                head = self.heads[node_index]
                output = head(x[i:i+1])[0]
                node_probs[curr_node.wnid] = nn.functional.softmax(output).tolist()
                pred_new_index = sorted(range(len(output)), key=lambda x: -output[x])[path_child_backtracks[-1]]
                global_path_probs.append(nn.functional.softmax(output)[pred_new_index])
                # If "other" predicted, either backtrack or ignore sample
                if pred_new_index == curr_node.num_children:
                    if self.backtracking:
                        # Store node backtrack metric
                        nodes_backtracked.append(curr_node.wnid)
                        # Pop current node from path
                        curr_path.pop()
                        global_path.append(curr_path[-1].wnid)
                        path_child_backtracks.pop()
                        path_probs.pop()
                        # Increment path_child_backtracks
                        path_child_backtracks[-1] += 1
                        # Replace curr_node with parent
                        curr_node = curr_path[-1]
                        nodes_explored += 1
                    else:
                        break
                else:
                    # Store path probability metric
                    path_probs.append(nn.functional.softmax(output)[pred_new_index])
                    next_wnid = curr_node.children_wnids[pred_new_index]
                    global_path.append(next_wnid)
                    if next_wnid in self.wnids:
                        # Explore highest probability child
                        next_node_index = self.wnids.index(next_wnid)
                        curr_node = self.nodes[next_node_index]
                        curr_path.append(curr_node)
                        path_child_backtracks.append(0)
                        nodes_explored += 1
                    else:
                        # Return leaf node
                        pred_old_index = curr_node.new_to_old[pred_new_index][0]
                        curr_node = None
            if pred_old_index >= 0:
                outputs[i,pred_old_index] = 1
            else:
                outputs[i,:] = -1
            self.add_sample_metrics(pred_old_index, global_path, global_path_probs,
                                    nodes_explored, nodes_backtracked, node_probs)
        return outputs.to(x.device)

class CIFAR10JointDecisionTree(JointDecisionTree):

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__('CIFAR10JointNodes', 'CIFAR10JointNodes',
            DEFAULT_CIFAR10_TREE, DEFAULT_CIFAR10_WNIDS,
            net=CIFAR10JointNodes(), num_classes=num_classes,
            pretrained=pretrained)

class CIFAR100JointDecisionTree(JointDecisionTree):

    def __init__(self, num_classes=100, pretrained=True):
        super().__init__('CIFAR100JointNodes', 'CIFAR100JointNodes',
            DEFAULT_CIFAR100_TREE, DEFAULT_CIFAR100_WNIDS,
            net=CIFAR100JointNodes(), num_classes=num_classes,
            pretrained=pretrained)

class TreeSup(nn.Module):

    accepts_path_graph = True

    def __init__(self, path_graph, path_wnids, dataset):
        super().__init__()
        import models

        self.net = models.ResNet10()
        self.nodes = Node.get_nodes(path_graph, path_wnids, dataset.classes)
        self.dataset = dataset

    def load_backbone(self, path):
        checkpoint = torch.load(path)
        state_dict = {
            key.replace('module.', '', 1): value
            for key, value in checkpoint['net'].items()
        }
        self.net.load_state_dict(state_dict, strict=False)

    def custom_loss(self, criterion, outputs, targets):
        loss = criterion(outputs, targets)
        for output, target in zip(outputs, targets):
            old_label = int(target)
            for node in self.nodes:
                new_labels = node.old_to_new_classes[target]
                if not new_labels:  # if new_label = [], node does not include target
                    continue
                output_sub = torch.cat([
                    torch.sum(output[node.new_to_old_classes[new_label]])
                    for new_label in range(node.num_classes)
                ])
                loss += criterion(output_sub, new_labels[0])
        return loss

    def forward(self, x):
        return self.net(x)


class CIFAR10TreeSup(TreeSup):

    def __init__(self, path_graph=DEFAULT_CIFAR10_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_CIFAR10_WNIDS,
            dataset=datasets.CIFAR10(root='./data'))


class CIFAR100TreeSup(TreeSup):

    def __init__(self, path_graph=DEFAULT_CIFAR100_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_CIFAR100_WNIDS,
            dataset=datasets.CIFAR100(root='./data'))


class TinyImagenet200TreeSup(TreeSup):

    def __init__(self, path_graph=DEFAULT_TINYIMAGENET200_TREE, num_classes=None):
        super().__init__(path_graph, DEFAULT_TINYIMAGENET200_WNIDS,
            dataset=data.TinyImagenet200(root='./data'))
