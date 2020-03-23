import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from nbdt.data.custom import Node, dataset_to_dummy_classes
from nbdt.utils import (
    Colors, dataset_to_default_path_graph, dataset_to_default_path_wnids,
    hierarchy_to_path_graph
)

__all__ = names = ('HardTreeSupLoss', 'SoftTreeSupLoss', 'CrossEntropyLoss')
keys = (
    'path_graph', 'path_wnids', 'max_leaves_supervised',
    'min_leaves_supervised', 'weighted_average', 'tree_supervision_weight',
    'classes'
)

def add_arguments(parser):
    parser.add_argument('--hierarchy',
                        help='Hierarchy to use. If supplied, will be used to '
                        'generate --path-graph. --path-graph takes precedence.')
    parser.add_argument('--path-graph', help='Path to graph-*.json file.')  # WARNING: hard-coded suffix -build in generate_fname
    parser.add_argument('--path-wnids', help='Path to wnids.txt file.')
    parser.add_argument('--max-leaves-supervised', type=int, default=-1,
                        help='Maximum number of leaves a node can have to '
                        'contribute loss, in tree-supervised training.')
    parser.add_argument('--min-leaves-supervised', type=int, default=-1,
                        help='Minimum number of leaves a node must have to '
                        'contribute loss, in tree-supervised training.')
    parser.add_argument('--weighted-average', action='store_true',
                        help='Use weighted average instead of average, for cluster '
                        'centers.')
    parser.add_argument('--tree-supervision-weight', type=float, default=1,
                        help='Weight assigned to tree supervision losses')


def set_default_values(args):
    assert not args.hierarchy and args.path_graph, \
        'Only one, between --hierarchy and --path-graph can be provided.'
    if args.hierarchy and not args.path_graph:
        args.path_graph = hierarchy_to_path_graph(args.hierarchy)
    if not args.path_graph:
        args.path_graph = dataset_to_default_path_graph(args.dataset)
    if not args.path_wnids:
        args.path_wnids = dataset_to_default_path_wnids(args.dataset)


CrossEntropyLoss = nn.CrossEntropyLoss


class TreeSupLoss(nn.Module):

    accepts_path_graph = True
    accepts_path_wnids = True
    accepts_classes = True
    accepts_max_leaves_supervised = True
    accepts_min_leaves_supervised = True
    accepts_tree_supervision_weight = True
    accepts_weighted_average = True
    accepts_classes = lambda trainset, **kwargs: trainset.classes

    def __init__(self,
            dataset,
            criterion,
            path_graph=None,
            path_wnids=None,
            classes=None,
            hierarchy=None,
            **kwargs):
        super().__init__()

        if dataset and hierarchy and not path_graph:
            path_graph = hierarchy_to_path_graph(dataset, hierarchy)
        if dataset and not path_graph:
            path_graph = dataset_to_default_path_graph(dataset)
        if dataset and not path_wnids:
            path_wnids = dataset_to_default_path_wnids(dataset)
        if dataset and not classes:
            classes = dataset_to_dummy_classes(dataset)

        self.init(dataset, criterion, path_graph, path_wnids, classes, **kwargs)

    def init(self,
            dataset,
            criterion,
            path_graph,
            path_wnids,
            classes,
            max_leaves_supervised=-1,
            min_leaves_supervised=-1,
            tree_supervision_weight=1.,
            weighted_average=False):
        """
        Extra init method makes clear which arguments are finally necessary for
        this class to function. The constructor for this class may generate
        some of these required arguments if initially missing.
        """
        self.dataset = dataset
        self.num_classes = len(classes)
        self.nodes = Node.get_nodes(path_graph, path_wnids, classes)
        self.max_leaves_supervised = max_leaves_supervised
        self.min_leaves_supervised = min_leaves_supervised
        self.tree_supervision_weight = tree_supervision_weight
        self.weighted_average = weighted_average
        self.criterion = criterion

    @staticmethod
    def assert_output_not_nbdt(outputs):
        """
        >>> x = torch.randn(1, 3, 224, 224)
        >>> TreeSupLoss.assert_output_not_nbdt(x)  # all good!
        >>> x._nbdt_output_flag = True
        >>> TreeSupLoss.assert_output_not_nbdt(x)  #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        AssertionError: ...
        >>> from nbdt.model import NBDT
        >>> import torchvision.models as models
        >>> model = models.resnet18()
        >>> y = model(x)
        >>> TreeSupLoss.assert_output_not_nbdt(y)  # all good!
        >>> model = NBDT('CIFAR10', model)
        >>> y = model(x)
        >>> TreeSupLoss.assert_output_not_nbdt(y)  #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        AssertionError: ...
        """
        assert getattr(outputs, '_nbdt_output_flag', False) is False, (
            "Uh oh! Looks like you passed an NBDT model's output to an NBDT "
            "loss. NBDT losses are designed to take in the *original* model's "
            "outputs, as input. NBDT models are designed to only be used "
            "during validation and inference, not during training. Confused? "
            " Check out github.com/alvinwan/nbdt#convert-neural-networks-to-decision-trees"
            " for examples and instructions.")


class HardTreeSupLoss(TreeSupLoss):

    def forward(self, outputs, targets):
        """
        The supplementary losses are all uniformly down-weighted so that on
        average, each sample incurs half of its loss from standard cross entropy
        and half of its loss from all nodes.

        The code below is structured weirdly to minimize number of tensors
        constructed and moved from CPU to GPU or vice versa. In short,
        all outputs and targets for nodes with 2 children are gathered and
        moved onto GPU at once. Same with those with 3, with 4 etc. On CIFAR10,
        the max is 2. On CIFAR100, the max is 8.
        """
        self.assert_output_not_nbdt(outputs)

        loss = self.criterion(outputs, targets)
        num_losses = outputs.size(0) * len(self.nodes) / 2.

        outputs_subs = defaultdict(lambda: [])
        targets_subs = defaultdict(lambda: [])
        targets_ints = [int(target) for target in targets.cpu().long()]
        for node in self.nodes:
            if self.max_leaves_supervised > 0 and \
                    node.num_leaves > self.max_leaves_supervised:
                continue

            if self.min_leaves_supervised > 0 and \
                    node.num_leaves < self.min_leaves_supervised:
                continue

            _, outputs_sub, targets_sub = HardTreeSupLoss.inference(
                node, outputs, targets_ints, self.weighted_average)

            key = node.num_classes
            assert outputs_sub.size(0) == len(targets_sub)
            outputs_subs[key].append(outputs_sub)
            targets_subs[key].extend(targets_sub)

        for key in outputs_subs:
            outputs_sub = torch.cat(outputs_subs[key], dim=0)
            targets_sub = torch.Tensor(targets_subs[key]).long().to(outputs_sub.device)

            if not outputs_sub.size(0):
                continue
            fraction = outputs_sub.size(0) / float(num_losses) \
                * self.tree_supervision_weight
            loss += self.criterion(outputs_sub, targets_sub) * fraction
        return loss

    @classmethod
    def inference(cls, node, outputs, targets, weighted_average=False):
        classes = [node.old_to_new_classes[int(t)] for t in targets]
        selector = [bool(cls) for cls in classes]
        targets_sub = [cls[0] for cls in classes if cls] if targets else None

        _outputs = outputs[selector]
        if _outputs.size(0) == 0:
            return selector, _outputs[:, :node.num_classes], targets_sub
        outputs_sub = cls.get_output_sub(_outputs, node, weighted_average)
        return selector, outputs_sub, targets_sub

    @staticmethod
    def get_output_sub(_outputs, node, weighted_average=False):
        if weighted_average:
            node.move_leaf_weights_to(_outputs.device)

        weights = [
            node.new_to_leaf_weights[new_label] if weighted_average else 1
            for new_label in range(node.num_classes)
        ]
        return torch.stack([
            (_outputs * weight).T
            [node.new_to_old_classes[new_label]].mean(dim=0)
            for new_label, weight in zip(range(node.num_classes), weights)
        ]).T


class SoftTreeSupLoss(HardTreeSupLoss):

    def forward(self, outputs, targets):
        self.assert_output_not_nbdt(outputs)

        loss = self.criterion(outputs, targets)
        bayesian_outputs = SoftTreeSupLoss.inference(
            self.nodes, outputs, self.num_classes, self.weighted_average)
        loss += self.criterion(bayesian_outputs, targets) * self.tree_supervision_weight
        return loss

    @classmethod
    def inference(cls, nodes, outputs, num_classes, weighted_average=False):
        """
        In theory, the loop over children below could be replaced with just a
        few lines:

            for index_child in range(len(node.children)):
                old_indexes = node.new_to_old_classes[index_child]
                class_probs[:,old_indexes] *= output[:,index_child][:,None]

        However, we collect all indices first, so that only one tensor operation
        is run.
        """
        class_probs = torch.ones((outputs.size(0), num_classes)).to(outputs.device)
        for node in nodes:
            output = cls.get_output_sub(outputs, node, weighted_average)
            output = F.softmax(output, dim=1)

            old_indices, new_indices = [], []
            for index_child in range(len(node.children)):
                old = node.new_to_old_classes[index_child]
                old_indices.extend(old)
                new_indices.extend([index_child] * len(old))

            assert len(set(old_indices)) == len(old_indices), (
                'All old indices must be unique in order for this operation '
                'to be correct.'
            )
            class_probs[:,old_indices] *= output[:,new_indices]
        return class_probs
