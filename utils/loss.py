import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from utils.data.custom import Node
from utils.utils import Colors

__all__ = names = ('HardTreeSupLoss', 'SoftTreeSupLoss')


def add_arguments(parser):
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


class TreeSupLoss(nn.Module):

    def __init__(self, path_graph, path_wnids, classes,
            max_leaves_supervised=-1, min_leaves_supervised=-1,
            tree_supervision_weight=1., weighted_average=False,
            criterion=nn.CrossEntropyLoss()):
        super().__init__()

        self.num_classes = len(classes)
        self.nodes = Node.get_nodes(path_graph, path_wnids, classes)
        self.max_leaves_supervised = max_leaves_supervised
        self.min_leaves_supervised = min_leaves_supervised
        self.tree_supervision_weight = tree_supervision_weight
        self.weighted_average = weighted_average
        self.criterion = criterion

    @classmethod
    def from_args_classes(cls, args, classes):

        for key in ('path_graph', 'path_wnids', 'max_leaves_supervised',
                'min_leaves_supervised', 'tree_supervision_weight',
                'weighted_average'):
            Colors.cyan(f'{key}:\t{getattr(args, key, None)}')

        return cls(
            path_graph=args.path_graph,
            path_wnids=args.path_wnids,
            classes=classes,
            max_leaves_supervised=args.max_leaves_supervised,
            min_leaves_supervised=args.min_leaves_supervised,
            tree_supervision_weight=args.tree_supervision_weight,
            weighted_average=args.weighted_average)


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
        targets_sub = [cls[0] for cls in classes if cls]

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

    def custom_loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        bayesian_outputs = SoftTreeSupLoss.inference(
            self.nodes, outputs, self.num_classes, self.weighted_average)
        loss += self.criterion(bayesian_outputs, targets) * self.tree_supervision_weight
        return loss

    @classmethod
    def inference(cls, nodes, outputs, num_classes, weighted_average=False):
        # Compute bayesian class probability outputs
        class_probs = torch.ones((outputs.size(0), num_classes)).to(outputs.device)
        for node in nodes:
            output = cls.get_output_sub(outputs, node, weighted_average)
            output = F.softmax(output)
            for index_child in range(len(node.children)):
                old_indexes = node.new_to_old_classes[index_child]
                for index_old in old_indexes:
                    class_probs[:,index_old] = class_probs[:,index_old].clone() * output[:,index_child]
        return class_probs
