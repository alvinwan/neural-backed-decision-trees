import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from nbdt.tree import Node, Tree
from nbdt.model import HardEmbeddedDecisionRules, SoftEmbeddedDecisionRules
from math import log
from nbdt.utils import (
    Colors,
    dataset_to_default_path_graph,
    dataset_to_default_path_wnids,
    hierarchy_to_path_graph,
    coerce_tensor,
    uncoerce_tensor,
)
from pathlib import Path
import os

__all__ = names = (
    "HardTreeSupLoss",
    "SoftTreeSupLoss",
    "SoftTreeLoss",
    "CrossEntropyLoss",
)


def add_arguments(parser):
    parser.add_argument(
        "--xent-weight", "--xw", type=float, help="Weight for cross entropy term"
    )
    parser.add_argument(
        "--xent-weight-end",
        "--xwe",
        type=float,
        help="Weight for cross entropy term at end of training."
        "If not set, set to cew",
    )
    parser.add_argument(
        "--xent-weight-power", "--xwp", type=float, help="Raise progress to this power."
    )
    parser.add_argument(
        "--tree-supervision-weight",
        "--tsw",
        type=float,
        default=1,
        help="Weight assigned to tree supervision losses",
    )
    parser.add_argument(
        "--tree-supervision-weight-end",
        "--tswe",
        type=float,
        help="Weight assigned to tree supervision losses at "
        "end of training. If not set, this is equal to tsw",
    )
    parser.add_argument(
        "--tree-supervision-weight-power",
        "--tswp",
        type=float,
        help="Raise progress to this power. > 1 to trend "
        "towards tswe more slowly. < 1 to trend more quickly",
    )
    parser.add_argument(
        "--tree-start-epochs",
        "--tse",
        type=int,
        help="epoch count to start tree supervision loss from (generate tree at that pt)",
    )
    parser.add_argument(
        "--tree-update-end-epochs",
        "--tuene",
        type=int,
        help="epoch count to stop generating new trees at",
    )
    parser.add_argument(
        "--tree-update-every-epochs",
        "--tueve",
        type=int,
        help="Recompute tree from weights every (this many) epochs",
    )


def set_default_values(args):
    assert not (
        args.hierarchy and args.path_graph
    ), "Only one, between --hierarchy and --path-graph can be provided."
    if args.hierarchy and not args.path_graph:
        args.path_graph = hierarchy_to_path_graph(args.dataset, args.hierarchy)
    if not args.path_graph:
        args.path_graph = dataset_to_default_path_graph(args.dataset)
    if not args.path_wnids:
        args.path_wnids = dataset_to_default_path_wnids(args.dataset)


CrossEntropyLoss = nn.CrossEntropyLoss


class TreeSupLoss(nn.Module):

    accepts_tree = lambda tree, **kwargs: tree
    accepts_criterion = lambda criterion, **kwargs: criterion
    accepts_dataset = lambda trainset, **kwargs: trainset.__class__.__name__
    accepts_path_graph = True
    accepts_path_wnids = True
    accepts_tree_supervision_weight = True
    accepts_classes = lambda trainset, **kwargs: trainset.classes
    accepts_hierarchy = True
    accepts_tree_supervision_weight_end = True
    accepts_tree_supervision_weight_power = True
    accepts_xent_weight = True
    accepts_xent_weight_end = True
    accepts_xent_weight_power = True

    def __init__(
        self,
        dataset,
        criterion,
        path_graph=None,
        path_wnids=None,
        classes=None,
        hierarchy=None,
        Rules=HardEmbeddedDecisionRules,
        tree=None,
        tree_supervision_weight=1.0,
        tree_supervision_weight_end=None,
        tree_supervision_weight_power=1,  # 1 for linear
        xent_weight=1,
        xent_weight_end=None,
        xent_weight_power=1,
    ):
        super().__init__()

        if not tree:
            tree = Tree(dataset, path_graph, path_wnids, classes, hierarchy=hierarchy)
        self.num_classes = len(tree.classes)
        self.tree = tree
        self.rules = Rules(tree=tree)
        self.tree_supervision_weight = tree_supervision_weight
        self.tree_supervision_weight_end = (
            tree_supervision_weight_end
            if tree_supervision_weight_end is not None
            else tree_supervision_weight
        )
        self.tree_supervision_weight_power = tree_supervision_weight_power
        self.xent_weight = xent_weight
        self.xent_weight_end = (
            xent_weight_end if xent_weight_end is not None else xent_weight
        )
        self.xent_weight_power = xent_weight_power
        self.criterion = criterion
        self.progress = 1
        self.epochs = 0

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
        >>> model = NBDT('CIFAR10', model, arch='ResNet18')
        >>> y = model(x)
        >>> TreeSupLoss.assert_output_not_nbdt(y)  #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        AssertionError: ...
        """
        assert getattr(outputs, "_nbdt_output_flag", False) is False, (
            "Uh oh! Looks like you passed an NBDT model's output to an NBDT "
            "loss. NBDT losses are designed to take in the *original* model's "
            "outputs, as input. NBDT models are designed to only be used "
            "during validation and inference, not during training. Confused? "
            " Check out github.com/alvinwan/nbdt#convert-neural-networks-to-decision-trees"
            " for examples and instructions."
        )

    def forward_tree(self, outputs, targets):
        raise NotImplementedError()

    def get_weight(self, start, end, power=1):
        progress = self.progress ** power
        return (1 - progress) * start + progress * end

    def forward(self, outputs, targets):
        loss_xent = self.criterion(outputs, targets)
        loss_tree = self.forward_tree(outputs, targets)

        tree_weight = self.get_weight(
            self.tree_supervision_weight,
            self.tree_supervision_weight_end,
            self.tree_supervision_weight_power,
        )
        xent_weight = self.get_weight(
            self.xent_weight, self.xent_weight_end, self.xent_weight_power
        )
        return loss_xent * xent_weight + loss_tree * tree_weight

    def set_epoch(self, cur, total):
        self.epochs = cur
        self.progress = cur / total
        if hasattr(super(), "set_epoch"):
            super().set_epoch(cur, total)


class HardTreeSupLoss(TreeSupLoss):
    def forward_tree(self, outputs, targets):
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

        loss = 0
        num_losses = outputs.size(0) * len(self.tree.inodes) / 2.0

        outputs_subs = defaultdict(lambda: [])
        targets_subs = defaultdict(lambda: [])
        targets_ints = [int(target) for target in targets.cpu().long()]
        for node in self.tree.inodes:
            (
                _,
                outputs_sub,
                targets_sub,
            ) = HardEmbeddedDecisionRules.get_node_logits_filtered(
                node, outputs, targets_ints
            )

            key = node.num_classes
            assert outputs_sub.size(0) == len(targets_sub)
            outputs_subs[key].append(outputs_sub)
            targets_subs[key].extend(targets_sub)

        for key in outputs_subs:
            outputs_sub = torch.cat(outputs_subs[key], dim=0)
            targets_sub = torch.Tensor(targets_subs[key]).long().to(outputs_sub.device)

            if not outputs_sub.size(0):
                continue
            fraction = (
                outputs_sub.size(0) / float(num_losses) * self.tree_supervision_weight
            )
            loss += self.criterion(outputs_sub, targets_sub) * fraction
        return loss


class SoftTreeSupLoss(TreeSupLoss):
    def __init__(self, *args, Rules=None, **kwargs):
        super().__init__(*args, Rules=SoftEmbeddedDecisionRules, **kwargs)

    def forward_tree(self, outputs, targets):
        self.assert_output_not_nbdt(outputs)
        return self.criterion(self.rules(outputs), targets)


class SoftTreeLoss(SoftTreeSupLoss):

    accepts_tree_start_epochs = True
    accepts_tree_update_every_epochs = True
    accepts_tree_update_end_epochs = True
    accepts_arch = True
    accepts_net = lambda net, **kwargs: net
    accepts_checkpoint_path = lambda checkpoint_path, **kwargs: checkpoint_path

    def __init__(
        self,
        *args,
        arch=None,
        checkpoint_path="./",
        net=None,
        tree_start_epochs=67,
        tree_update_every_epochs=10,
        tree_update_end_epochs=120,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.start_epochs = tree_start_epochs
        self.update_every_epochs = tree_update_every_epochs
        self.update_end_epochs = tree_update_end_epochs
        self.net = net
        self.arch = arch
        self.checkpoint_path = checkpoint_path

    def forward_tree(self, outputs, targets):
        if self.epochs < self.start_epochs:
            return self.criterion(outputs, targets)  # regular xent
        self.assert_output_not_nbdt(outputs)
        return self.criterion(self.rules(outputs), targets)

    def set_epoch(self, *args, **kwargs):
        super().set_epoch(*args, **kwargs)
        offset = self.epochs - self.start_epochs
        if (
            offset >= 0
            and offset % self.update_every_epochs == 0
            and self.epochs < self.update_end_epochs
        ):
            checkpoint_dir = self.checkpoint_path.replace(".pth", "")
            path_graph = os.path.join(checkpoint_dir, f"graph-epoch{self.epochs}.json")
            self.tree.update_from_model(
                self.net, self.arch, self.tree.dataset, path_graph=path_graph
            )


class SoftSegTreeSupLoss(SoftTreeSupLoss):
    def forward(self, outputs, targets):
        self.assert_output_not_nbdt(outputs)

        loss = self.criterion(outputs, targets)
        coerced_outputs = coerce_tensor(outputs)
        bayesian_outputs = self.rules(coerced_outputs)
        bayesian_outputs = uncoerce_tensor(bayesian_outputs, outputs.shape)
        loss += self.criterion(bayesian_outputs, targets) * self.tree_supervision_weight
        return loss
