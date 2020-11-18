"""
For external use as part of nbdt package. This is a model that
runs inference as an NBDT. Note these make no assumption about the
underlying neural network other than it (1) is a classification model and
(2) returns logits.
"""

import torch.nn as nn
from nbdt.utils import (
    dataset_to_default_path_graph,
    dataset_to_default_path_wnids,
    hierarchy_to_path_graph,
    coerce_tensor,
    uncoerce_tensor,
)
from nbdt.models.utils import load_state_dict_from_key, coerce_state_dict
from nbdt.tree import Node, Tree
from nbdt.thirdparty.wn import get_wnids, synset_to_name
from nbdt.thirdparty.nx import get_root
from torch.distributions import Categorical

import torch
import torch.nn as nn
import torch.nn.functional as F


model_urls = {
    (
        "ResNet18",
        "CIFAR10",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR10-ResNet18-induced-ResNet18-SoftTreeSupLoss.pth",
    (
        "wrn28_10_cifar10",
        "CIFAR10",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR10-wrn28_10_cifar10-induced-wrn28_10_cifar10-SoftTreeSupLoss.pth",
    (
        "wrn28_10_cifar10",
        "CIFAR10",
        "wordnet",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR10-wrn28_10_cifar10-wordnet-SoftTreeSupLoss.pth",
    (
        "ResNet18",
        "CIFAR100",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR100-ResNet18-induced-ResNet18-SoftTreeSupLoss.pth",
    (
        "wrn28_10_cifar100",
        "CIFAR100",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR100-wrn28_10_cifar100-induced-wrn28_10_cifar100-SoftTreeSupLoss.pth",
    (
        "ResNet18",
        "TinyImagenet200",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-TinyImagenet200-ResNet18-induced-ResNet18-SoftTreeSupLoss-tsw10.0.pth",
    (
        "wrn28_10",
        "TinyImagenet200",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-TinyImagenet200-wrn28_10-induced-wrn28_10-SoftTreeSupLoss-tsw10.0.pth",
}


#########
# RULES #
#########


class EmbeddedDecisionRules(nn.Module):
    def __init__(
        self,
        dataset=None,
        path_graph=None,
        path_wnids=None,
        classes=(),
        hierarchy=None,
        tree=None,
    ):
        super().__init__()
        if not tree:
            tree = Tree(dataset, path_graph, path_wnids, classes, hierarchy=hierarchy)
        self.tree = tree
        self.correct = 0
        self.total = 0
        self.I = torch.eye(len(self.tree.classes))

    @staticmethod
    def get_node_logits(outputs, node=None, new_to_old_classes=None, num_classes=None):
        """Get output for a particular node

        This `outputs` above are the output of the neural network.
        """
        assert node or (
            new_to_old_classes and num_classes
        ), "Either pass node or (new_to_old_classes mapping and num_classes)"
        new_to_old_classes = new_to_old_classes or node.child_index_to_class_index
        num_classes = num_classes or node.num_classes
        return torch.stack(
            [
                outputs.T[new_to_old_classes[child_index]].mean(dim=0)
                for child_index in range(num_classes)
            ]
        ).T

    @classmethod
    def get_all_node_outputs(cls, outputs, nodes):
        """Run hard embedded decision rules.

        Returns the output for *every single node.
        """
        wnid_to_outputs = {}
        for node in nodes:
            node_logits = cls.get_node_logits(outputs, node)
            node_outputs = {"logits": node_logits}

            if len(node_logits.size()) > 1:
                node_outputs["preds"] = torch.max(node_logits, dim=1)[1]
                node_outputs["probs"] = F.softmax(node_logits, dim=1)
                node_outputs["entropy"] = Categorical(
                    probs=node_outputs["probs"]
                ).entropy()

            wnid_to_outputs[node.wnid] = node_outputs
        return wnid_to_outputs

    def forward_nodes(self, outputs):
        return self.get_all_node_outputs(outputs, self.tree.inodes)


class HardEmbeddedDecisionRules(EmbeddedDecisionRules):
    @classmethod
    def get_node_logits_filtered(cls, node, outputs, targets):
        """'Smarter' inference for a hard node.

        If you have targets for the node, you can selectively perform inference,
        only for nodes where the label of a sample is well-defined.
        """
        classes = [node.class_index_to_child_index[int(t)] for t in targets]
        selector = [bool(cls) for cls in classes]
        targets_sub = [cls[0] for cls in classes if cls]

        outputs = outputs[selector]
        if outputs.size(0) == 0:
            return selector, outputs[:, : node.num_classes], targets_sub

        outputs_sub = cls.get_node_logits(outputs, node)
        return selector, outputs_sub, targets_sub

    @classmethod
    def traverse_tree(cls, wnid_to_outputs, tree):
        """Convert node outputs to final prediction.

        Note that the prediction output for this function can NOT be trained
        on. The outputs have been detached from the computation graph.
        """
        # move all to CPU, detach from computation graph
        example = wnid_to_outputs[tree.inodes[0].wnid]
        n_samples = int(example["logits"].size(0))
        device = example["logits"].device

        for wnid in tuple(wnid_to_outputs.keys()):
            outputs = wnid_to_outputs[wnid]
            outputs["preds"] = list(map(int, outputs["preds"].cpu()))
            outputs["probs"] = outputs["probs"].detach().cpu()

        decisions = []
        preds = []
        for index in range(n_samples):
            decision = [{"node": tree.root, "name": "root", "prob": 1, "entropy": 0}]
            node = tree.root
            while not node.is_leaf():
                if node.wnid not in wnid_to_outputs:
                    node = None
                    break
                outputs = wnid_to_outputs[node.wnid]
                index_child = outputs["preds"][index]
                prob_child = float(outputs["probs"][index][index_child])
                node = node.children[index_child]
                decision.append(
                    {
                        "node": node,
                        "name": node.name,
                        "prob": prob_child,
                        "next_index": index_child,
                        "entropy": float(outputs["entropy"][index]),
                    }
                )
            preds.append(tree.wnid_to_class_index[node.wnid])
            decisions.append(decision)
        return torch.Tensor(preds).long().to(device), decisions

    def predicted_to_logits(self, predicted):
        """Convert predicted classes to one-hot logits."""
        if self.I.device != predicted.device:
            self.I = self.I.to(predicted.device)
        return self.I[predicted]

    def forward_with_decisions(self, outputs):
        wnid_to_outputs = self.forward_nodes(outputs)
        predicted, decisions = self.traverse_tree(wnid_to_outputs, self.tree)
        logits = self.predicted_to_logits(predicted)
        logits._nbdt_output_flag = True  # checked in nbdt losses, to prevent mistakes
        return logits, decisions

    def forward(self, outputs):
        outputs, _ = self.forward_with_decisions(outputs)
        return outputs


class SoftEmbeddedDecisionRules(EmbeddedDecisionRules):
    @classmethod
    def traverse_tree(cls, wnid_to_outputs, tree):
        """
        In theory, the loop over children below could be replaced with just a
        few lines:

            for index_child in range(len(node.children)):
                old_indexes = node.child_index_to_class_index[index_child]
                class_probs[:,old_indexes] *= output[:,index_child][:,None]

        However, we collect all indices first, so that only one tensor operation
        is run. The output is a single distribution over all leaves. The
        ordering is determined by the original ordering of the provided logits.
        (I think. Need to check nbdt.data.custom.Node)
        """
        example = wnid_to_outputs[tree.inodes[0].wnid]
        num_samples = example["logits"].size(0)
        num_classes = len(tree.classes)
        device = example["logits"].device
        class_probs = torch.ones((num_samples, num_classes)).to(device)

        for node in tree.inodes:
            outputs = wnid_to_outputs[node.wnid]

            old_indices, new_indices = [], []
            for index_child in range(len(node.children)):
                old = node.child_index_to_class_index[index_child]
                old_indices.extend(old)
                new_indices.extend([index_child] * len(old))

            assert len(set(old_indices)) == len(old_indices), (
                "All old indices must be unique in order for this operation "
                "to be correct."
            )
            class_probs[:, old_indices] *= outputs["probs"][:, new_indices]
        return class_probs

    def forward_with_decisions(self, outputs):
        wnid_to_outputs = self.forward_nodes(outputs)
        outputs = self.forward(outputs, wnid_to_outputs)
        _, predicted = outputs.max(1)

        decisions = []
        node = self.tree.inodes[0]
        leaf_to_steps = self.tree.get_leaf_to_steps()
        for index, prediction in enumerate(predicted):
            leaf = self.tree.wnids_leaves[prediction]
            steps = leaf_to_steps[leaf]
            probs = [1]
            entropies = [0]
            for step in steps[:-1]:
                _out = wnid_to_outputs[step["node"].wnid]
                _probs = _out["probs"][0]
                probs.append(_probs[step["next_index"]])
                entropies.append(Categorical(probs=_probs).entropy().item())
            for step, prob, entropy in zip(steps, probs, entropies):
                step["prob"] = float(prob)
                step["entropy"] = float(entropy)
            decisions.append(steps)
        return outputs, decisions

    def forward(self, outputs, wnid_to_outputs=None):
        if not wnid_to_outputs:
            wnid_to_outputs = self.forward_nodes(outputs)
        logits = self.traverse_tree(wnid_to_outputs, self.tree)
        logits._nbdt_output_flag = True  # checked in nbdt losses, to prevent mistakes
        return logits


##########
# MODELS #
##########


class NBDT(nn.Module):
    def __init__(
        self,
        dataset,
        model,
        arch=None,
        path_graph=None,
        path_wnids=None,
        classes=None,
        hierarchy=None,
        pretrained=None,
        **kwargs,
    ):
        super().__init__()

        if dataset and not hierarchy and not path_graph:
            assert arch, "Must specify `arch` if no `hierarchy` or `path_graph`"
            hierarchy = f"induced-{arch}"
        if pretrained and not arch:
            raise UserWarning(
                "To load a pretrained NBDT, you need to specify the `arch`. "
                "`arch` is the name of the architecture. e.g., ResNet18"
            )
        if isinstance(model, str):
            raise NotImplementedError("Model must be nn.Module")

        tree = Tree(dataset, path_graph, path_wnids, classes, hierarchy=hierarchy)
        self.init(
            dataset,
            model,
            tree,
            arch=arch,
            pretrained=pretrained,
            hierarchy=hierarchy,
            **kwargs,
        )

    def init(
        self,
        dataset,
        model,
        tree,
        arch=None,
        pretrained=False,
        hierarchy=None,
        eval=True,
        Rules=HardEmbeddedDecisionRules,
    ):
        """
        Extra init method makes clear which arguments are finally necessary for
        this class to function. The constructor for this class may generate
        some of these required arguments if initially missing.
        """
        self.rules = Rules(tree=tree)
        self.model = model

        if pretrained:
            assert arch is not None
            keys = [(arch, dataset), (arch, dataset, hierarchy)]
            state_dict = load_state_dict_from_key(keys, model_urls, pretrained=True)
            self.load_state_dict(state_dict)

        if eval:
            self.eval()

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = coerce_state_dict(state_dict, self.model.state_dict())
        return self.model.load_state_dict(state_dict, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def forward(self, x):
        x = self.model(x)
        x = self.rules(x)
        return x

    def forward_with_decisions(self, x):
        x = self.model(x)
        x, decisions = self.rules.forward_with_decisions(x)
        return x, decisions


class HardNBDT(NBDT):
    def __init__(self, *args, **kwargs):
        kwargs.update({"Rules": HardEmbeddedDecisionRules})
        super().__init__(*args, **kwargs)


class SoftNBDT(NBDT):
    def __init__(self, *args, **kwargs):
        kwargs.update({"Rules": SoftEmbeddedDecisionRules})
        super().__init__(*args, **kwargs)


class SegNBDT(NBDT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        assert len(x.shape) == 4, "Input must be of shape (N,C,H,W) for segmentation"
        x = self.model(x)
        original_shape = x.shape
        x = coerce_tensor(x)
        x = self.rules.forward(x)
        x = uncoerce_tensor(x, original_shape)
        return x


class HardSegNBDT(SegNBDT):
    def __init__(self, *args, **kwargs):
        kwargs.update({"Rules": HardEmbeddedDecisionRules})
        super().__init__(*args, **kwargs)


class SoftSegNBDT(SegNBDT):
    def __init__(self, *args, **kwargs):
        kwargs.update({"Rules": SoftEmbeddedDecisionRules})
        super().__init__(*args, **kwargs)
