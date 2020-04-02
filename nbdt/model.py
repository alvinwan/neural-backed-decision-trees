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
    hierarchy_to_path_graph)
from nbdt.models.utils import load_state_dict_from_key, coerce_state_dict
from nbdt.data.custom import Node, dataset_to_dummy_classes
from nbdt.graph import get_root, get_wnids, synset_to_name, wnid_to_name

import torch
import torch.nn as nn
import torch.nn.functional as F


model_urls = {
    ('ResNet18', 'CIFAR10'): 'https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR10-ResNet18-induced-ResNet18-SoftTreeSupLoss.pth',
    ('wrn28_10_cifar10', 'CIFAR10'): 'https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR10-wrn28_10_cifar10-induced-wrn28_10_cifar10-SoftTreeSupLoss.pth',
    ('wrn28_10_cifar10', 'CIFAR10', 'wordnet'): 'https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR10-wrn28_10_cifar10-wordnet-SoftTreeSupLoss.pth',
    ('ResNet18', 'CIFAR100'): 'https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR100-ResNet18-induced-ResNet18-SoftTreeSupLoss.pth',
    ('wrn28_10_cifar100', 'CIFAR100'): 'https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR100-wrn28_10_cifar100-induced-wrn28_10_cifar100-SoftTreeSupLoss.pth',
    ('ResNet18', 'TinyImagenet200'): 'https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-TinyImagenet200-ResNet18-induced-ResNet18-SoftTreeSupLoss-tsw10.0.pth',
    ('wrn28_10', 'TinyImagenet200'): 'https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-TinyImagenet200-wrn28_10-induced-wrn28_10-SoftTreeSupLoss-tsw10.0.pth',
}


#########
# RULES #
#########


class EmbeddedDecisionRules(nn.Module):

    def __init__(self,
            dataset,
            path_graph=None,
            path_wnids=None,
            classes=()):

        if not path_graph:
            path_graph = dataset_to_default_path_graph(dataset)
        if not path_wnids:
            path_wnids = dataset_to_default_path_wnids(dataset)
        if not classes:
            classes = dataset_to_dummy_classes(dataset)
        super().__init__()
        assert all([dataset, path_graph, path_wnids, classes])

        self.classes = classes
        self.num_classes = len(classes)

        self.nodes = Node.get_nodes(path_graph, path_wnids, classes)
        self.G = self.nodes[0].G
        self.wnid_to_node = {node.wnid: node for node in self.nodes}

        self.wnids = get_wnids(path_wnids)
        self.wnid_to_class = {wnid: cls for wnid, cls in zip(self.wnids, self.classes)}

        self.correct = 0
        self.total = 0

        self.I = torch.eye(len(classes))

    @staticmethod
    def get_output_sub(_outputs, node):
        return torch.stack([
            _outputs.T[node.new_to_old_classes[new_label]].mean(dim=0)
            for new_label in zip(range(node.num_classes))
        ]).T


class HardEmbeddedDecisionRules(EmbeddedDecisionRules):

    @classmethod
    def inference(cls, node, outputs, targets):
        _outputs = outputs
        targets_sub = targets
        selector = [True] * outputs.size(0)

        if targets:
            classes = [node.old_to_new_classes[int(t)] for t in targets]
            selector = [bool(cls) for cls in classes]
            targets_sub = [cls[0] for cls in classes if cls] if targets else None

            _outputs = outputs[selector]
            if _outputs.size(0) == 0:
                return selector, _outputs[:, :node.num_classes], targets_sub

        outputs_sub = cls.get_output_sub(_outputs, node)
        return selector, outputs_sub, targets_sub

    def forward_with_decisions(self, outputs):
        wnid_to_pred_selector = {}
        for node in self.nodes:
            selector, outputs_sub, _ = self.inference(node, outputs, ())
            if not any(selector):
                continue
            _, preds_sub = torch.max(outputs_sub, dim=1)
            preds_sub = list(map(int, preds_sub.cpu()))
            probs_sub = F.softmax(outputs_sub, dim=1).detach().cpu()
            wnid_to_pred_selector[node.wnid] = (preds_sub, probs_sub, selector)

        _, predicted = outputs.max(1)

        n_samples = outputs.size(0)
        n_classes = outputs.size(1)
        predicted, decisions = self.traverse_tree(
            predicted, wnid_to_pred_selector, n_samples)

        if self.I.device != outputs.device:
            self.I = self.I.to(outputs.device)

        outputs = self.I[predicted]
        outputs._nbdt_output_flag = True  # checked in nbdt losses, to prevent mistakes
        return outputs, decisions

    def forward(self, outputs):
        outputs, _ = self.forward_with_decisions(outputs)
        return outputs

    def traverse_tree(self, _, wnid_to_pred_selector, n_samples):
        wnid_root = get_root(self.G)
        node_root = self.wnid_to_node[wnid_root]
        decisions = []
        preds = []
        for index in range(n_samples):
            decision = [{'node': node_root, 'name': 'root', 'prob': 1}]
            wnid, node = wnid_root, node_root
            while node is not None:
                if node.wnid not in wnid_to_pred_selector:
                    wnid = node = None
                    break
                pred_sub, prob_sub, selector = wnid_to_pred_selector[node.wnid]
                if not selector[index]:  # we took a wrong turn. wrong.
                    wnid = node = None
                    break
                index_new = sum(selector[:index + 1]) - 1
                index_child = pred_sub[index_new]
                prob_child = float(prob_sub[index_new][index_child])
                wnid = node.children[index_child]
                node = self.wnid_to_node.get(wnid, None)
                decision.append({'node': node, 'name': wnid_to_name(wnid), 'prob': prob_child})
            cls = self.wnid_to_class.get(wnid, None)
            pred = -1 if cls is None else self.classes.index(cls)
            preds.append(pred)
            decisions.append(decision)
        return torch.Tensor(preds).long(), decisions


class SoftEmbeddedDecisionRules(EmbeddedDecisionRules):

    @classmethod
    def inference(cls, nodes, outputs, num_classes):
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
            output = cls.get_output_sub(outputs, node)
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

    def forward_with_decisions(self, outputs):
        outputs = self.forward(outputs)
        _, predicted = outputs.max(1)

        decisions = []
        node = self.nodes[0]
        leaf_to_path_nodes = Node.get_leaf_to_path(self.nodes)
        for index, prediction in enumerate(predicted):
            leaf = node.wnids[prediction]
            decision = leaf_to_path_nodes[leaf]
            for justification in decision:
                justification['prob'] = -1  # TODO(alvin): fill in prob
            decisions.append(decision)
        return outputs, decisions

    def forward(self, outputs):
        outputs = self.inference(self.nodes, outputs, self.num_classes)
        outputs._nbdt_output_flag = True  # checked in nbdt losses, to prevent mistakes
        return outputs


##########
# MODELS #
##########


class NBDT(nn.Module):

    def __init__(self,
            dataset,
            model,
            arch=None,
            path_graph=None,
            path_wnids=None,
            classes=None,
            hierarchy=None,
            pretrained=None,
            **kwargs):
        super().__init__()

        if dataset and not hierarchy and not path_graph:
            assert arch, 'Must specify `arch` if no `hierarchy` or `path_graph`'
            hierarchy = f'induced-{arch}'
        if dataset and hierarchy and not path_graph:
            path_graph = hierarchy_to_path_graph(dataset, hierarchy)
        if dataset and not path_graph:
            path_graph = dataset_to_default_path_graph(dataset)
        if dataset and not path_wnids:
            path_wnids = dataset_to_default_path_wnids(dataset)
        if dataset and not classes:
            classes = dataset_to_dummy_classes(dataset)
        if pretrained and not arch:
            raise UserWarning(
                'To load a pretrained NBDT, you need to specify the `arch`. '
                '`arch` is the name of the architecture. e.g., ResNet18')
        if isinstance(model, str):
            raise NotImplementedError('Model must be nn.Module')

        self.init(dataset, model, path_graph, path_wnids, classes,
            arch=arch, pretrained=pretrained, hierarchy=hierarchy, **kwargs)

    def init(self,
            dataset,
            model,
            path_graph,
            path_wnids,
            classes,
            arch=None,
            pretrained=False,
            hierarchy=None,
            Rules=HardEmbeddedDecisionRules):
        """
        Extra init method makes clear which arguments are finally necessary for
        this class to function. The constructor for this class may generate
        some of these required arguments if initially missing.
        """
        self.rules = Rules(dataset, path_graph, path_wnids, classes)
        self.model = model

        if pretrained:
            assert arch is not None
            keys = [(arch, dataset), (arch, dataset, hierarchy)]
            state_dict = load_state_dict_from_key(
                keys, model_urls, pretrained=True)
            self.load_state_dict(state_dict)
        self.eval()

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = coerce_state_dict(state_dict, self.model.state_dict())
        return self.model.load_state_dict(state_dict, **kwargs)

    def state_dict(self):
        return self.model.state_dict()

    def forward(self, x):
        x = self.model(x)
        x = self.rules.forward(x)
        return x

    def forward_with_decisions(self, x):
        x = self.model(x)
        x, decisions = self.rules.forward_with_decisions(x)
        return x, decisions


class HardNBDT(NBDT):

    def __init__(self, *args, **kwargs):
        kwargs.update({
            'Rules': HardEmbeddedDecisionRules
        })
        super().__init__(*args, **kwargs)


class SoftNBDT(NBDT):

    def __init__(self, *args, **kwargs):
        kwargs.update({
            'Rules': SoftEmbeddedDecisionRules
        })
        super().__init__(*args, **kwargs)
