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

        self.nodes = Node.get_nodes(path_graph, path_wnids, classes)
        self.G = self.nodes[0].G
        self.wnid_to_node = {node.wnid: node for node in self.nodes}

        self.wnids = get_wnids(path_wnids)
        self.wnid_to_class = {wnid: cls for wnid, cls in zip(self.wnids, self.classes)}

        self.correct = 0
        self.total = 0

        self.I = torch.eye(len(classes))

    @staticmethod
    def get_node_logits(outputs, node):
        """Get output for a particular node

        This `outputs` above are the output of the neural network.
        """
        return torch.stack([
            outputs.T[node.new_to_old_classes[new_label]].mean(dim=0)
            for new_label in range(node.num_classes)
        ]).T

    @classmethod
    def get_all_node_outputs(cls, outputs, nodes):
        """Run hard embedded decision rules.

        Returns the output for *every single node.
        """
        wnid_to_outputs = {}
        for node in nodes:
            node_logits = cls.get_node_logits(outputs, node)
            wnid_to_outputs[node.wnid] = {
                'logits': node_logits,
                'preds': torch.max(node_logits, dim=1)[1],
                'probs': F.softmax(node_logits, dim=1)
            }
        return wnid_to_outputs

    def forward_nodes(self, outputs):
        return self.get_all_node_outputs(outputs, self.nodes)


class HardEmbeddedDecisionRules(EmbeddedDecisionRules):

    @classmethod
    def get_node_logits_filtered(cls, node, outputs, targets):
        """'Smarter' inference for a hard node.

        If you have targets for the node, you can selectively perform inference,
        only for nodes where the label of a sample is well-defined.
        """
        classes = [node.old_to_new_classes[int(t)] for t in targets]
        selector = [bool(cls) for cls in classes]
        targets_sub = [cls[0] for cls in classes if cls]

        outputs = outputs[selector]
        if outputs.size(0) == 0:
            return selector, outputs[:, :node.num_classes], targets_sub

        outputs_sub = cls.get_node_logits(outputs, node)
        return selector, outputs_sub, targets_sub

    @classmethod
    def traverse_tree(cls, wnid_to_outputs, nodes, wnid_to_class, classes):
        """Convert node outputs to final prediction.

        Note that the prediction output for this function can NOT be trained
        on. The outputs have been detached from the computation graph.
        """
        # move all to CPU, detach from computation graph
        example = wnid_to_outputs[nodes[0].wnid]
        n_samples = int(example['logits'].size(0))

        for wnid in tuple(wnid_to_outputs.keys()):
            outputs = wnid_to_outputs[wnid]
            outputs['preds'] = list(map(int, outputs['preds'].cpu()))
            outputs['probs'] = outputs['probs'].detach().cpu()

        wnid_to_node = {node.wnid: node for node in nodes}
        wnid_root = get_root(nodes[0].G)
        node_root = wnid_to_node[wnid_root]

        decisions = []
        preds = []
        for index in range(n_samples):
            decision = [{'node': node_root, 'name': 'root', 'prob': 1}]
            wnid, node = wnid_root, node_root
            while node is not None:
                if node.wnid not in wnid_to_outputs:
                    wnid = node = None
                    break
                outputs = wnid_to_outputs[node.wnid]
                index_child = outputs['preds'][index]
                prob_child = float(outputs['probs'][index][index_child])
                wnid = node.children[index_child]
                node = wnid_to_node.get(wnid, None)
                decision.append({'node': node, 'name': wnid_to_name(wnid), 'prob': prob_child})
            cls = wnid_to_class.get(wnid, None)
            pred = -1 if cls is None else classes.index(cls)
            preds.append(pred)
            decisions.append(decision)
        return torch.Tensor(preds).long(), decisions

    def predicted_to_logits(self, predicted):
        """Convert predicted classes to one-hot logits."""
        if self.I.device != predicted.device:
            self.I = self.I.to(predicted.device)
        return self.I[predicted]

    def forward_with_decisions(self, outputs):
        wnid_to_outputs = self.forward_nodes(outputs)
        predicted, decisions = self.traverse_tree(
            wnid_to_outputs, self.nodes, self.wnid_to_class, self.classes)
        logits = self.predicted_to_logits(predicted)
        logits._nbdt_output_flag = True  # checked in nbdt losses, to prevent mistakes
        return logits, decisions

    def forward(self, outputs):
        outputs, _ = self.forward_with_decisions(outputs)
        return outputs


class SoftEmbeddedDecisionRules(EmbeddedDecisionRules):

    @classmethod
    def traverse_tree(cls, wnid_to_outputs, nodes):
        """
        In theory, the loop over children below could be replaced with just a
        few lines:

            for index_child in range(len(node.children)):
                old_indexes = node.new_to_old_classes[index_child]
                class_probs[:,old_indexes] *= output[:,index_child][:,None]

        However, we collect all indices first, so that only one tensor operation
        is run. The output is a single distribution over all leaves. The
        ordering is determined by the original ordering of the provided logits.
        (I think. Need to check nbdt.data.custom.Node)
        """
        example = wnid_to_outputs[nodes[0].wnid]
        num_samples = example['logits'].size(0)
        num_classes = len(nodes[0].original_classes)
        device = example['logits'].device
        class_probs = torch.ones((num_samples, num_classes)).to(device)

        for node in nodes:
            outputs = wnid_to_outputs[node.wnid]

            old_indices, new_indices = [], []
            for index_child in range(len(node.children)):
                old = node.new_to_old_classes[index_child]
                old_indices.extend(old)
                new_indices.extend([index_child] * len(old))

            assert len(set(old_indices)) == len(old_indices), (
                'All old indices must be unique in order for this operation '
                'to be correct.'
            )
            class_probs[:,old_indices] *= outputs['probs'][:,new_indices]
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
        wnid_to_outputs = self.forward_nodes(outputs)
        logits = self.traverse_tree(wnid_to_outputs, self.nodes)
        logits._nbdt_output_flag = True  # checked in nbdt losses, to prevent mistakes
        return logits


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
            eval=True,
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

        if eval:
            self.eval()

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = coerce_state_dict(state_dict, self.model.state_dict())
        return self.model.load_state_dict(state_dict, **kwargs)

    def state_dict(self):
        return self.model.state_dict()

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
