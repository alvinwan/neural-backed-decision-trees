from nbdt.graph import get_root, get_wnids, synset_to_name, wnid_to_name
from nbdt.utils import (
    set_np_printoptions, dataset_to_default_path_graph,
    dataset_to_default_path_wnids
)
from nbdt.loss import HardTreeSupLoss, SoftTreeSupLoss
from nbdt.data.custom import Node, dataset_to_dummy_classes
import torch
import torch.nn as nn
import numpy as np
import csv


__all__ = names = (
    'Noop', 'ConfusionMatrix', 'ConfusionMatrixJointNodes',
    'IgnoredSamples', 'HardEmbeddedDecisionRules', 'SoftEmbeddedDecisionRules')
keys = ('path_graph', 'path_wnids', 'weighted_average', 'classes', 'dataset')


def add_arguments(parser):
    pass


class Noop:

    accepts_classes = lambda trainset, **kwargs: trainset.classes

    def __init__(self, classes=()):
        set_np_printoptions()

        self.classes = classes
        self.num_classes = len(classes)
        self.epoch = None

    def start_epoch(self, epoch):
        self.epoch = epoch

    def start_train(self, epoch):
        assert epoch == self.epoch

    def update_batch(self, outputs, targets):
        pass

    def end_train(self, epoch):
        assert epoch == self.epoch

    def start_test(self, epoch):
        assert epoch == self.epoch

    def end_test(self, epoch):
        assert epoch == self.epoch

    def end_epoch(self, epoch):
        assert epoch == self.epoch


class ConfusionMatrix(Noop):

    def __init__(self, classes):
        super().__init__(classes)
        self.k = len(classes)
        self.m = None

    def start_train(self, epoch):
        super().start_train(epoch)
        raise NotImplementedError()

    def start_test(self, epoch):
        super().start_test(epoch)
        self.m = np.zeros((self.k, self.k))

    def update_batch(self, outputs, targets):
        super().update_batch(outputs, targets)
        _, predicted = outputs.max(1)
        if len(predicted.shape) == 1:
            predicted = predicted.numpy().ravel()
            targets = targets.numpy().ravel()
            ConfusionMatrix.update(self.m, predicted, targets)

    def end_test(self, epoch):
        super().end_test(epoch)
        recall = self.recall()
        for row, cls in zip(recall, self.classes):
            print(row, cls)
        print(recall.diagonal(), '(diagonal)')

    @staticmethod
    def update(confusion_matrix, preds, labels):
        preds = tuple(preds)
        labels = tuple(labels)

        for pred, label in zip(preds, labels):
            confusion_matrix[label, pred] += 1

    @staticmethod
    def normalize(confusion_matrix, axis):
        total = confusion_matrix.astype(np.float).sum(axis=axis)
        total = total[:, None] if axis == 1 else total[None]
        return confusion_matrix / total

    def recall(self):
        return ConfusionMatrix.normalize(self.m, 1)

    def precision(self):
        return ConfusionMatrix.normalize(self.m, 0)


class IgnoredSamples(Noop):
    """ Counter for number of ignored samples in decision tree """

    def __init__(self, classes=()):
        super().__init__(classes)
        self.ignored = None

    def start_test(self, epoch):
        super().start_test(epoch)
        self.ignored = 0

    def update_batch(self, outputs, targets):
        super().update_batch(outputs, targets)
        self.ignored += outputs[:,0].eq(-1).sum().item()

    def end_test(self, epoch):
        super().end_test(epoch)
        print("Ignored Samples: {}".format(self.ignored))


class HardEmbeddedDecisionRules(Noop):
    """Evaluation is hard."""

    accepts_dataset = lambda trainset, **kwargs: trainset.__class__.__name__
    accepts_path_graph = True
    accepts_path_wnids = True
    accepts_weighted_average = True

    name = 'NBDT-Hard'

    def __init__(self,
            dataset,
            path_graph=None,
            path_wnids=None,
            classes=(),
            weighted_average=False):

        if not path_graph:
            path_graph = dataset_to_default_path_graph(dataset)
        if not path_wnids:
            path_wnids = dataset_to_default_path_wnids(dataset)
        if not classes:
            classes = dataset_to_dummy_classes(dataset)
        super().__init__(classes)
        assert all([dataset, path_graph, path_wnids, classes])

        self.nodes = Node.get_nodes(path_graph, path_wnids, classes)
        self.G = self.nodes[0].G
        self.wnid_to_node = {node.wnid: node for node in self.nodes}

        self.wnids = get_wnids(path_wnids)
        self.wnid_to_class = {wnid: cls for wnid, cls in zip(self.wnids, self.classes)}

        self.weighted_average = weighted_average
        self.correct = 0
        self.total = 0

        self.I = torch.eye(len(classes))

    def forward_with_decisions(self, outputs):
        wnid_to_pred_selector = {}
        for node in self.nodes:
            selector, outputs_sub, _ = HardTreeSupLoss.inference(
                node, outputs, (), self.weighted_average)
            if not any(selector):
                continue
            _, preds_sub = torch.max(outputs_sub, dim=1)
            preds_sub = list(map(int, preds_sub.cpu()))
            wnid_to_pred_selector[node.wnid] = (preds_sub, selector)

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

    def update_batch(self, outputs, targets):
        super().update_batch(outputs, targets)
        predicted = self.forward(outputs).max(1)[1].to(targets.device)

        n_samples = outputs.size(0)
        self.total += n_samples
        self.correct += (predicted == targets).sum().item()
        accuracy = round(self.correct / float(self.total), 4) * 100
        return f'{self.name}: {accuracy}%'

    def traverse_tree(self, _, wnid_to_pred_selector, n_samples):
        wnid_root = get_root(self.G)
        node_root = self.wnid_to_node[wnid_root]
        decisions = []
        preds = []
        for index in range(n_samples):
            decision = [{'node': node_root, 'name': 'root'}]
            wnid, node = wnid_root, node_root
            while node is not None:
                if node.wnid not in wnid_to_pred_selector:
                    wnid = node = None
                    break
                pred_sub, selector = wnid_to_pred_selector[node.wnid]
                if not selector[index]:  # we took a wrong turn. wrong.
                    wnid = node = None
                    break
                index_new = sum(selector[:index + 1]) - 1
                index_child = pred_sub[index_new]
                wnid = node.children[index_child]
                node = self.wnid_to_node.get(wnid, None)
                decision.append({'node': node, 'name': wnid_to_name(wnid)})
            cls = self.wnid_to_class.get(wnid, None)
            pred = -1 if cls is None else self.classes.index(cls)
            preds.append(pred)
            decisions.append(decision)
        return torch.Tensor(preds).long(), decisions

    def end_test(self, epoch):
        super().end_test(epoch)
        accuracy = round(self.correct / self.total * 100., 2)
        print(f'{self.name} Accuracy: {accuracy}%, {self.correct}/{self.total}')


class SoftEmbeddedDecisionRules(HardEmbeddedDecisionRules):
    """Evaluation is soft."""

    name = 'NBDT-Soft'

    def forward_with_decisions(self, outputs):
        predicted = self.forward(outputs)

        decisions = []
        node = self.nodes[0]
        leaf_to_path_nodes = Node.get_leaf_to_path(self.nodes)
        for index, prediction in enumerate(predicted):
            leaf = node.wnids[prediction]
            decision = leaf_to_path_nodes[leaf]
            decisions.append(decision)
        return predicted, decisions

    def forward(self, outputs):
        outputs = SoftTreeSupLoss.inference(
            self.nodes, outputs, self.num_classes, self.weighted_average)
        outputs._nbdt_output_flag = True  # checked in nbdt losses, to prevent mistakes
        return outputs
