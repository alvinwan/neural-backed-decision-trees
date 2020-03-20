from nbdt.graph import get_root, get_wnids
from nbdt.utils import (
    set_np_printoptions, dataset_to_default_path_graph,
    dataset_to_default_path_wnids
)
from nbdt.loss import HardTreeSupLoss, SoftTreeSupLoss
from nbdt.data.custom import Node
import torch
import torch.nn as nn
import numpy as np
import csv


__all__ = names = (
    'Noop', 'ConfusionMatrix', 'ConfusionMatrixJointNodes',
    'IgnoredSamples', 'HardEmbeddedDecisionRules', 'SoftEmbeddedDecisionRules')
keys = ('path_graph', 'path_wnids', 'weighted_average', 'trainset', 'testset')


def add_arguments(parser):
    pass


class Noop:

    accepts_trainset = lambda trainset, **kwargs: trainset
    accepts_testset = lambda testset, **kwargs: testset

    def __init__(self, trainset, testset):
        set_np_printoptions()

        self.trainset = trainset
        self.testset = testset

        self.epoch = None

    def start_epoch(self, epoch):
        self.epoch = epoch

    def start_train(self, epoch):
        assert epoch == self.epoch

    def update_batch(self, outputs, predicted, targets):
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

    def __init__(self, trainset, testset):
        super().__init__(trainset, testset)
        self.k = len(trainset.classes)
        self.m = None

    def start_train(self, epoch):
        super().start_train(epoch)
        raise NotImplementedError()

    def start_test(self, epoch):
        super().start_test(epoch)
        self.m = np.zeros((self.k, self.k))

    def update_batch(self, outputs, predicted, targets):
        super().update_batch(outputs, predicted, targets)
        if len(predicted.shape) == 1:
            predicted = predicted.numpy().ravel()
            targets = targets.numpy().ravel()
            ConfusionMatrix.update(self.m, predicted, targets)

    def end_test(self, epoch):
        super().end_test(epoch)
        recall = self.recall()
        for row, cls in zip(recall, self.trainset.classes):
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

    def __init__(self, trainset, testset):
        super().__init__(trainset, testset)
        self.ignored = None

    def start_test(self, epoch):
        super().start_test(epoch)
        self.ignored = 0

    def update_batch(self, outputs, predicted, targets):
        super().update_batch(outputs, predicted, targets)
        self.ignored += outputs[:,0].eq(-1).sum().item()

    def end_test(self, epoch):
        super().end_test(epoch)
        print("Ignored Samples: {}".format(self.ignored))


class HardEmbeddedDecisionRules(Noop):
    """Evaluation is hard."""

    accepts_path_graph = True
    accepts_path_wnids = True
    accepts_weighted_average = True

    def __init__(self, trainset, testset, path_graph, path_wnids,
            weighted_average=False):
        super().__init__(trainset, testset)
        self.nodes = Node.get_nodes(path_graph, path_wnids, trainset.classes)
        self.G = self.nodes[0].G
        self.wnid_to_node = {node.wnid: node for node in self.nodes}

        self.wnids = get_wnids(path_wnids)
        self.classes = trainset.classes
        self.wnid_to_class = {wnid: cls for wnid, cls in zip(self.wnids, self.classes)}

        self.weighted_average = weighted_average
        self.correct = 0
        self.total = 0

    @classmethod
    def from_dataset(cls, trainset, testset, dataset, **kwargs):
        assert 'path_graph' not in kwargs and 'path_wnids' not in kwargs, \
            '`from_dataset` sets both the path_graph and path_wnids'
        path_graph = dataset_to_default_path_graph(dataset)
        path_wnids = dataset_to_default_path_wnids(dataset)
        return cls(trainset, testset, path_graph, path_wnids, **kwargs)

    def update_batch(self, outputs, predicted, targets):
        super().update_batch(outputs, predicted, targets)

        targets_ints = [int(target) for target in targets.cpu().long()]
        wnid_to_pred_selector = {}
        for node in self.nodes:
            selector, outputs_sub, targets_sub = HardTreeSupLoss.inference(
                node, outputs, targets, self.weighted_average)
            if not any(selector):
                continue
            _, preds_sub = torch.max(outputs_sub, dim=1)
            preds_sub = list(map(int, preds_sub.cpu()))
            wnid_to_pred_selector[node.wnid] = (preds_sub, selector)

        n_samples = outputs.size(0)
        predicted = self.traverse_tree(
            predicted, wnid_to_pred_selector, n_samples).to(targets.device)
        self.total += n_samples
        self.correct += (predicted == targets).sum().item()
        accuracy = round(self.correct / float(self.total), 4) * 100
        return f'NBDT-Hard: {accuracy}%'

    def traverse_tree(self, _, wnid_to_pred_selector, n_samples):
        wnid_root = get_root(self.G)
        node_root = self.wnid_to_node[wnid_root]
        preds = []
        for index in range(n_samples):
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
            cls = self.wnid_to_class.get(wnid, None)
            pred = -1 if cls is None else self.classes.index(cls)
            preds.append(pred)
        return torch.Tensor(preds).long()

    def end_test(self, epoch):
        super().end_test(epoch)
        accuracy = round(self.correct / self.total * 100., 2)
        print(f'NBDT-Hard Accuracy: {accuracy}%, {self.correct}/{self.total}')


class SoftEmbeddedDecisionRules(HardEmbeddedDecisionRules):
    """Evaluation is soft."""

    def __init__(self, trainset, testset, path_graph, path_wnids,
            weighted_average=False):
        super().__init__(trainset, testset, path_graph, path_wnids)
        self.num_classes = len(trainset.classes)

    def update_batch(self, outputs, predicted, targets):
        bayesian_outputs = SoftTreeSupLoss.inference(
            self.nodes, outputs, self.num_classes, self.weighted_average)
        n_samples = outputs.size(0)
        predicted = bayesian_outputs.max(1)[1].to(targets.device)
        self.total += n_samples
        self.correct += (predicted == targets).sum().item()
        accuracy = round(self.correct / float(self.total), 4) * 100
        return f'NBDT-Soft: {accuracy}%'

    def end_test(self, epoch):
        accuracy = round(self.correct / self.total * 100., 2)
        print(f'NBDT-Soft Accuracy: {accuracy}%, {self.correct}/{self.total}')
