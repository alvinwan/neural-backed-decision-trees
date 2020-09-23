from nbdt.utils import set_np_printoptions
from nbdt.model import (
    SoftEmbeddedDecisionRules as SoftRules,
    HardEmbeddedDecisionRules as HardRules
)
from torch.distributions import Categorical
import torch.nn.functional as F
from nbdt import metrics
import functools
import numpy as np
import os


__all__ = names = (
    'Noop', 'ConfusionMatrix', 'ConfusionMatrixJointNodes',
    'IgnoredSamples', 'HardEmbeddedDecisionRules', 'SoftEmbeddedDecisionRules',
    'EntropyStatistics')
keys = ('path_graph', 'path_wnids', 'classes', 'dataset', 'metric')


def add_arguments(parser):
    pass


def start_end_decorator(obj, name):
    start = getattr(obj, f'start_{name}', None)
    end = getattr(obj, f'end_{name}', None)
    assert start and end
    def decorator(f):
        @functools.wraps(f)
        def wrapper(epoch, *args, **kwargs):
            start(epoch)
            f(epoch, *args, **kwargs)
            end(epoch)
        return wrapper
    return decorator


class StartEndContext:

    def __init__(self, obj, name, epoch=0):
        self.obj = obj
        self.name = name
        self.epoch = epoch

    def __call__(self, epoch):
        self.epoch = epoch
        return self

    def __enter__(self):
        return getattr(self.obj, f'start_{self.name}')(self.epoch)

    def __exit__(self, type, value, traceback):
        getattr(self.obj, f'end_{self.name}')(self.epoch)


class Noop:

    accepts_classes = lambda trainset, **kwargs: trainset.classes

    def __init__(self, classes=()):
        set_np_printoptions()

        self.classes = classes
        self.num_classes = len(classes)
        self.epoch = None

    @property
    def epoch_function(self):
        return start_end_decorator(self, 'epoch')

    @property
    def train_function(self):
        return start_end_decorator(self, 'train')

    @property
    def test_function(self):
        return start_end_decorator(self, 'test')

    @property
    def epoch_context(self):
        return StartEndContext(self, 'epoch')

    def start_epoch(self, epoch):
        self.epoch = epoch

    def start_train(self, epoch):
        assert epoch == self.epoch

    def update_batch(self, outputs, targets, images):
        self._update_batch(outputs, targets)

    def _update_batch(self, outputs, targets):
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

    def _update_batch(self, outputs, targets):
        super()._update_batch(outputs, targets)
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

    def _update_batch(self, outputs, targets):
        super()._update_batch(outputs, targets)
        self.ignored += outputs[:,0].eq(-1).sum().item()
        return self.ignored

    def end_test(self, epoch):
        super().end_test(epoch)
        print("Ignored Samples: {}".format(self.ignored))


class DecisionRules(Noop):
    """Generic support for evaluating embedded decision rules."""

    accepts_dataset = lambda trainset, **kwargs: trainset.__class__.__name__
    accepts_path_graph = True
    accepts_path_wnids = True
    accepts_metric = True

    name = 'NBDT'

    def __init__(self, *args, Rules=HardRules, metric='top1', **kwargs):
        self.rules = Rules(*args, **kwargs)
        self.metric = getattr(metrics, metric)()

    def start_test(self, epoch):
        self.metric.clear()

    def _update_batch(self, outputs, targets):
        super()._update_batch(outputs, targets)
        outputs = self.rules.forward(outputs)
        self.metric.forward(outputs, targets)
        accuracy = round(self.metric.correct / float(self.metric.total), 4) * 100
        return accuracy

    def end_test(self, epoch):
        super().end_test(epoch)
        accuracy = round(self.metric.correct / self.metric.total * 100., 2)
        print(f'[{self.name}] Acc: {accuracy}%, {self.metric.correct}/{self.metric.total}')


class HardEmbeddedDecisionRules(DecisionRules):
    """Evaluation is hard."""

    name = 'NBDT-Hard'


class SoftEmbeddedDecisionRules(DecisionRules):
    """Evaluation is soft."""

    name = 'NBDT-Soft'

    def __init__(self, *args, Rules=None, **kwargs):
        super().__init__(*args, Rules=SoftRules, **kwargs)


class EntropyStatistics(Noop):

    def __init__(self, classes=(), k=20):
        super().__init__(classes)
        self.reset()
        self.k = k

    def start_test(self, epoch):
        super().start_test(epoch)
        self.reset()

    def reset(self):
        self.avg = 0
        self.std = 0
        self.max = []
        self.min = []
        self.i = 0

    @staticmethod
    def first(t):
        return t[0]

    def update_batch(self, outputs, targets, images):
        super().update_batch(outputs, targets, images)

        probs = F.softmax(outputs, dim=1)
        e = list(Categorical(probs=probs).entropy().cpu().detach().numpy())
        for e_i in e:
            self.i += 1
            avg_i_minus_1 = self.avg
            self.avg = avg_i_minus_1 + ((e_i - avg_i_minus_1) / self.i)
            self.std = self.std + (e_i - avg_i_minus_1) * (e_i - self.avg)

        e_images = list(zip(e, images))
        self.max = list(sorted(self.max + e_images, reverse=True, key=EntropyStatistics.first))[:self.k]
        self.min = list(sorted(self.min + e_images, key=EntropyStatistics.first))[:self.k]

    def end_test(self, epoch):
        super().end_test(epoch)
        _max = max(self.max, key=EntropyStatistics.first)[0]
        _min = min(self.min, key=EntropyStatistics.first)[0]
        print(f'[Entropy] avg {self.avg:.2e}, std {self.std:.2e}, max {_max:.2e}, min {_min:.2e}')
