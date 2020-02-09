import numpy as np


__all__ = names = ('Noop', 'ConfusionMatrix', )


class Noop:

    def __init__(self, trainset, testset):
        self.trainset = trainset
        self.testset = testset

        self.epoch = None

    def start_epoch(self, epoch):
        self.epoch = epoch

    def start_train(self, epoch):
        assert epoch == self.epoch

    def update_batch(self, predicted, targets):
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

    def update_batch(self, predicted, targets):
        super().update_batch(predicted, targets)
        if len(predicted.shape) == 1:
            predicted = predicted.numpy().ravel()
            targets = targets.numpy().ravel()
            self.update(predicted, targets)

    def end_test(self, epoch):
        super().end_test(epoch)
        # set_np_printoptions()
        recall = self.recall()
        for row, cls in zip(recall, self.trainset.classes):
            print(row, cls)
        print(recall.diagonal(), '(diagonal)')

    def update(self, preds, labels):
        preds = tuple(preds)
        labels = tuple(labels)

        for pred, label in zip(preds, labels):
            self.m[label, pred] += 1

    def normalize(self, axis):
        total = self.m.astype(np.float).sum(axis=axis)
        total = total[:, None] if axis == 1 else total[None]
        return self.m / total

    def recall(self):
        return self.normalize(1)

    def precision(self):
        return self.normalize(0)
