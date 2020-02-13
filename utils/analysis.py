import numpy as np


__all__ = names = ('Noop', 'ConfusionMatrix', 'ConfusionMatrixJointNodes')


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


class ConfusionMatrixJointNodes(ConfusionMatrix):
    """Calculates confusion matrix for tree of joint nodes"""

    def __init__(self, trainset, testset):
        assert hasattr(trainset, 'nodes'), (
            'Dataset must be for joint nodes, in order to run joint-node '
            'specific confusion matrix analysis. You can run the regular '
            'confusion matrix analysis instead.'
        )
        self.nodes = trainset.nodes

    def start_test(self, epoch):
        self.ms = [
            np.zeros((node.num_classes, node.num_classes))
            for node in self.nodes
        ]

    def update_batch(self, predicted, targets):
        for m, pred, targ in zip(self.ms, predicted.T, targets.T):
            pred = pred.numpy().ravel()
            targ = targ.numpy().ravel()
            ConfusionMatrix.update(m, pred, targ)

    def end_test(self, epoch):
        mean_accs = []
        for m, node in zip(self.ms, self.nodes):
            class_accs = ConfusionMatrix.normalize(m, 0).diagonal()
            mean_acc = np.mean(class_accs)
            print(node.wnid, node.classes, mean_acc, class_accs)
            mean_accs.append(mean_acc)

        min_acc = min(mean_accs)
        min_node = self.nodes[mean_accs.index(min_acc)]
        print(f'Node ({min_node.wnid}) with lowest accuracy ({min(mean_accs)}%)'
              f' (sorted accuracies): {sorted(mean_accs)}')
