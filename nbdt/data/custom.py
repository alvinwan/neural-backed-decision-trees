import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from nbdt.utils import DATASET_TO_NUM_CLASSES, DATASETS
from collections import defaultdict
from nbdt.thirdparty.wn import get_wnids, FakeSynset, wnid_to_synset, wnid_to_name
from nbdt.thirdparty.nx import get_leaves, get_leaf_to_path, read_graph
from nbdt.tree import Tree
from nbdt.utils import (
    dataset_to_default_path_graph,
    dataset_to_default_path_wnids,
    hierarchy_to_path_graph,
)
from . import imagenet
from . import cifar
import torch.nn as nn
import random


__all__ = names = (
    "CIFAR10IncludeLabels",
    "CIFAR100IncludeLabels",
    "TinyImagenet200IncludeLabels",
    "Imagenet1000IncludeLabels",
    "CIFAR10ExcludeLabels",
    "CIFAR100ExcludeLabels",
    "TinyImagenet200ExcludeLabels",
    "Imagenet1000ExcludeLabels",
    "CIFAR10ResampleLabels",
    "CIFAR100ResampleLabels",
    "TinyImagenet200ResampleLabels",
    "Imagenet1000ResampleLabels",
)


def add_arguments(parser):
    parser.add_argument("--probability-labels", nargs="*", type=float)
    parser.add_argument("--include-labels", nargs="*", type=int)
    parser.add_argument("--exclude-labels", nargs="*", type=int)
    parser.add_argument("--include-classes", nargs="*", type=int)


class ResampleLabelsDataset(Dataset):
    """
    Dataset that includes only the labels provided, with a limited number of
    samples. Note that labels are integers in [0, k) for a k-class dataset.

    :drop_classes bool: Modifies the dataset so that it is only a m-way
                        classification where m of k classes are kept. Otherwise,
                        the problem is still k-way.
    """

    accepts_probability_labels = True

    def __init__(self, dataset, probability_labels=1, drop_classes=False, seed=0):
        self.dataset = dataset
        self.classes = dataset.classes
        self.labels = list(range(len(self.classes)))
        self.probability_labels = self.get_probability_labels(
            dataset, probability_labels
        )

        self.drop_classes = drop_classes
        if self.drop_classes:
            self.classes, self.labels = self.get_classes_after_drop(
                dataset, probability_labels
            )

        assert self.labels, "No labels are included in `include_labels`"

        self.new_to_old = self.build_index_mapping(seed=seed)

    def get_probability_labels(self, dataset, ps):
        if not isinstance(ps, (tuple, list)):
            return [ps] * len(dataset.classes)
        if len(ps) == 1:
            return ps * len(dataset.classes)
        assert len(ps) == len(dataset.classes), (
            f"Length of probabilities vector {len(ps)} must equal that of the "
            f"dataset classes {len(dataset.classes)}."
        )
        return ps

    def apply_drop(self, dataset, ps):
        classes = [cls for p, cls in zip(ps, dataset.classes) if p > 0]
        labels = [i for p, i in zip(ps, range(len(dataset.classes))) if p > 0]
        return classes, labels

    def build_index_mapping(self, seed=0):
        """Iterates over all samples in dataset.

        Remaps all to-be-included samples to [0, n) where n is the number of
        samples with a class in the whitelist.

        Additionally, the outputted list is truncated to match the number of
        desired samples.
        """
        random.seed(seed)

        new_to_old = []
        for old, (_, label) in enumerate(self.dataset):
            if random.random() < self.probability_labels[label]:
                new_to_old.append(old)
        return new_to_old

    def __getitem__(self, index_new):
        index_old = self.new_to_old[index_new]
        sample, label_old = self.dataset[index_old]

        label_new = label_old
        if self.drop_classes:
            label_new = self.include_labels.index(label_old)

        return sample, label_new

    def __len__(self):
        return len(self.new_to_old)


class IncludeLabelsDataset(ResampleLabelsDataset):

    accepts_include_labels = True
    accepts_probability_labels = False

    def __init__(self, dataset, include_labels=(0,)):
        super().__init__(
            dataset,
            probability_labels=[
                int(cls in include_labels) for cls in range(len(dataset.classes))
            ],
        )


def get_resample_labels_dataset(dataset):
    class Cls(ResampleLabelsDataset):
        def __init__(self, *args, root="./data", probability_labels=1, **kwargs):
            super().__init__(
                dataset=dataset(*args, root=root, **kwargs),
                probability_labels=probability_labels,
            )

    Cls.__name__ = f"{dataset.__class__.__name__}ResampleLabels"
    return Cls


CIFAR10ResampleLabels = get_resample_labels_dataset(cifar.CIFAR10)
CIFAR100ResampleLabels = get_resample_labels_dataset(cifar.CIFAR100)
TinyImagenet200ResampleLabels = get_resample_labels_dataset(imagenet.TinyImagenet200)
Imagenet1000ResampleLabels = get_resample_labels_dataset(imagenet.Imagenet1000)


class IncludeClassesDataset(IncludeLabelsDataset):
    """
    Dataset that includes only the labels provided, with a limited number of
    samples. Note that classes are strings, like 'cat' or 'dog'.
    """

    accepts_include_labels = False
    accepts_include_classes = True

    def __init__(self, dataset, include_classes=()):
        super().__init__(
            dataset,
            include_labels=[dataset.classes.index(cls) for cls in include_classes],
        )


def get_include_labels_dataset(dataset):
    class Cls(IncludeLabelsDataset):
        def __init__(self, *args, root="./data", include_labels=(0,), **kwargs):
            super().__init__(
                dataset=dataset(*args, root=root, **kwargs),
                include_labels=include_labels,
            )

    Cls.__name__ = f"{dataset.__class__.__name__}IncludeLabels"
    return Cls


CIFAR10IncludeLabels = get_include_labels_dataset(cifar.CIFAR10)
CIFAR100IncludeLabels = get_include_labels_dataset(cifar.CIFAR100)
TinyImagenet200IncludeLabels = get_include_labels_dataset(imagenet.TinyImagenet200)
Imagenet1000IncludeLabels = get_include_labels_dataset(imagenet.Imagenet1000)


class ExcludeLabelsDataset(IncludeLabelsDataset):

    accepts_include_labels = False
    accepts_exclude_labels = True

    def __init__(self, dataset, exclude_labels=(0,)):
        k = len(dataset.classes)
        include_labels = set(range(k)) - set(exclude_labels)
        super().__init__(dataset=dataset, include_labels=include_labels)


def get_exclude_labels_dataset(dataset):
    class Cls(ExcludeLabelsDataset):
        def __init__(self, *args, root="./data", exclude_labels=(0,), **kwargs):
            super().__init__(
                dataset=dataset(*args, root=root, **kwargs),
                exclude_labels=exclude_labels,
            )

    Cls.__name__ = f"{dataset.__class__.__name__}ExcludeLabels"
    return Cls


CIFAR10ExcludeLabels = get_exclude_labels_dataset(cifar.CIFAR10)
CIFAR100ExcludeLabels = get_exclude_labels_dataset(cifar.CIFAR100)
TinyImagenet200ExcludeLabels = get_exclude_labels_dataset(imagenet.TinyImagenet200)
Imagenet1000ExcludeLabels = get_exclude_labels_dataset(imagenet.Imagenet1000)
