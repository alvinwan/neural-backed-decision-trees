import pytest
import torch
import torch.nn as nn
from nbdt.models import ResNet18


collect_ignore = ["setup.py", "main.py"]


@pytest.fixture
def label_cifar10():
    return torch.randint(10, (1,))


@pytest.fixture
def input_cifar10():
    return torch.randn(1, 3, 32, 32)


@pytest.fixture
def input_cifar100():
    return torch.randn(1, 3, 32, 32)


@pytest.fixture
def input_tinyimagenet200():
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


@pytest.fixture
def resnet18_cifar10():
    return ResNet18(num_classes=10)


@pytest.fixture
def resnet18_cifar100():
    return ResNet18(num_classes=100)


@pytest.fixture
def resnet18_tinyimagenet200():
    return ResNet18(num_classes=200)
