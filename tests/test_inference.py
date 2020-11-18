"""Tests that models work inference-time"""

from nbdt.model import SoftNBDT, HardNBDT


def test_nbdt_soft_cifar10(input_cifar10, resnet18_cifar10):
    model_soft = SoftNBDT(
        dataset="CIFAR10", model=resnet18_cifar10, hierarchy="induced"
    )
    model_soft(input_cifar10)


def test_nbdt_soft_cifar100(input_cifar100, resnet18_cifar100):
    model_soft = SoftNBDT(
        dataset="CIFAR100", model=resnet18_cifar100, hierarchy="induced"
    )
    model_soft(input_cifar100)


def test_nbdt_soft_tinyimagenet200(input_tinyimagenet200, resnet18_tinyimagenet200):
    model_soft = SoftNBDT(
        dataset="TinyImagenet200", model=resnet18_tinyimagenet200, hierarchy="induced"
    )
    model_soft(input_tinyimagenet200)


def test_nbdt_hard_cifar10(input_cifar10, resnet18_cifar10):
    model_hard = HardNBDT(
        dataset="CIFAR10", model=resnet18_cifar10, hierarchy="induced"
    )
    model_hard(input_cifar10)


def test_nbdt_hard_cifar100(input_cifar100, resnet18_cifar100):
    model_hard = HardNBDT(
        dataset="CIFAR100", model=resnet18_cifar100, hierarchy="induced"
    )
    model_hard(input_cifar100)


def test_nbdt_hard_tinyimagenet200(input_tinyimagenet200, resnet18_tinyimagenet200):
    model_hard = HardNBDT(
        dataset="TinyImagenet200", model=resnet18_tinyimagenet200, hierarchy="induced"
    )
    model_hard(input_tinyimagenet200)
