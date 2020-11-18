"""Tests that train utilities work as advertised"""

import torch
import torch.nn as nn
from nbdt.loss import SoftTreeSupLoss, HardTreeSupLoss
from nbdt.model import HardNBDT


def test_criterion_cifar10(criterion, label_cifar10):
    criterion = SoftTreeSupLoss(
        dataset="CIFAR10", criterion=criterion, hierarchy="induced"
    )
    criterion(torch.randn((1, 10)), label_cifar10)


def test_criterion_cifar100(criterion):
    criterion = SoftTreeSupLoss(
        dataset="CIFAR100", criterion=criterion, hierarchy="induced"
    )
    criterion(torch.randn((1, 100)), torch.randint(100, (1,)))


def test_criterion_tinyimagenet200(criterion):
    criterion = SoftTreeSupLoss(
        dataset="TinyImagenet200", criterion=criterion, hierarchy="induced"
    )
    criterion(torch.randn((1, 200)), torch.randint(200, (1,)))


def test_nbdt_gradient_hard(resnet18_cifar10, input_cifar10, label_cifar10, criterion):
    output_cifar10 = resnet18_cifar10(input_cifar10)
    assert output_cifar10.requires_grad

    criterion = HardTreeSupLoss(
        dataset="CIFAR10", criterion=criterion, hierarchy="induced"
    )
    loss = criterion(output_cifar10, label_cifar10)
    loss.backward()


def test_nbdt_gradient_soft(resnet18_cifar10, input_cifar10, label_cifar10, criterion):
    output_cifar10 = resnet18_cifar10(input_cifar10)
    assert output_cifar10.requires_grad

    criterion = SoftTreeSupLoss(
        dataset="CIFAR10", criterion=criterion, hierarchy="induced"
    )
    loss = criterion(output_cifar10, label_cifar10)
    loss.backward()
