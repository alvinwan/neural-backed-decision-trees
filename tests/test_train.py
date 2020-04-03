"""Tests that train utilities work as advertised"""

import torch
import torch.nn as nn
from nbdt.loss import SoftTreeSupLoss


def test_criterion_cifar10(criterion):
    criterion = SoftTreeSupLoss(dataset='CIFAR10', criterion=criterion, hierarchy='induced')
    criterion(torch.randn((1, 10)), torch.randint(10, (1,)))


def test_criterion_cifar100(criterion):
    criterion = SoftTreeSupLoss(dataset='CIFAR100', criterion=criterion, hierarchy='induced')
    criterion(torch.randn((1, 100)), torch.randint(100, (1,)))


def test_criterion_tinyimagenet200(criterion):
    criterion = SoftTreeSupLoss(dataset='TinyImagenet200', criterion=criterion, hierarchy='induced')
    criterion(torch.randn((1, 200)), torch.randint(200, (1,)))
