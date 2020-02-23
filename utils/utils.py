'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numpy as np

import torch.nn as nn
import torch.nn.init as init
from pathlib import Path

# tree-generation consntants
METHODS = ('prune', 'build', 'random', 'image')
DATASETS = ('CIFAR10', 'CIFAR100', 'TinyImagenet200')
DATASET_TO_FOLDER_NAME = {
    'CIFAR10': 'CIFAR10',
    'CIFAR100': 'CIFAR100',
    'TinyImagenet200': 'tiny-imagenet-200'
}

# main script constants
CIFAR10NODE = 'CIFAR10Node'
CIFAR10PATHSANITY = 'CIFAR10PathSanity'

DEFAULT_CIFAR10_TREE = './data/CIFAR10/graph-wordnet.json'
DEFAULT_CIFAR10_WNIDS = './data/CIFAR10/wnids.txt'
DEFAULT_CIFAR100_TREE = './data/CIFAR100/graph-wordnet.json'
DEFAULT_CIFAR100_WNIDS = './data/CIFAR100/wnids.txt'
DEFAULT_TINYIMAGENET200_TREE = './data/tiny-imagenet-200/graph-wordnet.json'
DEFAULT_TINYIMAGENET200_WNIDS = './data/tiny-imagenet-200/wnids.txt'


class Colors:
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\x1b[36m'

    @classmethod
    def red(cls, *args):
        print(cls.RED + args[0], *args[1:], cls.ENDC)

    @classmethod
    def green(cls, *args):
        print(cls.GREEN + args[0], *args[1:], cls.ENDC)

    @classmethod
    def cyan(cls, *args):
        print(cls.CYAN + args[0], *args[1:], cls.ENDC)

    @classmethod
    def bold(cls, *args):
        print(cls.BOLD + args[0], *args[1:], cls.ENDC)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except Exception as e:
    print(e)
    term_width = 50

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def set_np_printoptions():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def get_fname(args):
    assert not (args.resume and args.backbone), (
        'Can only specify loading from backbone architecture or a previous '
        'checkpoint. Specifying both will result in overriding.'
    )
    if args.backbone:
        return args.backbone
    return generate_fname(args)


def generate_fname(dataset, model, path_graph, wnid=None, name='',
        trainset=None, include_labels=(), exclude_labels=(),
        include_classes=(), num_samples=0, **kwargs):
    fname = 'ckpt'
    fname += '-' + dataset
    fname += '-' + model
    if dataset == CIFAR10NODE:
        fname += '-' + wnid
    if name:
        fname += '-' + name
    if path_graph:
        path = Path(path_graph)
        fname += '-' + path.stem.replace('graph-', '', 1)
    else:
        fname += '-wordnet'  # WARNING: hard-coded
    if include_labels:
        labels = ",".join(map(str, include_labels))
        fname += f'-incl{labels}'
    if exclude_labels:
        labels = ",".join(map(str, exclude_labels))
        fname += f'-excl{labels}'
    if include_classes:
        labels = ",".join(map(str, include_classes))
        fname += f'-incc{labels}'
    if num_samples != 0 and num_samples is not None:
        fname += f'-samples{num_samples}'
    return fname
