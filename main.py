'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import data, analysis

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

import models
from utils.utils import Colors
from utils.utils import (
    progress_bar, generate_fname, CIFAR10NODE, CIFAR10PATHSANITY,
    set_np_printoptions
)


set_np_printoptions()
datasets = ('CIFAR10', 'CIFAR100') + data.imagenet.names + data.custom.names


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--name', default='',
                    help='Name of experiment. Used for checkpoint filename')
parser.add_argument('--batch-size', default=512, type=int,
                    help='Batch size used for training')
parser.add_argument('--epochs', '-e', default=200, type=int,
                    help='By default, lr schedule is scaled accordingly')
parser.add_argument('--dataset', default='CIFAR10', choices=datasets)
parser.add_argument('--model', default='ResNet18', choices=list(models.get_model_choices()))
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--backbone', '-b',
                    help='Path to backbone network parameters to restore from')

parser.add_argument('--path-graph', help='Path to graph-*.json file.')  # WARNING: hard-coded suffix -build in generate_fname
parser.add_argument('--wnid', help='wordnet id for cifar10node dataset',
                    default='fall11')
parser.add_argument('--eval', help='eval only', action='store_true')
parser.add_argument('--test', action='store_true', help='run dataset tests')
parser.add_argument('--test-path-sanity', action='store_true',
                    help='test path classifier with oracle fc')
parser.add_argument('--test-path', action='store_true',
                    help='test path classifier with random init')
parser.add_argument('--analysis', choices=analysis.names,
                    help='Run analysis after each epoch')

parser.add_argument('--tree-supervision-weight', type=float,
                    help='Weight assigned to tree supervision losses')
parser.add_argument('--path-graph-analysis', help='Graph path, only for analysis')
parser.add_argument('--max-leaves-supervised', type=int, default=-1,
                    help='Maximum number of leaves a node can have to '
                    'contribute loss, in tree-supervised training.')
parser.add_argument('--min-leaves-supervised', type=int, default=-1,
                    help='Minimum number of leaves a node must have to '
                    'contribute loss, in tree-supervised training.')
parser.add_argument('--probability-labels', nargs='*', type=float)
parser.add_argument('--include-labels', nargs='*', type=int)
parser.add_argument('--exclude-labels', nargs='*', type=int)
parser.add_argument('--include-classes', nargs='*', type=int)

args = parser.parse_args()


if args.test:
    trainset = data.CIFAR10JointNodes()
    print(trainset[0][1])

    for wnid, text in (
            # ('fall11', 'root'),
            ('n03575240', 'instrument'),
            ('n03791235', 'motor vehicle'),
            ('n01471682', 'vertebrate')):
        dataset = data.CIFAR10Node(wnid)

        print(f'==> {text}')
        print(dict(dataset.node.old_to_new_classes))
        print(dataset.classes)
    exit()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if 'TinyImagenet200' in args.dataset:
    transform_train = data.TinyImagenet200.transform_train
    transform_test = data.TinyImagenet200.transform_val

if 'Imagenet1000' in args.dataset:
    transform_train = data.Imagenet1000.transform_train
    transform_test = data.Imagenet1000.transform_val

if args.test_path_sanity or args.test_path:
    assert 'PathSanity' in args.dataset

is_model_joint = 'JointNodes' in args.model
is_dataset_joint = 'JointNodes' in args.dataset
is_both_joint = is_model_joint and is_dataset_joint
is_neither_joint = not is_model_joint and not is_dataset_joint
both_joint_nodes = is_both_joint or is_neither_joint, (
        f'Either both model {args.model} and dataset {args.dataset} are '
        'JointNodes or neither are.'
    )

dataset = getattr(data, args.dataset)


def populate_kwargs(kwargs, object, name='Dataset', keys=()):
    for key in keys:
        value = getattr(args, key)
        if getattr(object, f'accepts_{key}', False) and value:
            kwargs[key] = value
            Colors.cyan(f'{key}:\t{value}')
        elif value:
            Colors.red(
                f'Warning: {name} does not support custom '
                f'{key}: {value}')


dataset_args = ()
if getattr(dataset, 'needs_wnid', False):
    dataset_args = (args.wnid,)

dataset_kwargs = {}
populate_kwargs(dataset_kwargs, dataset, name=f'Dataset {args.dataset}', keys=(
    'path_graph', 'include_labels', 'exclude_labels', 'include_classes',
    'probability_labels'))

# TODO: if root changes, it needs to be passed to the sanity dataset in IdInitJointTree models
# and jointNodes
trainset = dataset(*dataset_args, **dataset_kwargs, root='./data', train=True, download=True, transform=transform_train)
testset = dataset(*dataset_args, **dataset_kwargs, root='./data', train=False, download=True, transform=transform_test)

assert trainset.classes == testset.classes, (trainset.classes, testset.classes)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

Colors.cyan(f'Training with dataset {args.dataset} and {len(trainset.classes)} classes')

# Model
print('==> Building model..')
model = getattr(models, args.model)

# TODO(alvin): should dataset trees be passed to models, isntead of re-passing
# the tree path?
model_kwargs = {}
populate_kwargs(model_kwargs, model, name=f'Model {args.model}', keys=(
    'path_graph', 'max_leaves_supervised', 'min_leaves_supervised',
    'tree_supervision_weight'))

net = model(
    num_classes=len(trainset.classes),
    **model_kwargs
)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.test_path_sanity or args.test_path:
    net = models.linear(trainset.get_input_dim(), len(trainset.classes))

if args.test_path_sanity:
    net.set_weight(trainset.get_weights())

fname = generate_fname(**vars(args))
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    try:
        checkpoint = torch.load('./checkpoint/{}.pth'.format(fname))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        Colors.cyan(f'==> Checkpoint found for epoch {start_epoch} with accuracy '
              f'{best_acc} at {fname}')
    except FileNotFoundError as e:
        print('==> No checkpoint found. Skipping...')
        print(e)
def get_net():
    if device == 'cuda':
        return net.module
    return net

if args.backbone:
    print('==> Loading backbone..')
    try:
        checkpoint = torch.load(args.backbone)
        net.load_state_dict(checkpoint['net'])
    except:
        if hasattr(get_net(), 'load_backbone'):
            get_net().load_backbone(args.backbone)
        else:
            Colors.red('==> FAILED to load backbone. No `load_backbone` provided for model.')

criterion = getattr(trainset, 'criterion', nn.CrossEntropyLoss)()  # TODO(alvin): WARNING JointNodes custom_loss hard-coded to CrossEntropyLoss
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(epoch, lr):
    if epoch <= 150 / 350. * args.epochs:  # 32k iterations
      return lr
    elif epoch <= 250 / 350. * args.epochs:  # 48k iterations
      return lr/10
    else:
      return lr/100

# Training
def train(epoch, analyzer):
    analyzer.start_train(epoch)
    lr = adjust_learning_rate(epoch, args.lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = get_loss(criterion, outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = get_prediction(outputs)
        _correct, _total = get_evaluation(predicted, targets)
        correct += _correct
        total += _total

        stat = analyzer.update_batch(outputs, predicted, targets)
        extra = f'| {stat}' if stat else ''

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) %s'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, extra))

    analyzer.end_train(epoch)

def get_evaluation(predicted, targets):
    if hasattr(get_net(), 'custom_evaluation'):
        return get_net().custom_evaluation(predicted, targets)
    return predicted.eq(targets).sum().item(), np.prod(targets.size())

def get_prediction(outputs):
    if hasattr(get_net(), 'custom_prediction'):
        return get_net().custom_prediction(outputs)
    _, predicted = outputs.max(1)
    return predicted

def get_loss(criterion, outputs, targets):
    if hasattr(get_net(), 'custom_loss'):
        return get_net().custom_loss(criterion, outputs, targets)
    return criterion(outputs, targets)

def test(epoch, analyzer, checkpoint=True):
    analyzer.start_test(epoch)

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = get_loss(criterion, outputs, targets)

            test_loss += loss.item()
            predicted = get_prediction(outputs)
            _correct, _total = get_evaluation(predicted, targets)
            correct += _correct
            total += _total

            if device == 'cuda':
                predicted = predicted.cpu()
                targets = targets.cpu()

            stat = analyzer.update_batch(outputs, predicted, targets)
            extra = f'| {stat}' if stat else ''

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) %s'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, extra))

    # Save checkpoint.
    acc = 100.*correct/total
    print("Accuracy: {}, {}/{}".format(acc, correct, total))
    if acc > best_acc and checkpoint:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        fname = generate_fname(**vars(args))
        print(f'Saving to {fname} ({acc})..')
        torch.save(state, './checkpoint/{}.pth'.format(fname))
        best_acc = acc

    if hasattr(get_net(), 'save_metrics'):
        gt_classes = []
        for _, targets in testloader:
            gt_classes.extend(targets.tolist())
        get_net().save_metrics(gt_classes)

    analyzer.end_test(epoch)


generate = getattr(analysis, args.analysis) if args.analysis else analysis.Noop

analyzer_kwargs = {}
populate_kwargs(analyzer_kwargs, generate, name=f'Analyzer {args.analysis}', keys=(
    'path_graph_analysis',))
analyzer = generate(trainset, testset, **analyzer_kwargs)

if args.eval:
    if not args.resume:
        Colors.red(' * Warning: Model is not loaded from checkpoint. Use --resume')

    analyzer.start_epoch(0)
    test(0, analyzer, checkpoint=False)
    exit()

for epoch in range(start_epoch, args.epochs):
    analyzer.start_epoch(epoch)
    train(epoch, analyzer)
    test(epoch, analyzer)
    analyzer.end_epoch(epoch)

print(f'Best accuracy: {best_acc} // Checkpoint name: {fname}')
