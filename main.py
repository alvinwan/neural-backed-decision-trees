'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import custom_datasets, nmn_datasets, analysis

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

import models
from utils.utils import (
    progress_bar, generate_fname, CIFAR10NODE, CIFAR10PATHSANITY,
    set_np_printoptions
)


set_np_printoptions()
datasets = ('CIFAR10', 'CIFAR100') + custom_datasets.names + nmn_datasets.names


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--name', default='',
                    help='Name of experiment. Used for checkpoint filename')
parser.add_argument('--batch-size', default=128, type=int,
                    help='Batch size used for training')
parser.add_argument('--epochs', '-e', default=350, type=int,
                    help='By default, lr schedule is scaled accordingly')
parser.add_argument('--dataset', default='CIFAR10', choices=datasets)
parser.add_argument('--model', default='ResNet18', choices=list(models.get_model_choices()))
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--backbone', '-b',
                    help='Path to backbone network parameters to restore from')

parser.add_argument('--path-tree', help='Path to tree-?.xml file.')  # WARNING: hard-coded suffix -build in generate_fname
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

args = parser.parse_args()


if args.test:
    import xml.etree.ElementTree as ET

    dataset = nmn_datasets.CIFAR10IncludeLabels()
    print(len(dataset))

    dataset = nmn_datasets.CIFAR10PathSanity()
    print(dataset[0][0])

    for wnid, text in (
            # ('fall11', 'root'),
            ('n03575240', 'instrument'),
            ('n03791235', 'motor vehicle'),
            ('n02370806', 'hoofed mammal')):
        dataset = nmn_datasets.CIFAR10Node(wnid)

        print(text)
        print(dataset.node.mapping)
        print(dataset.classes)

    with open('./data/cifar10/wnids.txt') as f:
        wnids = [line.strip() for line in f.readlines()]

    tree = ET.parse('./data/cifar10/tree.xml');
    for wnid in wnids:
        node = tree.find('.//synset[@wnid="{}"]'.format(wnid))
        assert len(node.getchildren()) == 0, (
            node.get('words'), [child.get('words') for child in node.getchildren()]
        )

    print(' '.join([node.get('wnid') for node in tree.iter()
          if len(node.getchildren()) > 0 and node.get('wnid')]))
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
    transform_train = custom_datasets.TinyImagenet200.transform_train
    transform_test = custom_datasets.TinyImagenet200.transform_val

if args.test_path_sanity or args.test_path:
    assert 'PathSanity' in args.dataset
if args.model == 'CIFAR10JointNodes':
    assert args.dataset == 'CIFAR10JointNodes'

if args.dataset in nmn_datasets.names:
    dataset = getattr(nmn_datasets, args.dataset)
elif args.dataset in custom_datasets.names:
    dataset = getattr(custom_datasets, args.dataset)
else:
    dataset = getattr(torchvision.datasets, args.dataset)

dataset_args = ()
dataset_kwargs = {}
if getattr(dataset, 'needs_wnid', False):
    dataset_args = (args.wnid,)
if getattr(dataset, 'accepts_path_tree', False) and args.path_tree:
    dataset_kwargs['path_tree'] = args.path_tree
elif args.path_tree:
    print(
        f' => Warning: Dataset {args.dataset} does not support custom '
        f'tree paths: {args.path_tree}')

# TODO: if root changes, it needs to be passed to the sanity dataset in IdInitJointTree models
# and jointNodes
trainset = dataset(*dataset_args, **dataset_kwargs, root='./data', train=True, download=True, transform=transform_train)
testset = dataset(*dataset_args, **dataset_kwargs, root='./data', train=False, download=True, transform=transform_test)

assert trainset.classes == testset.classes, (trainset.classes, testset.classes)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print(f'Training with dataset {args.dataset} and {len(trainset.classes)} classes')

# Model
print('==> Building model..')
model = getattr(models, args.model)

# TODO(alvin): should dataset trees be passed to models, isntead of re-passing
# the tree path?
model_kwargs = {}
if getattr(model, 'accepts_path_tree', False) and args.path_tree:
    model_kwargs['path_tree'] = args.path_tree
elif args.path_tree:
    print(
        f' => Warning: Model {args.model} does not support custom '
        f'tree paths: {args.path_tree}')

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

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    fname = generate_fname(**vars(args))
    try:
        checkpoint = torch.load('./checkpoint/{}.pth'.format(fname))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print(f'==> Checkpoint found for epoch {start_epoch} with accuracy '
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
            print('==> FAILED to load backbone. No `load_backbone` provided for model.')

criterion = nn.CrossEntropyLoss()  # TODO(alvin): WARNING JointNodes custom_loss hard-coded to CrossEntropyLoss
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
        total += np.prod(targets.size())
        correct += predicted.eq(targets).sum().item()

        analyzer.update_batch(outputs, predicted, targets)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    analyzer.end_train(epoch)

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
            total += np.prod(targets.size())
            correct += predicted.eq(targets).sum().item()

            if device == 'cuda':
                predicted = predicted.cpu()
                targets = targets.cpu()

            analyzer.update_batch(outputs, predicted, targets)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print("Accuracy: {}, {}/{}".format(acc, correct, total))
    if acc > best_acc and checkpoint:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        fname = generate_fname(**vars(args))
        torch.save(state, './checkpoint/{}.pth'.format(fname))
        best_acc = acc

    if hasattr(get_net(), 'save_metrics'):
        gt_classes = []
        for _, targets in testloader:
            gt_classes.extend(targets.tolist())
        get_net().save_metrics(gt_classes)

    analyzer.end_test(epoch)


generate = getattr(analysis, args.analysis) if args.analysis else analysis.Noop
analyzer = generate(trainset, testset)

if args.eval:
    if not args.resume:
        print(' * Warning: Model is not loaded from checkpoint. Use --resume')

    analyzer.start_epoch(0)
    test(0, analyzer, checkpoint=False)
    exit()

for epoch in range(start_epoch, args.epochs):
    analyzer.start_epoch(epoch)
    train(epoch, analyzer)
    test(epoch, analyzer)
    analyzer.end_epoch(epoch)

print(f'Best accuracy: {best_acc} // Checkpoint name: {fname}')
