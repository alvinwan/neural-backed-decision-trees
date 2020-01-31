'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils.datasets import CIFAR10NodeDataset

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import models
from utils.utils import progress_bar


CIFAR10NODE = 'CIFAR10node'
datasets = ('CIFAR10', 'CIFAR100', CIFAR10NODE)


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch-size', default=128, type=int,
                    help='Batch size used for training')
parser.add_argument('--epochs', '-e', default=350, type=int,
                    help='By default, lr schedule is scaled accordingly')
parser.add_argument('--dataset', default='CIFAR10', choices=datasets)
parser.add_argument('--model', default='ResNet18', choices=list(models.get_model_choices()))
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', action='store_true', help='run dataset tests')

parser.add_argument('--wnid', help='wordnet id for cifar10node dataset',
                    default='fall11')

args = parser.parse_args()


if args.test:
    import xml.etree.ElementTree as ET

    for wnid, text in (
            ('fall11', 'root'),
            ('n03575240', 'instrument'),
            ('n03791235', 'motor vehicle'),
            ('n02370806', 'hoofed mammal')):
        dataset = CIFAR10NodeDataset(wnid)

        print(text)
        print(dataset.mapping)

    with open('./data/cifar10/wnids.txt') as f:
        wnids = [line.strip() for line in f.readlines()]

    tree = ET.parse('./data/cifar10/tree.xml');
    for wnid in wnids:
        node = tree.find('.//synset[@wnid="{}"]'.format(wnid))
        assert len(node.getchildren()) == 0, (
            node.get('words'), [child.get('words') for child in node.getchildren()]
        )
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

dataset_args = ()
if args.dataset == CIFAR10NODE:
    dataset = CIFAR10NodeDataset
    dataset_args = (args.wnid,)
else:
    dataset = getattr(torchvision.datasets, args.dataset)

trainset = dataset(*dataset_args, root='./data', train=True, download=True, transform=transform_train)
testset = dataset(*dataset_args, root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print(f'Training with dataset {args.dataset} and classes {trainset.classes}')

# Model
print('==> Building model..')
net = getattr(models, args.model)(
    num_classes=len(trainset.classes)
)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(epoch, lr):
    if epoch <= 150 / 350. * args.epochs:  # 32k iterations
      return lr
    elif epoch <= 250 / 350. * args.epochs:  # 48k iterations
      return lr/10
    else:
      return lr/100

# Training
def train(epoch):
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
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        fname = 'ckpt'
        fname += '-' + args.dataset
        fname += '-' + args.model
        if args.dataset == CIFAR10NODE:
            fname += '-' + args.wnid

        torch.save(state, './checkpoint/{}.pth'.format(fname))
        best_acc = acc


for epoch in range(start_epoch, args.epochs):
    train(epoch)
    test(epoch)
