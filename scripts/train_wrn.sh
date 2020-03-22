python main.py --lr=0.01 --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./nbdt/hierarchies/CIFAR10/graph-induced-wrn28_10_cifar10.json --pretrained --loss=${loss}
python main.py --lr=0.01 --dataset=CIFAR100 --model=wrn28_10_cifar100 --path-graph=./nbdt/hierarchies/CIFAR100/graph-induced-wrn28_10_cifar100.json --pretrained --loss=${loss}
python main.py --lr=0.1 --dataset=TinyImagenet200 --model=wrn28_10 --path-graph=./nbdt/hierarchies/TinyImagenet200/graph-induced-wrn28_10.json --tree-supervision-weight=10 --loss=${loss} --batch-size=128
