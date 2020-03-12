for loss in HardTreeSupLoss SoftTreeSupLoss; do
  python main.py --lr=0.01 --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-resume=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --tree-supervision-weight=10 --loss=${loss}
  python main.py --lr=0.01 --dataset=CIFAR100 --model=wrn28_10_cifar100 --path-graph=./data/CIFAR100/graph-induced-wrn28_10_cifar100.json --path-resume=checkpoint/ckpt-CIFAR100-wrn28_10_cifar100.pth --tree-supervision-weight=10 --loss=${loss}
done;
