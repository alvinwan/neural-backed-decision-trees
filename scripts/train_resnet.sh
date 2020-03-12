for loss in HardTreeSupLoss SoftTreeSupLoss; do
  for dataset in CIFAR10 CIFAR100 TinyImagenet200; do
    python main.py --lr=0.01 --dataset=${dataset} --model=${model} --path-graph=./data/${dataset}/graph-induced-${model}.json --path-resume=checkpoint/ckpt-${dataset}-${model}.pth --tree-supervision-weight=10 --loss=${loss}
  done;
done;
