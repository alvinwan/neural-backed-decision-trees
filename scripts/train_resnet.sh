for loss in SoftTreeSupLoss; do
  for dataset in CIFAR10 CIFAR100 TinyImagenet200; do
    python main.py --lr=0.1 --dataset=${dataset} --model=${model} --path-graph=./hierarchies/${dataset}/graph-induced-${model}.json --tree-supervision-weight=10 --loss=${loss}
  done;
done;
