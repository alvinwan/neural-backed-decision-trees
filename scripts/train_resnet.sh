for model in ResNet18; do
  python main.py --lr=0.1 --dataset=CIFAR10 --model=${model} --path-graph=./nbdt/hierarchies/CIFAR10/graph-induced-${model}.json --loss=SoftTreeSupLoss
  python main.py --lr=0.1 --dataset=CIFAR100 --model=${model} --path-graph=./nbdt/hierarchies/CIFAR100/graph-induced-${model}.json --loss=SoftTreeSupLoss
  python main.py --lr=0.1 --dataset=TinyImagenet200 --model=${model} --path-graph=./nbdt/hierarchies/TinyImagenet200/graph-induced-${model}.json --loss=SoftTreeSupLoss --tree-supervision-weight=10
done;
