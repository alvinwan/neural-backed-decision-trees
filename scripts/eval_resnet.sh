for model in ResNet10 ResNet18; do
  for dataset in CIFAR10 CIFAR100 TinyImagenet200; do
    for analysis in SoftEmbeddedDecisionRules HardEmbeddedDecisionRules; do
      python main.py --dataset=${dataset} --model=${model} --path-graph=./hierarchies/${dataset}/graph-induced-${model}.json --tree-supervision-weight=10 --loss=SoftTreeSupLoss --eval --resume --analysis=${analysis}
    done;
  done;
done;
