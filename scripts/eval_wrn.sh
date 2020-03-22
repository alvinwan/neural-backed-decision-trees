for loss in SoftTreeSupLoss;  # HardTreeSupLoss;  # optionally eval with hard loss too
  do
    for analysis in SoftEmbeddedDecisionRules HardEmbeddedDecisionRules; do
      for i in "CIFAR10 wrn28_10_cifar10 1" "CIFAR100 wrn28_10_cifar100 1" "TinyImagenet200 wrn28_10 10"; do
        read dataset model weight <<< "${i}";

        python main.py --dataset=${dataset} --model=${model} --path-graph=./nbdt/hierarchies/${dataset}/graph-induced-${model}.json --tree-supervision-weight=${weight} --loss=${loss} --eval --resume --analysis=${analysis}
    done;
  done;
