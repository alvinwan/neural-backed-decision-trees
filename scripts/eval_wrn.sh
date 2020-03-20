for loss in SoftTreeSupLoss;  # HardTreeSupLoss;  # optionally eval with hard loss too
  do
    for analysis in SoftEmbeddedDecisionRules HardEmbeddedDecisionRules; do
      for i in "CIFAR10 wrn28_10_cifar10" "CIFAR100 wrn28_10_cifar100" "TinyImagenet200 wrn28_10"; do
        read dataset model <<< "${i}";

        python main.py --dataset=${dataset} --model=${model} --path-graph=./data/${dataset}/graph-induced-${model}.json --tree-supervision-weight=10 --loss=${loss} --eval --resume --analysis=${analysis}
    done;
  done;
