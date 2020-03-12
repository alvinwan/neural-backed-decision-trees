for analysis in SoftEmbeddedDecisionRules HardEmbeddedDecisionRules; do
  python main.py --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --tree-supervision-weight=10 --loss=SoftTreeSupLoss --eval --resume --analysis=${analysis}
done;
