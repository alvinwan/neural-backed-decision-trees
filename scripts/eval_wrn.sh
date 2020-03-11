for analysis in TreePrior TreeBayesianPrior; do
  python main.py --dataset=CIFAR10 --model=CIFAR10TreeBayesianSup --analysis=CIFAR10Decision${analysis} --tree-supervision-weight=10 --eval --resume --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-graph-analysis=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json
done;
