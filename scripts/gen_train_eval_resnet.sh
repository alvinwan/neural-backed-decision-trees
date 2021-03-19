# Want to train with wordnet hierarchy? Just set `--hierarchy=wordnet` below.

for i in "CIFAR10 1" "CIFAR100 1" "TinyImagenet200 10"; do
  read dataset weight <<< "${i}";

  # 1. generate hieararchy
  nbdt-hierarchy --dataset=${dataset} --arch=ResNet18

  # 2. train with soft tree supervision loss
  python main.py --dataset=${dataset} --arch=${model} --hierarchy=induced-${model} --loss=SoftTreeSupLoss --tree-supervision-weight=${weight}

  # 3. evaluate with soft then hard inference
  for analysis in SoftEmbeddedDecisionRules HardEmbeddedDecisionRules; do
    python main.py --dataset=${dataset} --arch=${model} --hierarchy=induced-${model} --loss=SoftTreeSupLoss --tree-supervision-weight=${weight} --eval --resume --analysis=${analysis}
  done
done;
