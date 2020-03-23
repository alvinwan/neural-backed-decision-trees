# Want to train with wordnet hierarchy? Just set `--hierarchy=wordnet` below.

for i in "CIFAR10 wrn28_10_cifar10 1" "CIFAR100 wrn28_10_cifar100 1" "TinyImagenet200 wrn28_10 10"; do
  read dataset model weight <<< "${i}";

  # 1. generate hieararchy
  nbdt-hierarchy --induced-model=${model} --dataset=${dataset}

  # 2. train with soft tree supervision loss
  python main.py --lr=0.01 --dataset=${dataset} --model=${model} --hierarchy=induced-${model} --pretrained --loss=SoftTreeSupLoss --tree-supervision-weight=${weight}

  # 3. evaluate with soft then hard inference
  for analysis in SoftEmbeddedDecisionRules HardEmbeddedDecisionRules; do
    python main.py --dataset=${dataset} --model=${model} --hierarchy=induced-${model} --loss=SoftTreeSupLoss --eval --resume --analysis=${analysis} --tree-supervision-weight=${weight}
  done
done;
