# Want to train with wordnet hierarchy? Just set `--hierarchy=wordnet` below.

$MODELS = @("CIFAR10 1", "CIFAR100 1", "TinyImagenet200 10")

foreach ($model in $MODELS) {

  $params = $model.split(" ")

  $dataset=$params[0]
  $weight=$params[1]



  # 1. generate hieararchy
  nbdt-hierarchy --dataset=$dataset --arch=ResNet18

  # 2. train with soft tree supervision loss
  python main.py --dataset=$dataset --arch=$model --hierarchy=induced-$model --loss=SoftTreeSupLoss --tree-supervision-weight=$weight

  # 3. evaluate with soft then hard inference

  $analysisRules = @("SoftEmbeddedDecisionRules", "HardEmbeddedDecisionRules")
  
  foreach ($analysis in $analysisRules) {
    python main.py --dataset=$dataset --arch=$model --hierarchy=induced-$model --loss=SoftTreeSupLoss --tree-supervision-weight=$weight --eval --resume --analysis=$analysis
  }
}
