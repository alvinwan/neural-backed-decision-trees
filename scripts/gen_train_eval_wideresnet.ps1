# Want to train with wordnet hierarchy? Just set `--hierarchy=wordnet` below.

$MODEL_NAME="wrn28_10"
$CIFAR100="CIFAR100" +  " " + $MODEL_NAME + "_cifar100 1"
$CIFAR10="CIFAR10 $MODEL_NAME" + "_cifar10 1"
$MODELS=@($CIFAR10, $CIFAR100, "TinyImagenet200 $MODEL_NAME 10")

foreach ($model in $MODELS) {

  $params = $model.split(" ")

  $dataset=$params[0]
  $model=$params[1]
  $weight=$params[2]

  # 1. generate hieararchy
  nbdt-hierarchy  --dataset=$dataset --arch=$model

  # 2. train with soft tree supervision loss
  python main.py --lr=0.01 --dataset=$dataset --arch=$model --hierarchy=induced-$model --pretrained --loss=SoftTreeSupLoss --tree-supervision-weight=$weight

  # 3. evaluate with soft then hard inference
  $analysisRules = @("SoftEmbeddedDecisionRules", "HardEmbeddedDecisionRules")
  
  foreach ($analysis in $analysisRules) {
    python main.py --dataset=${dataset} --arch=${model} --hierarchy=induced-${model} --loss=SoftTreeSupLoss --eval --resume --analysis=${analysis} --tree-supervision-weight=${weight}
  }
}