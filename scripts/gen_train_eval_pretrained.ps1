# Want to train with wordnet hierarchy? Just set `--hierarchy=wordnet` below.
# This script is for networks that DO come with a pretrained checkpoint provided either by a model zoo or by the NBDT utility itself.

$model="wrn28_10_cifar10"
$dataset="CIFAR10"
$weight=1

# 1. generate hieararchy
nbdt-hierarchy --dataset=$dataset --arch=$model

# 2. train with soft tree supervision loss
python main.py --lr=0.01 --dataset=$dataset --model=$model --hierarchy=induced-$model --pretrained --loss=SoftTreeSupLoss --tree-supervision-weight=$weight

# 3. evaluate with soft then hard inference
$analysisRules = @("SoftEmbeddedDecisionRules", "HardEmbeddedDecisionRules")
  
foreach ($analysis in $analysisRules) {
  python main.py --dataset=$dataset --model=$model --hierarchy=induced-$model --loss=SoftTreeSupLoss --eval --resume --analysis=$analysis --tree-supervision-weight=$weight
} 
