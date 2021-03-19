# Want to train with wordnet hierarchy? Just set `--hierarchy=wordnet` below.
# This script is for networks that do NOT come with a pretrained checkpoint provided either by a model zoo or by the NBDT utility itself.

model="ResNet18"
dataset=CIFAR10
weight=1

# 0. train the baseline neural network
python main.py --dataset=${dataset} --arch=${model}

# 1. generate hieararchy -- for models without a pretrained checkpoint, use `checkpoint`
nbdt-hierarchy --dataset=${dataset} --checkpoint=./checkpoint/ckpt-${dataset}-${model}.pth

# 2. train with soft tree supervision loss -- for models without a pretrained checkpoint, use `path-resume` OR just train from scratch, without `path-resume`
# python main.py --lr=0.01 --dataset=${dataset} --model=${model} --hierarchy=induced-${model} --path-resume=./checkpoint/ckpt-${dataset}-${model}.pth --loss=SoftTreeSupLoss --tree-supervision-weight=${weight}  # fine-tuning
python main.py --dataset=${dataset} --arch=${model} --hierarchy=induced-${model} --loss=SoftTreeSupLoss --tree-supervision-weight=${weight}  # training from scratch

# 3. evaluate with soft then hard inference
for analysis in SoftEmbeddedDecisionRules HardEmbeddedDecisionRules; do
  python main.py --dataset=${dataset} --arch=${model} --hierarchy=induced-${model} --loss=SoftTreeSupLoss --eval --resume --analysis=${analysis} --tree-supervision-weight=${weight}
done
