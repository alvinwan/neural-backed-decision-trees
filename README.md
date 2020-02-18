## Setup

- downloaded tiny-imagenet-200 from https://tiny-imagenet.herokuapp.com/

## Tree

> Too lazy? Run `bash scripts/generate_trees.sh` to generate trees for all
datasets with all methods.

First, generate the wnids. All the Imagenet datasets come with wnids. This is only needed for CIFAR{10,100}.

```
python generate_wnids.py --dataset=CIFAR100
```

Next, build the tree. By default, the tree uses wordnet hierarchy and is built from scratch.

```
python generate_tree.py --dataset=CIFAR100 --method=build
```

> One of the old methods `prune` would prune the `structure_released.xml` from
from http://image-net.org/download-toolbox. This is no longer used but the code
is still in the codebase.

Finally, check the tree is somewhat sane.

```
python test_generated_tree.py --dataset=CIFAR100 --method=build
```

You may use any of the `CIFAR10`, `CIFAR100`, `TinyImagenet200` datasets. Additionally, use `--method` to randomly generate a binary-ish tree. Randomness is seeded.

## Training

To get started: First, train the nodes, with a shared backbone. Optionally pass in a `--path-tree=...` to customize your tree.

```
python main.py --model=CIFAR100JointNodes --dataset=CIFAR100JointNodes --batch-size=512 --epochs=200
```

Second, train the final fully-connected layer. If you passed in `--path-tree` to the last command, make sure to pass in the same tree path to this one.

```
python main.py --model=CIFAR100JointTree --dataset=CIFAR100 --batch-size=512 --epochs=200 --lr=0.01
```

This is the 'standard' pipeline. There are a few other pipelines to be aware of.

### Models

There are other models that do not need special flags. You can simply swap out the models in the 'standard' pipeline.

- `CIFAR10*`
- `TinyImagenet200*`
- `CIFAR100BalancedJointNodes`, `CIFAR100BalancedJointTree` (not helpful)

#### Frozen Backbone

So far, our best models are fine-tuned, where the shared backcbone is pretrained and frozen. The commands below train the frozen variants of the model.

```
python main.py --model=CIFAR100FreezeJointNodes --dataset=CIFAR100JointNodes --batch-size=512 --epochs=200 --backbone=./checkpoint/ckpt-ResNet10-CIFAR100.pth
python main.py --model=CIFAR100FreezeJointTree --dataset=CIFAR100 --batch-size=512 --epochs=200 --lr=0.01
```

#### Individual Nodes

One of our earliest experiments was to train each node individually, without sharing backbones. Consider all wnids in the tree, that are *not* leaves.

```
for wnid in wnids; do python main.py --model=ResNet10 --dataset=CIFAR10Node --wnid=${wnid} --batch-size=512 --epochs=200; done
python main.py --model=CIFAR10Tree --dataset=CIFAR10 --batch-size=512 --epochs=200
```

### Inference Modes

These inference modes do not require the second fully-connected layer training. Instead, inference is run directly on the outputted tree.

https://docs.google.com/spreadsheets/d/1xFTubRU3J6e8XLSgzZHURQAp66UtaCqNS5ehCd-K9jI/edit#gid=0

-----------------

Really just [Kuang Liu's pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), but with a few extra commits:
- Learning rate automatically adjusted
- Model functions accept a num_classes argument
- CLI supports a `--model` flag, to pick models by name
- proper .gitignore

# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |

## Learning rate adjustment
I manually change the `lr` during training:
- `0.1` for epoch `[0,150)`
- `0.01` for epoch `[150,250)`
- `0.001` for epoch `[250,350)`

Resume the training with `python main.py --resume --lr=0.01`
