## Setup

To get started,

1. Generate the tree, per the section below.
2. Then, launch training scripts, which use those trees.

## Graphs

### Generation

> Too lazy? Run `bash scripts/generate_trees.sh` to generate trees for all
datasets with all methods.

First, generate the wnids. All the Imagenet datasets come with wnids. This is only needed for CIFAR{10,100}.

```
python generate_wnids.py
```

Next, build the tree. By default, the tree uses wordnet hierarchy and is built from scratch.

```
python generate_graph.py
```

### Test Graph

Finally, check the tree is somewhat sane.

```
python test_generated_graph.py
```

Make sure that your output ends with `==> All checks pass!`.

### Visualize Graph

Run the visualization generation script to obtain both the JSON representing
the tree and the HTML file containing a d3 visualization.

```
python generate_vis.py
```

The above script will output the following.

```
==> Reading from ./data/CIFAR10/graph-wordnet.json
==> Found just 1 root.
==> Wrote HTML to out/wordnet-tree.html
==> Wrote HTML to out/wordnet-graph.html
```

There are two visualizations. Open up `out/wordnet-tree.html` in your browser
to view the d3 tree visualization.

<img width="1436" alt="Screen Shot 2020-02-22 at 1 52 51 AM" src="https://user-images.githubusercontent.com/2068077/75101893-ca8f4b80-5598-11ea-9b47-7adcc3fc3027.png">

Open up `out/wordnet-graph.html` in your browser to view the d3 graph
visualization.

### Random Graphs

Use `--method=random` to randomly generate a binary-ish tree. Additionally,
random trees feature two more flags:

- `--seed` to generate random leaf orderings and
- `--branching-factor` to generate trees with different branching factors.

### Random Augmentations

<!-- TODO(alvin): describe extra nodes being added and different ways of adding them -->

### Datasets

For all of the above calls, you may use any of the `CIFAR10`, `CIFAR100`, `TinyImagenet200` datasets, by passing the `--dataset` flag.

## Training

<!-- TODO(alvin): add in tree supervised training -->

To get started: First, train the nodes, with a shared backbone. Optionally pass in a `--path-graph=...` to customize your tree.

```
python main.py --model=CIFAR100JointNodes --dataset=CIFAR100JointNodes
```

Second, train the final fully-connected layer. If you passed in `--path-graph` to the last command, make sure to pass in the same tree path to this one.

```
python main.py --model=CIFAR100JointTree --dataset=CIFAR100 --lr=0.01
```

This is the 'standard' pipeline. There are a few other pipelines to be aware of.

### Cross Entropy v. Binary Cross Entropy

<!-- TODO (alvin): add notes about cross entropy v binary cross entropy versions -->

### Frozen Backbone

So far, our best models are fine-tuned, where the shared backbone is pretrained and frozen. The commands below train the frozen variants of the model.

```
python main.py --model=CIFAR100FreezeJointNodes --dataset=CIFAR100JointNodes --backbone=./checkpoint/ckpt-ResNet10-CIFAR100.pth
python main.py --model=CIFAR100FreezeJointTree --dataset=CIFAR100 --lr=0.01
```

### Identity Initialization

<!-- TODO(alvin) -->

### Balancing

<!-- TODO(alvin) class imbalance + loss imbalance -->

### Individual Nodes

One of our earliest experiments was to train each node individually, without sharing backbones. Consider all wnids in the tree, that are *not* leaves.

```
for wnid in wnids; do python main.py --model=ResNet10 --dataset=CIFAR10Node --wnid=${wnid}; done
python main.py --model=CIFAR10Tree --dataset=CIFAR10
```

### Inference Modes

These inference modes do not require the second fully-connected layer training. Instead, inference is run directly on the outputted tree.

Notes:
```
CUDA_VISIBLE_DEVICES=1 python main.py --dataset=CIFAR10 --model=CIFAR10TreeSup --analysis=CIFAR10DecisionTreePrior --eval --resume --path-graph=./data/CIFAR10/graph-wordnet-single.json --path-graph-analysis=./data/CIFAR10/graph-wornet-single.json
CUDA_VISIBLE_DEVICES=1 python main.py --dataset=CIFAR100 --model=CIFAR100TreeSup --analysis=CIFAR100DecisionTreePrior --eval --resume --path-graph=./data/CIFAR100/graph-wordnet-single.json --path-graph-analysis=./data/CIFAR100/graph-wordnet-single.json
```

## Results

https://docs.google.com/spreadsheets/d/1DrvP4msf8Bn0dF1qnpdI5fjLgEp8K6xFbxXntSn1j2s/edit#gid=0

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
