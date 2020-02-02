Notes:
- downloaded structure_released.xml from http://image-net.org/download-toolbox
- downloaded tinyimagenet from https://tiny-imagenet.herokuapp.com/

### Results

| Dataset | ResNet10 | ResNet18 | ResNet101 | ResNet10 Tree | ResNet10 JointTree |
| --- | --- | --- | --- | --- | --- |
| *batch size* | 512 | 512 | 128 | 512 | 512 |
| CIFAR10 | 93.64% | 94.92% | 95.31% | 93.75% | 93.11% |
| CIFAR100 | 73.66% | 75.92% | 79.46% | - | 68.24% |

- The ResNet10Tree and the ResNet101 models have comparable complexity -- former with 9 gflops, 135 mb params and the latter with 8 gflops, 155 mb params.
- The ResNet10JointTree and the ResNet10 models have comparable complexity

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
