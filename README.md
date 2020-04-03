# Neural-Backed Decision Trees

[Project Page](http://nbdt.alvinwan.com) &nbsp;//&nbsp; [Paper](http://nbdt.alvinwan.com/paper/) &nbsp;//&nbsp; [No-code Web Demo](http://nbdt.alvinwan.com/demo/) &nbsp;//&nbsp; [Colab Notebook](https://colab.research.google.com/github/alvinwan/neural-backed-decision-trees/blob/master/examples/load_pretrained_nbdts.ipynb)

*By Alvin Wan, \*Lisa Dunlap, \*Daniel Ho, Jihan Yin, Scott Lee, Henry Jin, Suzanne Petryk, Sarah Adel Bargal, Joseph E. Gonzalez*
<sub>*denotes equal contribution</sub>

Run decision trees that achieve state-of-the-art accuracy for explainable models on CIFAR10, CIFAR100, TinyImagenet200, and ImageNet. NBDTs achieve accuracies within 1% of the original neural network on CIFAR10, CIFAR100, and TinyImagenet200 with the recently state-of-the-art WideResNet; and within 2% of the original neural network on ImageNet, using recently state-of-the-art EfficientNet. We attain an ImageNet top-1 accuracy of 75.13%.

**Table of Contents**

- [Quickstart: Running and loading NBDTs](#quickstart)
- [Convert your own neural network into a decision tree](#convert-neural-networks-to-decision-trees)
- [Training and evaluation](#training-and-evaluation)
- [Results](#results)
- [Setup for Development](#setup-for-development)
- [Citation](#citation)

![pipeline](https://user-images.githubusercontent.com/2068077/76384774-1ffb8480-631d-11ea-973f-7cac2a60bb10.jpg)

Per the pipeline illustration above, we (1) [generate the hierarchy](https://github.com/alvinwan/neural-backed-decision-trees#1-Hierarchies) and (2) train the neural network [with a tree supervision loss](https://github.com/alvinwan/neural-backed-decision-trees#2-Tree-Supervision-Loss). Then, we (3) [run inference](https://github.com/alvinwan/neural-backed-decision-trees#3-Inference) by featurizing images using the network backbone and running embedded decision rules.

# Quickstart

## Running Pretrained NBDT on Examples

<i>Don't want to download? Try your own images on the [web demo](http://nbdt.alvinwan.com/demo/).</i>

Pip install the `nbdt` utility and run it on an image of your choosing. This can be a local image path or an image URL. Below, we evaluate on an image of a cat, from the web. This cat is pictured below.

```bash
pip install nbdt
nbdt https://images.pexels.com/photos/126407/pexels-photo-126407.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=32
```

This outputs both the class prediction and all the intermediate decisions, like below:

```
Prediction: cat // Decisions: animal (99.47%), chordate (99.20%), carnivore (99.42%), cat (99.86%)
```

By default, this evaluation utility uses WideResNet pretrained on CIFAR10. You can also pass classes not seen in CIFAR10. Below, we pass a picture of a bear and a zebra. This zebra is also pictured below.

```bash
nbdt https://images.pexels.com/photos/750539/pexels-photo-750539.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=32
nbdt https://images.pexels.com/photos/158109/kodiak-brown-bear-adult-portrait-wildlife-158109.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=32
```

Like before, this outputs the class prediction and intermediate decisions. Although the *Bear* and *Zebra* classes were not seen at train time, the model still correctly picks *Animal* over *Vehicle* for both. Note that for *Zebra*, the obvious closest class is *Horse*, matching the model's prediction below.

```
Prediction: horse // Decisions: animal (99.31%), ungulate (99.25%), horse (99.62%)
Prediction: dog // Decisions: animal (99.51%), chordate (99.35%), carnivore (99.69%), dog (99.22%)
```

<img src="https://images.pexels.com/photos/126407/pexels-photo-126407.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=125" height=125 align=left>
<img src="https://images.pexels.com/photos/158109/kodiak-brown-bear-adult-portrait-wildlife-158109.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=125" height=125 align=left>
<img src="https://images.pexels.com/photos/1490908/pexels-photo-1490908.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=125" height=125 align=left>
<img src="https://images.pexels.com/photos/750539/pexels-photo-750539.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=125" height=125>

<sub>*Pictures are taken from [pexels.com](http://pexels.com), which are free to use per the [Pexels license](https://www.pexels.com/photo-license/).*</sub>

## Loading Pretrained NBDTs in Code

<i>Don't want to download? Try inference on a pre-filled [Google Colab Notebook](https://colab.research.google.com/github/alvinwan/neural-backed-decision-trees/blob/master/examples/load_pretrained_nbdts.ipynb).</i>

If you haven't already, pip install the `nbdt` utility.

```bash
pip install nbdt
```

Then, pick an NBDT inference mode (hard or soft), dataset, and backbone. By default, we support ResNet18 and WideResNet28x10 for CIFAR10, CIFAR100, and TinyImagenet200. See [nbdt-pytorch-image-models](https://github.com/alvinwan/nbdt-pytorch-image-models) for EfficientNet-EdgeTPUSmall on ImageNet.

<sub>[Try below script on Google Colab](https://colab.research.google.com/github/alvinwan/neural-backed-decision-trees/blob/master/examples/load_pretrained_nbdts.ipynb)</sub>

```python
from nbdt.model import SoftNBDT
from nbdt.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10  # use wrn28_10 for TinyImagenet200

model = wrn28_10_cifar10()
model = SoftNBDT(
  pretrained=True,
  dataset='CIFAR10',
  arch='wrn28_10_cifar10',
  model=model)
```

Note `torchvision.models.resnet18` only supports 224x224 input. However, `nbdt.models.resnet.ResNet18` supports variable size inputs. See [Models](#models) for instructions on using your favorite image classification neural network.

**Example in ~30 lines**: See [`nbdt/bin/nbdt`](https://github.com/alvinwan/neural-backed-decision-trees/blob/master/nbdt/bin/nbdt), which loads the pretrained model, loads an image, and runs inference on the image in ~30 lines. This file is the executable `nbdt` in the previous section. Try this in a [Google Colab Notebook](https://colab.research.google.com/github/alvinwan/neural-backed-decision-trees/blob/master/examples/load_pretrained_nbdts.ipynb).

# Convert Neural Networks to Decision Trees

<!--<i>Don't want to download? Try on MNIST in a pre-filled [Google Colab Notebook]().</i>-->

**To convert your neural network** into a neural-backed decision tree, perform the following 3 steps:

1. **First**, if you haven't already, pip install the `nbdt` utility:

  ```bash
  pip install nbdt
  ```

2. **Second, train the original neural network with an NBDT loss**. All NBDT losses work by wrapping the original criterion. To demonstrate this, we wrap the original loss `criterion` with a soft tree supervision loss.

  ```python
  from nbdt.loss import SoftTreeSupLoss
  criterion = SoftTreeSupLoss(dataset='CIFAR10', criterion=criterion)  # `criterion` is your original loss function e.g., nn.CrossEntropyLoss
  ```

3. **Third, perform inference or validate using an NBDT model**. All NBDT models work by wrapping the original model you trained in step 2. To demonstrate this, we wrap the `model` with a custom NBDT wrapper below. Note this model wrapper is *only* for inference and validation, *not* for train time.

  ```python
  from nbdt.model import SoftNBDT
  model = SoftNBDT(dataset='CIFAR10', model=model)  # `model` is your original model
  ```

**Example integration with repository**: See [`nbdt-pytorch-image-models`](https://github.com/alvinwan/nbdt-pytorch-image-models), which applies this 3-step integration to a popular image classification repository `pytorch-image-models`.

<!--**Example notebook with MNIST training**: Try the [Google Colab Notebook]() with example training for MNIST, applying this 3-step integration to a simple pipeline.-->

<details><summary><b>Example integration with a random neural network in 16 lines</b> <i>[click to expand]</i></summary>
<div>

You can also include arbitrary image classification neural networks not explicitly supported in this repository. For example, after installing [`pretrained-models.pytorch`](https://github.com/Cadene/pretrained-models.pytorch#quick-examples) using pip, you can instantiate and pass any pretrained model into our NBDT utility functions.

```python
import torch.nn as nn
from nbdt.model import SoftNBDT
from nbdt.loss import SoftTreeSupLoss
from nbdt.hierarchy import generate_hierarchy
import pretrainedmodels

model = pretrainedmodels.__dict__['fbresnet152'](num_classes=1000, pretrained='imagenet')

# 1. generate hierarchy from pretrained model
generate_hierarchy(dataset='Imagenet1000', arch='fbresnet152', model=model)

# 2. Fine-tune model with tree supervision loss
criterion = nn.CrossEntropyLoss()
criterion = SoftTreeSupLoss(dataset='Imagenet1000', hierarchy='induced-fbresnet152', criterion=criterion)

# 3. Run inference using embedded decision rules
model = SoftNBDT(model=model, dataset='Imagenet1000', hierarchy='induced-fbresnet152')
```

For more information on generating different hierarchies, see [Induced Hierarchy](#induced-hierarchy).

</div>
</details>

<details><summary><b>Want to build and use your own induced hierarchy?</b> <i>[click to expand]</i></summary>
<div>

Use the `nbdt-hierarchy` utility to generate a new induced hierarchy from a pretrained model.

```bash
nbdt-hierarchy --arch=efficientnet_b0 --dataset=Imagenet1000
```

Then, pass the hierarchy name to the loss and models. You may alternatively pass the fully-qualified `path_graph` path.

```python
from nbdt.loss import SoftTreeSupLoss
from nbdt.model import SoftNBDT

criterion = SoftTreeSupLoss(dataset='Imagenet1000', criterion=criterion, hierarchy='induced-efficientnet_b0')
model = SoftNBDT(dataset='Imagenet1000', model=model, hierarchy='induced-efficientnet_b0')
```

For more information on generating different hierarchies, see [Induced Hierarchy](#induced-hierarchy).

</div>
</details>

# Training and Evaluation

**To reproduce experimental results**, start by cloning the repository and installing all requirements.

```bash
git clone git@github.com:alvinwan/neural-backed-decision-trees.git  # or http addr if you don't have private-public github key setup
cd neural-backed-decision-trees
python setup.py develop
```

To reproduce the core experimental results in our paper -- ignoring ablation studies -- simply run the following bash script:

```bash
bash scripts/gen_train_eval_wideresnet.sh
```

Want more transparent step-by-step instructions? The bash scripts above are explained in more detail in the following sections: [Induced Hierarchy](https://github.com/alvinwan/neural-backed-decision-trees#Induced-Hierarchy), [Soft Tree Supervision Loss](https://github.com/alvinwan/neural-backed-decision-trees#Tree-Supervision-Loss), and [Soft Inference](https://github.com/alvinwan/neural-backed-decision-trees#Soft-Inference). These scripts reproduce our CIFAR10, CIFAR100, and TinyImagenet200 results. To reproduce our ImageNet results, see [`nbdt-pytorch-image-models`](https://github.com/alvinwan/nbdt-pytorch-image-models).

For all scripts, you can use any [`torchvision`](https://pytorch.org/docs/stable/torchvision/models.html) model or any [`pytorchcv`](https://github.com/osmr/imgclsmob/tree/master/pytorch) model, as we directly support both model zoos. Customization for each step is explained below.

## 1. Hierarchies

### Induced Hierarchy

Run the following to generate and test induced hierarchies for CIFAR10 based off of the WideResNet model.

```bash
nbdt-hierarchy --arch=wrn28_10_cifar10 --dataset=CIFAR10
```

<details><summary><b>See how it works and how to configure.</b> <i>[click to expand]</i></summary>
<div>

![induced_structure](https://user-images.githubusercontent.com/2068077/76388304-0e6aaa80-6326-11ea-8c9b-6d08cb89fafe.jpg)

The script loads the pretrained model (Step A), populates the leaves of the tree with fully-connected layer weights (Step B) and performs hierarchical agglomerative clustering (Step C). Note that the above command can be rerun with different architectures, different datasets, or random neural network checkpoints to produce different hierarchies.

```bash
# different architecture: ResNet18
nbdt-hierarchy --arch=ResNet18 --dataset=CIFAR10

# different dataset: ImageNet
nbdt-hierarchy --arch=efficientnet_b7 --dataset=Imagenet1000

# arbitrary checkpoint
wget https://download.pytorch.org/models/resnet18-5c106cde.pth -O resnet18.pth
nbdt-hierarchy --checkpoint=resnet18.pth --dataset=Imagenet1000
```

You can also run the hierarchy generation from source directly, without using the command-line tool, by passing in a pretrained model.

```
from nbdt.hierarchy import generate_hierarchy
from nbdt.models import wrn28_10_cifar10

model = wrn28_10_cifar10(pretrained=True)
generate_hierarchy(dataset='Imagenet1000', arch='wrn28_10_cifar10', model=model)
```

</div>
</details>

<details><summary><b>See example visualization.</b> <i>[click to expand]</i></summary>
<div>

By default, the generation script outputs the HTML file containing a d3
visualization. All visualizations are stored in `out/`. We will generate another visualization with larger font size and includes wordnet IDs where available.

```
nbdt-hierarchy --vis-sublabels --vis-zoom=1.25 --dataset=CIFAR10 --arch=wrn28_10_cifar10
```

The above script's output will end with the following.

```
==> Reading from ./nbdt/hierarchies/CIFAR10/graph-induced-wrn28_10_cifar10.json
Found just 1 root.
==> Wrote HTML to out/induced-wrn28_10_cifar10-tree.html
```

Open up `out/induced-wrn28_10_cifar10-tree.html` in your browser to view the d3 tree visualization.

<img width="873" alt="Screen Shot 2020-03-24 at 1 51 49 AM" src="https://user-images.githubusercontent.com/2068077/77406559-1426ae00-6d72-11ea-90da-ae3e78b7b206.png">

</div>
</details>

<details><summary><b>Want to reproduce hierarchy visualizations from the paper?</b> <i>[click to expand]</i></summary>
<div>

To generate figures from the paper, use a larger zoom and do not include sublabels. The checkpoints used to generate the induced hierarchy visualizations are included in this repository's hub of models.

```
nbdt-hierarchy --vis-zoom=2.5 --dataset=CIFAR10 --arch=ResNet10 --vis-force-labels-left conveyance vertebrate chordate vehicle motor_vehicle mammal placental
nbdt-hierarchy --vis-zoom=2.5 --dataset=CIFAR10 --arch=wrn28_10_cifar10 --vis-leaf-images --vis-image-resize-factor=1.5 --vis-force-labels-left motor_vehicle craft chordate vertebrate carnivore ungulate craft
nbdt-hierarchy --vis-zoom=2.5 --dataset=CIFAR10 --arch=wrn28_10_cifar10 --vis-color-nodes whole --vis-no-color-leaves --vis-force-labels-left motor_vehicle craft chordate vertebrate carnivore ungulate craft
```

<img width="275" alt="CIFAR10-induced-wrn28_10_cifar10" src="https://user-images.githubusercontent.com/2068077/77971875-bd0a6700-72a4-11ea-80e8-c1308fce6c74.jpg">
<img width="275" alt="CIFAR10_ResNet10_Tree" src="https://user-images.githubusercontent.com/2068077/77971877-bed42a80-72a4-11ea-8019-e398a90829ff.jpg">
<img width="275" src="https://user-images.githubusercontent.com/2068077/77971990-0a86d400-72a5-11ea-826b-c2ea7bbf3d80.jpg">

</div>
</details>


### WordNet Hierarchy

Run the following to generate and test WordNet hierarchies for CIFAR10, CIFAR100, and TinyImagenet200. The script also downloads the NLTK WordNet corpus.

```bash
bash scripts/generate_hierarchies_wordnet.sh
```

<details><summary><b>See how it works.</b> <i>[click to expand]</i></summary>
<div>

The below just explains the above `generate_hierarchies_wordnet.sh`, using CIFAR10. You do not need to run the following after running the above bash script.

```bash
# Generate mapping from classes to WNID. This is required for CIFAR10 and CIFAR100.
nbdt-wnids --dataset=CIFAR10

# Generate hierarchy, using the WNIDs. This is required for all datasets: CIFAR10, CIFAR100, TinyImagenet200
nbdt-hierarchy --method=wordnet --dataset=CIFAR10
```
</details>

<details><summary><b>See example visualization.</b> <i>[click to expand]</i></summary>
<div>

We can generate a visualization with a slightly improved zoom and with wordnet IDs. By default, the script builds the Wordnet hierarchy for CIFAR10.

```
nbdt-hierarchy --method=wordnet --vis-zoom=1.25 --vis-sublabels
```

<img width="1002" alt="Screen Shot 2020-03-24 at 2 02 16 AM" src="https://user-images.githubusercontent.com/2068077/77407533-81870e80-6d73-11ea-9841-14b2caf13285.png">

</div>
</details>


### Random Hierarchy

Use `--method=random` to randomly generate a binary-ish hierarchy. Optionally, use the `--seed` (`--seed=-1` to *not* shuffle leaves) and `--branching-factor` flags. When debugging, we set branching factor to the number of classes. For example, the sanity check hierarchy for CIFAR10 is

```bash
nbdt-hierarchy --seed=-1 --branching-factor=10 --dataset=CIFAR10
```

## 2. Tree Supervision Loss

In the below training commands, we uniformly use `--path-resume=<path/to/checkpoint> --lr=0.01` to fine-tune instead of training from scratch. Our results using a recently state-of-the-art pretrained checkpoint (WideResNet) were fine-tuned. Run the following to fine-tune WideResNet with soft tree supervision loss on CIFAR10.

```bash
python main.py --lr=0.01 --dataset=CIFAR10 --arch=wrn28_10_cifar10 --hierarchy=induced-wrn28_10_cifar10 --pretrained --loss=SoftTreeSupLoss
```

<details><summary><b>See how it works and how to configure.</b> <i>[click to expand]</i></summary>
<div>

![tree_supervision_loss](https://user-images.githubusercontent.com/2068077/77226784-3208ce80-6b38-11ea-84bb-5128e3836665.jpg)

The tree supervision loss features two variants: a hard version and a soft version. Simply change the loss to `HardTreeSupLoss` or `SoftTreeSupLoss`, depending on the one you want.

```bash
# fine-tune the wrn pretrained checkpoint on CIFAR10 with hard tree supervision loss
python main.py --lr=0.01 --dataset=CIFAR10 --arch=wrn28_10_cifar10 --hierarchy=induced-wrn28_10_cifar10 --pretrained --loss=HardTreeSupLoss

# fine-tune the wrn pretrained checkpoint on CIFAR10 with soft tree supervision loss
python main.py --lr=0.01 --dataset=CIFAR10 --arch=wrn28_10_cifar10 --hierarchy=induced-wrn28_10_cifar10 --pretrained --loss=SoftTreeSupLoss
```

To train from scratch, use `--lr=0.1` and do not pass the `--path-resume` or `--pretrained` flags. We fine-tune WideResnet on CIFAR10, CIFAR100, but where the baseline neural network accuracy is reproducible, we train from scratch.
</div>
</details>

## 3. Inference

Like with the tree supervision loss variants, there are two inference variants: one is hard and one is soft. Below, we run soft inference on the model we just trained with the soft loss.

Run the following bash script to obtain these numbers.

```bash
python main.py --dataset=CIFAR10 --arch=wrn28_10_cifar10 --hierarchy=induced-wrn28_10_cifar10 --loss=SoftTreeSupLoss --eval --resume --analysis=SoftEmbeddedDecisionRules
```

<details><summary><b>See how it works and how to configure.</b> <i>[click to expand]</i></summary>
<div>

![inference_modes](https://user-images.githubusercontent.com/2068077/76388544-9f418600-6326-11ea-9214-17356c71a066.jpg)

Note the following commands are nearly identical to the corresponding train commands -- we drop the `lr`, `pretrained` flags and add `resume`, `eval`, and the `analysis` type (hard or soft inference). The best results in our paper, oddly enough, were obtained by running hard and soft inference *both* on the neural network supervised by a soft tree supervision loss. This is reflected in the commands below.

```bash
# running soft inference on soft-supervised model
python main.py --dataset=CIFAR10 --arch=wrn28_10_cifar10 --hierarchy=induced-wrn28_10_cifar10 --loss=SoftTreeSupLoss --eval --resume --analysis=SoftEmbeddedDecisionRules

# running hard inference on soft-supervised model
python main.py --dataset=CIFAR10 --arch=wrn28_10_cifar10 --hierarchy=induced-wrn28_10_cifar10 --loss=SoftTreeSupLoss --eval --resume --analysis=HardEmbeddedDecisionRules
```
</div>
</details>

# Results

We compare against all previous decision-tree-based methods that report on CIFAR10, CIFAR100, and/or ImageNet, including methods that hinder interpretability by using impure leaves or a random forest. We report the baseline with the highest accuracy, of all these methods: Deep  Neural  Decision  Forest  (DNDF  updated with ResNet18), Explainable Observer-Classifier (XOC), Deep ConvolutionalDecision Jungle (DCDJ), Network of Experts (NofE), Deep Decision Network(DDN), and Adaptive Neural Trees (ANT).

|                      | CIFAR10 | CIFAR100 | TinyImagenet200 | ImageNet |
|----------------------|---------|----------|-----------------|----------|
| NBDT-S (Ours)        | 97.57%  | 82.87%   | 66.66%          | 75.13%   |
| NBDT-H (Ours)        | 97.55%  | 82.21%   | 64.39%          | 74.79%   |
| Best Pre-NBDT Acc    | 94.32%  | 76.24%   | 44.56%          | 61.29%   |
| Best Pre-NBDT Method | DNDF    | NofE     | DNDF            | NofE     |
| Our improvement      | 3.25%   | 6.63%    | 22.1%           | 13.84%   |

As the last row denotes, we outperform all previous decision-tree-based methods by anywhere from 3% (CIFAR10) to 13%+ (ImageNet). Note that accuracies in our pretrained checkpoints for small to medium datasets (CIFAR10, CIFAR100, and TinyImagenet200) may fluctuate by 0.1-0.2%, as we retrained all models with the current public version of this repository.

# Setup for Development

As discussed above, you can use the `nbdt` python library to integrate NBDT training into any existing training pipeline. However, if you wish to use the barebones training utilities here, refer to the following sections for adding custom models and datasets.

If you have not already, start by cloning the repository and installing all requirements.

```bash
git clone git@github.com:alvinwan/neural-backed-decision-trees.git  # or http addr if you don't have private-public github key setup
cd neural-backed-decision-trees
python setup.py develop
```

As a sample, we've included copies of the WideResNet bash script but for ResNet18.

```bash
bash scripts/gen_train_eval_resnet.sh
```

For any models that have pretrained checkpoints for the datasets of interest (e.g., CIFAR10, CIFAR100, and ImageNet models from `pytorchcv` or ImageNet models from `torchvision`), modify `scripts/gen_train_eval_pretrained.sh`; it suffices to change the model name. For all models that do not have pretrained checkpoint for the dataset of interest, modify `scripts/gen_train_eval_nopretrained.sh`.

## Models

Without any modifications to `main.py`, you can replace ResNet18 with your favorite network: Pass  any [`torchvision.models`](https://pytorch.org/docs/stable/torchvision/models.html) model or any [`pytorchcv`](https://github.com/osmr/imgclsmob/tree/master/pytorch) model to `--arch`, as we directly support both model zoos. Note that the former only supports models pretrained on ImageNet. The latter supports models pretrained on CIFAR10, CIFAR100, andd ImageNet; for each dataset, the corresponding model name includes the dataset e.g., `wrn28_10_cifar10`. However, neither supports models pretrained on TinyImagenet.

To add a new model from scratch:

1. Create a new file containing your network, such as `./nbdt/models/yournet.py`. This file should contain an `__all__` only exposing functions that return a model. These functions should accept `pretrained: bool` and `progress: bool`, then forward all other keyword arguments to the model constructor.
2. Expose your new file via `./nbdt/models/__init__.py`: `from .yournet import *`.
3. Train the original neural network on the target dataset. e.g., `python main.py --arch=yournet18`.

## Dataset

Without any modifications to `main.py`, you can use any image classification dataset found at [`torchvision.datasets`](https://pytorch.org/docs/stable/torchvision/datasets.html) by passing it to `--dataset`. To add a new dataset from scratch:

1. Create a new file containing your dataset, such as `./nbdt/data/yourdata.py`. Say the data class is `YourData10`. Like before, only expose the dataset class via `__all__`. This dataset class should support a `.classes` attribute which returns a list of human-readable class names.
2. Expose your new file via `'./nbdt/data/__init__.py'`: `from .yourdata import *`.
3. Create a text file with wordnet IDs in `./nbdt/wnids/{dataset}.txt`. This list should be in the same order that your dataset's `.classes` is. You may optionally use the utility `nbdt-wnids` to generate wnids (see note below)
4. Train the original neural network on the target dataset. e.g., `python main.py --dataset=YourData10`

> **\*Note**: You may optionally use the utility `nbdt-wnids` to generate wnids:
> ```
> nbdt-wnids --dataset=YourData10
> ```
> , where `YourData` is your dataset name. If a provided class name from `YourData.classes` does not exist in the WordNet corpus, the script will generate a fake wnid. This does not affect training but subsequent analysis scripts will be unable to provide WordNet-imputed node meanings.

## Tests

To run tests, use the following command

```
pytest nbdt tests
```

# Citation

If you find this work useful for your research, please cite our [paper](http://nbdt.alvinwan.com/paper/):

```
@misc{nbdt,
    title={NBDT: Neural-Backed Decision Trees},
    author={Alvin Wan and Lisa Dunlap and Daniel Ho and Jihan Yin and Scott Lee and Henry Jin and Suzanne Petryk and Sarah Adel Bargal and Joseph E. Gonzalez},
    year={2020},
    eprint={2004.00221},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
