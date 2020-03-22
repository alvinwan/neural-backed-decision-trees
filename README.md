# Neural-Backed Decision Trees

Run decision trees that achieve state-of-the-art accuracy for explainable models on CIFAR10, CIFAR100, TinyImagenet200, and Imagenet. NBDTs achieve accuracies within 1% of the original neural network on CIFAR10, CIFAR100, and TinyImagenet200 with the recently state-of-the-art WideResNet.

<sub>**NBDT Accuracy per dataset**: CIFAR10 (97.57%), CIFAR100 (82.87%), TinyImagenet200 (66.66%), Imagenet (67.47%). [See more results](#results)</sub>

**Table of Contents**

- [Quickstart: Running and loading NBDTs](#quickstart)
- [Convert your own neural network into a decision tree](#convert-neural-networks-to-decision-trees)
- [Training and evaluation](#training-and-evaluation)
- [Results](#results)
- [Developing](#developing)

![pipeline](https://user-images.githubusercontent.com/2068077/76384774-1ffb8480-631d-11ea-973f-7cac2a60bb10.jpg)

Per the pipeline illustration above, we (1) [generate the hierarchy](https://github.com/alvinwan/neural-backed-decision-trees#1-Hierarchies) and (2) train the neural network [with a tree supervision loss](https://github.com/alvinwan/neural-backed-decision-trees#2-Tree-Supervision-Loss). Then, we (3) [run inference](https://github.com/alvinwan/neural-backed-decision-trees#3-Inference) by featurizing images using the network backbone and running embedded decision rules.

<!-- TODO: link to paper-->

# Quickstart

## Running Pretrained NBDT on Examples

Pip install the `nbdt` utility and run it on an image of your choosing. This can be a local image path or an image URL. Below, we evaluate on an image of a cat, from the web. This cat is pictured below.

```bash
pip install nbdt
nbdt-eval https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=32
```

This outputs both the class prediction and all the intermediate decisions, like below:

TODO

By default, this evaluation utility uses WideResNet pretrained on CIFAR10. You can also pass classes not seen in CIFAR10. Below, we pass a picture of a bear. This bear is also pictured below.

```bash
nbdt-eval https://images.pexels.com/photos/1466592/pexels-photo-1466592.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=32
```

Like before, this outputs the class prediction and intermediate decisions. Although this class was not seen at train time, the model still correctly disambiguates animal from vehicle, when classifying the bear.

TODO

<img src="https://images.pexels.com/photos/126407/pexels-photo-126407.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=300" width=297 align=left>
<img src="https://images.pexels.com/photos/158109/kodiak-brown-bear-adult-portrait-wildlife-158109.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=300" width=252 align=left>
<img src="https://images.pexels.com/photos/1490908/pexels-photo-1490908.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=300" width=252>

<sub>*Pictures are taken from [pexels.com](http://pexels.com), which are free to use per the [Pexels license](https://www.pexels.com/photo-license/).*</sub>

## Loading Pretrained NBDTs in Code

If you haven't already, pip install the `nbdt` utility.

```bash
pip install nbdt
```

Then, pick an NBDT inference mode (hard or soft), dataset, and backbone. By default, we support ResNet18 and WideResNet28_10 for CIFAR10, CIFAR100, and TinyImagenet200. See [nbdt-pytorch-image-models](https://github.com/alvinwan/nbdt-pytorch-image-models) for EfficientNet-EdgeTPUSmall on Imagenet.

```python
from nbdt.model import SoftNBDT
from pytorchcv.models.wrn_cifar import wrn28_10_cifar10

model = wrn28_10_cifar10()
model = SoftNBDT(dataset='CIFAR10', model=model, hierarchy='induced-wrn28_10_cifar10', pretrained=True)
```

> **Note about model names**: WideResNet models are at `pytorchcv.models.wrn_cifar.wrn28_10_cifar{10,100}` and `nbdt.models.wideresnet.wrn28_10` (for TinyImagenet200). The ResNet models provided with nbdt (`nbdt.models.resnet.ResNet{10,18,34,50,101,152}`) support all datasets. Conversely, the ResNet models provided in torchvision `torchvision.models.resnet18` only supports 224x224 input. Here are import statements for all models with pretrained NBDTs:
> ```python
> from pytorchcv.models.wrn_cifar import wrn28_10_cifar10, wrn28_10_cifar100
> from nbdt.models.resnet import ResNet10, ResNet18
> from nbdt.models.wideresnet import wrn28_10
> ```

# Convert Neural Networks to Decision Trees

**To convert your neural network** into a neural-backed decision tree, perform the following 3 steps:

1. **First**, if you haven't already, pip install the `nbdt` utility:

  ```bash
  pip install nbdt
  ```

2. **Second**, wrap your loss function `criterion` with a custom NBDT loss. Below, we demonstrate usage of the soft tree supervision loss, on the CIFAR10 dataset. By default, we support the CIFAR10, CIFAR100, TinyImagenet200, and Imagenet1000 image classification datasets.

  <!-- TODO: If no wnids, generate fake ones. Attempt to dataset.classes. For a new dataset, use cli to generate induced-hierarchy from checkpoint. -->

  ```python
  from nbdt.loss import SoftTreeSupLoss
  criterion = SoftTreeSupLoss(dataset='CIFAR10', criterion=criterion)  # `criterion` is your original loss function e.g., nn.CrossEntropyLoss
  ```

3. **Third**, wrap your `model` with a custom NBDT wrapper as shown below. This is only to run prediction as an NBDT during validation or inference time. Do not wrap your model like below, during training.

  ```python
  from nbdt.model import SoftNBDT
  model = SoftNBDT(dataset='CIFAR10', model=model)  # `model` is your original model
  ```

  > **Do not wrap your model during training**. When training, the tree supervision loss expects the neural network logits as input, not the NBDT outputs.

:arrow_right: **Example**: See [`nbdt-pytorch-image-models`](https://github.com/alvinwan/nbdt-pytorch-image-models), which applies this 3-step integration to a popular image classification repository `pytorch-image-models`.

<!-- TODO: include simpler example -->

<details><summary>Want to build or use your own induced hierarchy? <i>[click to expand]</i></summary>
<div>

(Optional) You may also build and use your own induced hierarchies, instead of the default induced hierarchy provided. Use the `nbdt` utility to generate a new induced hierarchy from a pretrained model, then specify the hierarchy to use.

```bash
nbdt-hierarchy --model=ResNet34 --dataset=CIFAR10
```

```python
from nbdt.loss import SoftTreeSupLoss
from nbdt.model import SoftNBDT

criterion = SoftTreeSupLoss(dataset='CIFAR10', criterion=criterion, hierarchy='induced-ResNet34')
model = SoftNBDT(dataset='CIFAR10', model=model, hierarchy='induced-ResNet34')
```
</div>
</details>

# Training and Evaluation

**To reproduce experimental results**, start by cloning the repository and installing all requirements.

```bash
git clone git@github.com:alvinwan/neural-backed-decision-trees.git
cd neural-backed-decision-trees
python setup.py develop
```

To reproduce the core experimental results in our paper -- ignoring ablation studies -- simply run the following bash scripts:

```bash
bash scripts/generate_hierarchies_induced_wrn.sh
bash scripts/train_wrn.sh
bash scripts/eval_wrn.sh
```

The bash scripts above are explained in more detail in [Induced Hierarchy](https://github.com/alvinwan/neural-backed-decision-trees#Induced-Hierarchy), [Soft Tree Supervision Loss](https://github.com/alvinwan/neural-backed-decision-trees#Tree-Supervision-Loss), and [Soft Inference](https://github.com/alvinwan/neural-backed-decision-trees#Soft-Inference). To reproduce our Imagenet results, see [`nbdt-pytorch-image-models`](https://github.com/alvinwan/nbdt-pytorch-image-models).

You can amend these scripts with the appropriate models or datasets to customize your experiments. See step-by-step instructions for and examples of customization in [Developing](#developing).

## 1. Hierarchies

### Induced Hierarchy

Run the following to generate and test induced hierarchies for CIFAR10, CIFAR100, based off of the WideResNet model. The script also downloads pretrained WideResNet models.

```bash
bash scripts/generate_hierarchies_induced_wrn.sh
```

<details><summary>Line-by-line Explanation. <i>[click to expand]</i></summary>
<div>

![induced_structure](https://user-images.githubusercontent.com/2068077/76388304-0e6aaa80-6326-11ea-8c9b-6d08cb89fafe.jpg)

The below just explains the above `generate_hierarches_induced.sh`, using CIFAR10. You do not need to run the following after running the above bash script. Note that the following commands can be rerun with different checkpoints from different architectures, to produce different hierarchies.

```bash
# Step A. Download and evaluate pre-trained weights for WideResNet on CIFAR10.
python main.py --eval --pretrained --model=wrn28_10_cifar10 --dataset=CIFAR10

# Step B through D. Generate induced hierarchies, using the pretrained checkpoints. Also tests hierarchy and outputs visualization
nbdt-hierarchy --method=induced --induced-checkpoint=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --dataset=CIFAR10
```
</div>
</details>

### Wordnet Hierarchy

Run the following to generate and test Wordnet hierarchies for CIFAR10, CIFAR100, and TinyImagenet200. The script also downloads the NLTK Wordnet corpus.

```bash
bash scripts/generate_hierarchies_wordnet.sh
```

<details><summary>Line-by-line Explanation. <i>[click to expand]</i></summary>
<div>
The below just explains the above `generate_hierarchies_wordnet.sh`, using CIFAR10. You do not need to run the following after running the above bash script.

```bash
# Generate mapping from classes to WNID. This is required for CIFAR10 and CIFAR100.
nbdt-wnids --dataset=CIFAR10

# Generate hierarchy, using the WNIDs. This is required for all datasets: CIFAR10, CIFAR100, TinyImagenet200
nbdt-hierarchy --single-path --dataset=CIFAR10
```
</details>

### Random Hierarchy

Use `--method=random` to randomly generate a binary-ish hierarchy. Optionally, use the `--seed` (`--seed=-1` to *not* shuffle leaves) and `--branching-factor` flags. When debugging, we set branching factor to the number of classes. For example, the sanity check hierarchy for CIFAR10 is

```bash
nbdt-hierarchy --seed=-1 --branching-factor=10 --single-path --dataset=CIFAR10
```

### Visualize Hierarchy

By default, the generation script outputs both the JSON representing
the hierarchy and the HTML file containing a d3 visualization. All visualizations
are stored in `out/`.

<details><summary>See example output.</summary>
<div>

The above script will output the following.

```
==> Reading from ./data/CIFAR10/graph-wordnet-single.json
==> Found just 1 root.
==> Wrote HTML to out/wordnet-single-tree.html
==> Wrote HTML to out/wordnet-single-graph.html
```

There are two visualizations. Open up `out/wordnet-single-tree.html` in your browser
to view the d3 tree visualization.

<img width="1436" alt="Screen Shot 2020-02-22 at 1 52 51 AM" src="https://user-images.githubusercontent.com/2068077/75101893-ca8f4b80-5598-11ea-9b47-7adcc3fc3027.png">

Open up `out/wordnet-single-graph.html` in your browser to view the d3 graph
visualization.
</div>
</details>

## 2. Tree Supervision Loss

In the below training commands, we uniformly use `--path-resume=<path/to/checkpoint> --lr=0.01` to fine-tune instead of training from scratch. Our results using a recently state-of-the-art pretrained checkpoint (WideResNet) were fine-tuned.

Run the following bash script to fine-tune WideResNet with both hard and soft tree supervision loss on CIFAR10, CIFAR100.

```bash
bash scripts/train_wrn.sh
```

<details><summary>Line-by-line Explanation. <i>[click to expand]</i></summary>
<div>

![tree_supervision_loss](https://user-images.githubusercontent.com/2068077/77226784-3208ce80-6b38-11ea-84bb-5128e3836665.jpg)

As before, the below just explains the above `train_wrn.sh`. You do not need to run the following after running the previous bash script.

```bash
# fine-tune the wrn pretrained checkpoint on CIFAR10 with hard tree supervision loss
python main.py --lr=0.01 --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./nbdt/hierarchies/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-resume=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --tree-supervision-weight=10 --loss=HardTreeSupLoss

# fine-tune the wrn pretrained checkpoint on CIFAR10 with soft tree supervision loss
python main.py --lr=0.01 --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./nbdt/hierarchies/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-resume=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --tree-supervision-weight=10 --loss=SoftTreeSupLoss
```

To train from scratch, use `--lr=0.1` and do not pass the `--path-resume` flag. We fine-tune WideResnet on CIFAR10, CIFAR100, but where the baseline neural network accuracy is reproducible, we train from scratch.
</div>
</details>

## 3. Inference

Like with the tree supervision loss variants, there are two inference variants: one is hard and one is soft. The best results in our paper, oddly enough, were obtained by running hard and soft inference *both* on the neural network supervised by a soft tree supervision loss.

Run the following bash script to obtain these numbers.

```bash
bash scripts/eval_wrn.sh
```

<details><summary>Line-by-line Explanation. <i>[click to expand]</i></summary>
<div>

![inference_modes](https://user-images.githubusercontent.com/2068077/76388544-9f418600-6326-11ea-9214-17356c71a066.jpg)

As before, the below just explains the above `eval_wrn.sh`. You do not need to run the following after running the previous bash script. Note the following commands are nearly identical to the corresponding train commands -- we drop the `lr`, `path-resume` flags and add `resume`, `eval`, and the `analysis` type (hard or soft inference).

```bash
# running soft inference on soft-supervised model
python main.py --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./nbdt/hierarchies/CIFAR10/graph-induced-wrn28_10_cifar10.json --tree-supervision-weight=10 --loss=SoftTreeSupLoss --eval --resume --analysis=SoftEmbeddedDecisionRules

# running hard inference on soft-supervised model
python main.py --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./nbdt/hierarchies/CIFAR10/graph-induced-wrn28_10_cifar10.json --tree-supervision-weight=10 --loss=SoftTreeSupLoss --eval --resume --analysis=HardEmbeddedDecisionRules
```
</div>
</details>

# Results



# Developing

TODO: python setup.py develop + instructions for adding new datasets, new models, etc.

## Architecture

As a sample, we've included copies of all the above bash scripts but for ResNet18. Simply add new model names or new dataset names to these bash scripts to test our method with more models or datasets.

```bash
bash scripts/generate_hierarchies_induced_resnet.sh  # this will train the network on the provided datasets if no checkpoints are found
bash scripts/train_resnet.sh
bash scripts/eval_resnet.sh
```
