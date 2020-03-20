# Neural-Backed Decision Trees

Run a decision tree that achieves accuracy within 1% of a recently state-of-the-art (WideResNet) neural network's accuracy on CIFAR10 and CIFAR100 and within 1.5% on TinyImagenet200.

![pipeline](https://user-images.githubusercontent.com/2068077/76384774-1ffb8480-631d-11ea-973f-7cac2a60bb10.jpg)

Per the pipeline illustration above, we (1) [generate the hierarchy](https://github.com/alvinwan/neural-backed-decision-trees#1-Hierarchies) and (2) train the neural network [with a tree supervision loss](https://github.com/alvinwan/neural-backed-decision-trees#2-Tree-Supervision-Loss). Then, we (3) [run inference](https://github.com/alvinwan/neural-backed-decision-trees#3-Inference) by featurizing images using the network backbone and running embedded decision rules.

# Getting Started

**To integrate neural-backed decision trees into your own neural network**, simply pip install this repository. (Coming soon: Download zip with prebuilt induced hierarchies for certain datasets. If no wnids, generate fake ones. Attempt to get path_graph, dataset.classes. Use a simple wrapper to run classification network as nbdt. Use custom tsl with a simple function call. For a new dataset, use cli to generate induced-hierarchy from checkpoint. TODO)

```
pip install nbdt
wget ...  # CIFAR10 ResNet model
nbdt --method=induced ...  # generate induced hierarchy
```

```
from nbdt.loss import SoftTreeSupLoss

criterion = # original loss function
criterion = SoftTreeSupLoss.with_defaults('CIFAR10', criterion=criterion)
```

```
from nbdt.model import SoftNBDT

model = # original model
model = SoftNBDT.with_defaults('CIFAR10', model=model)
```

**To reproduce experimental results**, start by cloning the repository and installing all requirements.

```
git clone git@github.com:alvinwan/neural-backed-decision-trees.git
cd neural-backed-decision-trees
pip install -r requirements.txt
```

To reproduce the core experimental results in our paper -- ignoring ablation studies -- simply run the following bash scripts:

```
bash scripts/generate_hierarchies_induced_wrn.sh
bash scripts/train_wrn.sh
bash scripts/eval_wrn.sh
```

The bash scripts above are explained in more detail in [Induced Hierarchy](https://github.com/alvinwan/neural-backed-decision-trees#Induced-Hierarchy), [Soft Tree Supervision Loss](https://github.com/alvinwan/neural-backed-decision-trees#Tree-Supervision-Loss), and [Soft Inference](https://github.com/alvinwan/neural-backed-decision-trees#Soft-Inference).

# 1. Hierarchies

## Induced Hierarchy

Run the following to generate and test induced hierarchies for CIFAR10, CIFAR100, based off of the WideResNet model. The script also downloads pretrained WideResNet models.

```
bash scripts/generate_hierarchies_induced_wrn.sh
```

![induced_structure](https://user-images.githubusercontent.com/2068077/76388304-0e6aaa80-6326-11ea-8c9b-6d08cb89fafe.jpg)


The below just explains the above `generate_hierarches_induced.sh`, using CIFAR10. You do not need to run the following after running the above bash script. Note that the following commands can be rerun with different checkpoints from different architectures, to produce different hierarchies.

```
# Step A. Download and evaluate pre-trained weights for WideResNet on CIFAR10.
python main.py --eval --pretrained --model=wrn28_10_cifar10 --dataset=CIFAR10

# Step B through D. Generate induced hierarchies, using the pretrained checkpoints
python generate_hierarchy.py --method=induced --induced-checkpoint=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --dataset=CIFAR10

# Test hierarchy
python test_generated_hierarchy.py --method=induced --induced-checkpoint=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --dataset=CIFAR10
```

## Wordnet Hierarchy

Run the following to generate and test Wordnet hierarchies for CIFAR10, CIFAR100, and TinyImagenet200. The script also downloads the NLTK Wordnet corpus.

```
bash scripts/generate_hierarchies_wordnet.sh
```

The below just explains the above `generate_hierarchies_wordnet.sh`, using CIFAR10. You do not need to run the following after running the above bash script.

```
# Generate mapping from classes to WNID. This is required for CIFAR10 and CIFAR100.
python generate_wnids.py --dataset=CIFAR10

# Generate hierarchy, using the WNIDs. This is required for all datasets: CIFAR10, CIFAR100, TinyImagenet200
python generate_hierarchy.py --single-path --dataset=CIFAR10

# Test hierarchy. This is optional but supported for all datasets. Make sure that your output ends with `==> All checks pass!`.
python test_generated_hierarchy.py --single-path --dataset=CIFAR10
```

## Random Hierarchy

Use `--method=random` to randomly generate a binary-ish hierarchy. Additionally,
random trees feature two more flags:

- `--seed` to generate random leaf orderings. Use `seed=-1` to *not* randomly shuffle leaves.
- `--branching-factor` to generate trees with different branching factors. Setting branching factor to the number of classes is a nice sanity check. We used this for debugging, ourselves.

For example, to generate a sanity check hierarchy for CIFAR10, use

```
python generate_hierarchy.py --seed=-1 --branching-factor=10 --single-path --dataset=CIFAR10
```

## Visualize Hierarchy

Run the visualization generation script to obtain both the JSON representing
the hierarchy and the HTML file containing a d3 visualization.

```
python generate_vis.py --single-path
```

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

# 2. Tree Supervision Loss

In the below training commands, we uniformly use `--path-resume=<path/to/checkpoint> --lr=0.01` to fine-tune instead of training from scratch. Our results using a recently state-of-the-art pretrained checkpoint (WideResNet) were fine-tuned.

Run the following bash script to fine-tune WideResNet with both hard and soft tree supervision loss on CIFAR10, CIFAR100.

```
bash scripts/train_wrn.sh
```

As before, the below just explains the above `train_wrn.sh`. You do not need to run the following after running the previous bash script.

```
# fine-tune the wrn pretrained checkpoint on CIFAR10 with hard tree supervision loss
python main.py --lr=0.01 --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./nbdt/hierarchies/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-resume=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --tree-supervision-weight=10 --loss=HardTreeSupLoss

# fine-tune the wrn pretrained checkpoint on CIFAR10 with soft tree supervision loss
python main.py --lr=0.01 --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./nbdt/hierarchies/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-resume=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --tree-supervision-weight=10 --loss=SoftTreeSupLoss
```

To train from scratch, use `--lr=0.1` and do not pass the `--path-resume` flag. We fine-tune WideResnet on CIFAR10, CIFAR100, but where the baseline neural network accuracy is reproducible, we train from scratch.

# 3. Inference

![inference_modes](https://user-images.githubusercontent.com/2068077/76388544-9f418600-6326-11ea-9214-17356c71a066.jpg)

Like with the tree supervision loss variants, there are two inference variants: one is hard and one is soft. The best results in our paper, oddly enough, were obtained by running hard and soft inference *both* on the neural network supervised by a soft tree supervision loss.

Run the following bash script to obtain these numbers.

```
bash scripts/eval_wrn.sh
```

As before, the below just explains the above `eval_wrn.sh`. You do not need to run the following after running the previous bash script. Note the following commands are nearly identical to the corresponding train commands -- we drop the `lr`, `path-resume` flags and add `resume`, `eval`, and the `analysis` type (hard or soft inference).

```
# running soft inference on soft-supervised model
python main.py --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./nbdt/hierarchies/CIFAR10/graph-induced-wrn28_10_cifar10.json --tree-supervision-weight=10 --loss=SoftTreeSupLoss --eval --resume --analysis=SoftEmbeddedDecisionRules

# running hard inference on soft-supervised model
python main.py --dataset=CIFAR10 --model=wrn28_10_cifar10 --path-graph=./nbdt/hierarchies/CIFAR10/graph-induced-wrn28_10_cifar10.json --tree-supervision-weight=10 --loss=SoftTreeSupLoss --eval --resume --analysis=HardEmbeddedDecisionRules
```

# Configuration

## Architecture

As a sample, we've included copies of all the above bash scripts but for ResNet10 and ResNet18. Simply add new model names or new dataset names to these bash scripts to test our method with more models or datasets.

```
bash scripts/generate_hierarchies_induced_resnet.sh  # this will train the network on the provided datasets if no checkpoints are found
bash scripts/train_resnet.sh
bash scripts/eval_resnet.sh
```

## Importing Other Models (`torchvision`, `pytorchcv`)

To add new models present in [`pytorchcv`](https://github.com/osmr/imgclsmob/tree/master/pytorch),
just add a new line to `models/__init__.py` importing the models you want. For
example, we added `from pytorchcv.models.wrn_cifar import *` for CIFAR wideresnet
models.

To add new models present in [`torchvision`](https://pytorch.org/docs/stable/torchvision/models.html), likewise just add a new line to `models/__init__.py`. For example, to import all, use `from torchvision.models import *`.

You can immediately start using these models with any of our utilities
above, including the custom tree supervision losses and extracted decision trees.

```
python main.py --model=wrn28_10_cifar10 --eval
python main.py --model=wrn28_10_cifar10 --eval --pretrained  # loads pretrained model
python main.py --model=wrn28_10_cifar10 --eval --pretrained --analysis=HardEmbeddedDecisionRules  # run the extracted hard decision tree
python main.py --model=wrn28_10_cifar10 --loss=HardTreeSupLoss --batch-size=256  # train with tree supervision loss
```

To download a pretrained checkpoint for a `pytorchcv` model, simply add the
`--pretrained` flag.

```
python main.py --model=wrn28_10_cifar10 --pretrained
```

If a pretrained checkpoint is already downloaded to disk, pass the path
using `--path-checkpoint`

```
python main.py --model=wrn28_10_cifar10 --path-checkpoint=...
```
