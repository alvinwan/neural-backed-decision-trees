# Neural-Backed Decision Trees

Run a decision tree that achieves accuracy within 1% of a recently state-of-the-art (WideResNet) neural network's accuracy on CIFAR10 and CIFAR100 and within 1.5% on TinyImagenet200.

![pipeline](https://user-images.githubusercontent.com/2068077/76384774-1ffb8480-631d-11ea-973f-7cac2a60bb10.jpg)

Per the pipeline illustration above, we (1) [generate the hierarchy](https://github.com/alvinwan/neural-backed-decision-trees#1-Hierarchies) and (2) train the neural network [with a tree supervision loss](https://github.com/alvinwan/neural-backed-decision-trees#2-Tree-Supervision-Loss). Then, we (3) [run inference](https://github.com/alvinwan/neural-backed-decision-trees#3-Inference) by featurizing images using the network backbone and running embedded decision rules.

# Getting Started

**To integrate neural-backed decision trees into your own neural network**, simply pip install this repository. (Coming soon: Use cli to generate induced-hierarchy from checkpoint. Use a simple wrapper to run classification network as nbdt. Use custom tsl with a simple function call. TODO)

**To reproduce experimental results**, start by cloning the repository and installing all requirements.

```
git clone git@github.com:alvinwan/neural-backed-decision-trees.git
cd neural-backed-decision-trees
pip install -r requirements.txt
```

To reproduce the core experimental results in our paper -- ignoring ablation studies -- simply run the following bash script:

```
bash (TODO)
```

The bash script above is equivalent to following steps in [Induced Hierarchy](https://github.com/alvinwan/neural-backed-decision-trees#Induced-Hierarchy), [Soft Tree Supervision Loss](https://github.com/alvinwan/neural-backed-decision-trees#Tree-Supervision-Loss), and [Soft Inference](https://github.com/alvinwan/neural-backed-decision-trees#Soft-Inference).

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
  python test_generated_graph.py --method=induced --induced-checkpoint=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --dataset=CIFAR100
```

## Wordnet Hierarchy

Run the following to generate and test Wordnet hierarchies for CIFAR10, CIFAR100, and TinyImagenet200. The script also downloads the NLTK Wordnet corpus.

```
bash scripts/generate_hierarchies_wordnet.sh
```

The below just explains the above `generate_hierarchies_wordnet.sh`, using CIFAR10. You do not need to run the following after running the above bash script.

```
# Generate mapping from classes to WNID. This is required for CIFAR10 and CIFAR100.
python generate_wnids.py --single-path --dataset=CIFAR10

# Generate hierarchy, using the WNIDs. This is required for all datasets: CIFAR10, CIFAR100, TinyImagenet200
python generate_hierarchy.py --single-path --dataset=CIFAR10

# Test hierarchy. This is optional but supported for all datasets. Make sure that your output ends with `==> All checks pass!`.
python test_generated_graph.py --single-path --dataset=CIFAR10
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

# 2. Tree Supervision Loss

In the below training commands, we uniformly use `--path-backbone=<path/to/checkpoint> --lr=0.01` to fine-tune instead of training from scratch. Our results using a recently state-of-the-art pretrained checkpoint (WideResNet) were fine-tuned.

Run the following bash script to fine-tune WideResNet with both hard and soft tree supervision loss on CIFAR10, CIFAR100.

```
bash scripts/train_wrn.sh
```

As before, the below just explains the above `train_wrn.sh`. You do not need to run the following after running the previous bash script.

```
# fine-tune the wrn pretrained checkpoint on CIFAR10 with hard tree supervision loss
python main.py --lr=0.01 --dataset=CIFAR10 --model=CIFAR10TreeSup --backbone=wrn28_10_cifar10 --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-backbone=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --tree-supervision-weight=10

# fine-tune the wrn pretrained checkpoint on CIFAR10 with soft tree supervision loss
python main.py --lr=0.01 --dataset=CIFAR10 --model=CIFAR10TreeBayesianSup --backbone=wrn28_10_cifar10 --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-backbone=checkpoint/ckpt-CIFAR10-wrn28_10_cifar10.pth --tree-supervision-weight=10
```

To train from scratch, use `--lr=0.01` and do not pass the `--path-backbone` flag.

# 3. Inference

![inference_modes](https://user-images.githubusercontent.com/2068077/76388544-9f418600-6326-11ea-9214-17356c71a066.jpg)

Like with the tree supervision loss variants, there are two inference variants: one is hard and one is soft. The best results in our paper, oddly enough, were obtained by running hard and soft inference *both* on the neural network supervised by a soft tree supervision loss.

Run the following bash script to obtain these numbers.

```
bash scripts/eval_wrn.sh
```

As before, the below just explains the above `eval_wrn.sh`. You do not need to run the following after running the previous bash script.

```
# running soft inference on soft-supervised model
python main.py --dataset=CIFAR10 --model=CIFAR10TreeBayesianSup --analysis=CIFAR10DecisionTreeBayesianPrior --tree-supervision-weight=10 --eval --resume --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-graph-analysis=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json

# running hard inference on soft-supervised model
python main.py --dataset=CIFAR10 --model=CIFAR10TreeBayesianSup --analysis=CIFAR10DecisionTreePrior --tree-supervision-weight=10 --eval --resume --path-graph=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json --path-graph-analysis=./data/CIFAR10/graph-induced-wrn28_10_cifar10.json
```

# Configuration

## Architecture

As a sample, we've included copies of all the above bash scripts but for ResNet10 and ResNet18.

## Checkpoints

To add new models present in [`pytorchcv`](https://github.com/osmr/imgclsmob/tree/master/pytorch),
just add a new line to `models/__init__.py` importing the models you want. For
example, we added `from pytorchcv.models.wrn_cifar import *` for CIFAR wideresnet
models. You can immediately start using this model with any of our utilities
above, including the custom tree supervision losses and extracted decision trees.

```
python main.py --model=wrn28_10_cifar10 --eval
python main.py --model=wrn28_10_cifar10 --eval --pretrained  # loads pretrained model
python main.py --model=wrn28_10_cifar10 --eval --pretrained --analysis=CIFAR10DecisionTreePrior  # run the extracted hard decision tree
python main.py --model=CIFAR10TreeSup --backbone=wrn28_10_cifar10 --batch-size=256  # train with tree supervision loss
```

To "convert" a pretrained checkpoint from a `pytorchcv` checkpoint to ours, use
the following. This will also output train accuracy.

```
python main.py --model=wrn28_10_cifar10 --pretrained --lr=0 --epochs=0
```

Then, you can use the `--resume` flag instead of `--pretrained`.
