from torchvision import datasets
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from utils import custom_datasets
import numpy as np
import operator
import xml.etree.ElementTree as ET
import os
import re

from utils.xmlutils import get_leaves, prune_single_child_nodes, \
    prune_duplicate_leaves
from sklearn.cluster import KMeans

# hacky way of "removing" layer in pytorch
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class TreeNode:
    def __init__(self, label):
        self.children = []
        self.label = label

    def print(self, ntabs=0):
        print(ntabs * ' ' + str(self.label))
        for child in self.children:
            child.print(ntabs + 3)

# creates ETree recursively from node
index_node = 0
def extendTree(cur_element, node):
    global index_node
    index_node += 1
    print("Extending to %d" % node.label)
    new_element = ET.SubElement(cur_element, "synset", {'_wnid': index_node})
    for child in node.children:
        extendTree(new_element, child)

# feature_set - np array of feature maps from each image
# tree_map - mapping (using list) from feature_set to node, which node does each image belong to
# Note - Looks like etree can only build a tree top down, so we'll use our own structure
def kmeans_cluster(feature_set, tree_map, debug=False):
    node_set = set(tree_map)   # set of nodes at this level
    n_classes = len(node_set)

    if debug:
        print("Clustering for %d" % n_classes)

    n_clusters = max(2, n_classes // 2)

    kmeans = KMeans(n_clusters = n_clusters).fit(feature_set)

    if debug:
        print("Clustering finished")
    cluster_labels = kmeans.labels_

    # count occurrences
    cluster_count = {treenode:{i: 0 for i in range(n_clusters)} for treenode in set(tree_map)}
    for cluster_label, treenode in zip(cluster_labels, tree_map):
        cluster_count[treenode][cluster_label] += 1

    if debug:
        print("Finished counting occurrences")

    cluster_belong = {} # maps which label does each node belong to. treenode -> label
    new_node_dict  = {} # maps new nodes, new node label -> new treenode
    # get cluster to which each node belongs in
    for treenode in node_set:
        cluster = max(cluster_count[treenode].items(), key=operator.itemgetter(1))[0]
        cluster_belong[treenode] = cluster

        if cluster not in new_node_dict:
            new_node_dict[cluster] = TreeNode(cluster)
        new_node_dict[cluster].children.append(treenode)

    # setup new tree map
    if debug:
        print("Finished finding belonging label")
    new_tree_map = []
    for old_node in tree_map:
        new_tree_map.append(new_node_dict[cluster_belong[old_node]])

    if debug:
        print("Finished setting up new tree map")

    return new_tree_map


import argparse

if __name__ == '__main__':

    dataset_choices = ('CIFAR10', 'CIFAR100') + custom_datasets.names

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--dataset', default='CIFAR100', choices=dataset_choices)
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size used for pulling data')
    parser.add_argument('--sample', default=1.0, type=float, help='How much of the dataset to sample')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load datasets
    print("==> Preparing data...")

    # Currently only using one set with no data augmentation for feeding into model for feature clustering
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset in custom_datasets.names:
        dataset = getattr(custom_datasets, args.dataset)
    else:
        dataset = getattr(datasets, args.dataset)

    dataset_args = ()
    if getattr(dataset, 'needs_wnid', False):
        dataset_args = (args.wnid,)

    trainset = dataset(*dataset_args, root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # load pretrained model
    print("==> Preparing model...")

    model_ft = getattr(models, args.model.lower())(pretrained=True)
    # hacky way to remove last layer
    model_ft.fc = Identity()

    if device == 'cuda':
        print("Using CUDA")
        model_ft = torch.nn.DataParallel(model_ft)

    feature_set = []
    label_set = []

    i = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        outputs = model_ft(inputs).cpu().detach().numpy()
        labels = labels.cpu().numpy()

        feature_set.append(outputs)
        label_set.append(labels)

        i += args.batch_size
        if i > args.sample * len(trainset):
            break

    feature_set = np.concatenate(feature_set)
    label_set = np.concatenate(label_set)
    print("Number of samples: %d" % len(feature_set))

    #############################################
    #   Feature maps set up, time to cluster    #
    #############################################
    # build list of nodes for each label
    new_node_dict = {}  # label -> treenode
    tree_map = []
    for label in label_set:
        if label not in new_node_dict:
            new_node_dict[label] = TreeNode(label)
        tree_map.append(new_node_dict[label])
    # pass to clustering algo
    while len(set(tree_map)) != 2:
        tree_map = kmeans_cluster(feature_set, tree_map, debug=True)

    # we now have the tree set up. Recursively create the tree, using our data struct
    root = ET.Element('root')
    for node in set(tree_map):
        node.print()
        extendTree(root, node)

    # set up labels for leaf nodes
    label_to_idx_dict = trainset.class_to_idx
    idx_to_label_dict = {label_to_idx_dict[label]: label for label in label_to_idx_dict}
    with open(os.path.join('data', args.dataset, 'wnids.txt')) as f:
        wnid_dict = [wnid.strip() for wnid in f.readlines()]
    print(len(wnid_dict))
    # TODO: add wnids
    for leaf in get_leaves(root):
        leaf.attrib['label'] = idx_to_label_dict[int(re.search(r'\d+$', leaf.tag).group())]
        leaf.attrib['wnid'] = wnid_dict[int(re.search(r'\d+$', leaf.tag).group())]

    directory = os.path.join('data', args.dataset)
    path = os.path.join(directory, 'tree-image.xml')
    tree = ET.ElementTree(element=root)
    tree = prune_single_child_nodes(tree)
    tree = prune_single_child_nodes(tree)
    tree = prune_single_child_nodes(tree)

    # prune duplicate leaves
    tree = prune_duplicate_leaves(tree)
    tree.write(path)

    print('Wrote clustered tree to {}'.format(path))
