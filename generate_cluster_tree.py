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
    new_element = ET.SubElement(cur_element, "synset", {
        'wnid': str(index_node),
        'cluster': str(node.label)
    })
    for child in node.children:
        extendTree(new_element, child)


# generate random tree structure given some list of labels
def randomizeTree(root, node_list, branching_factor=2):
    print(node_list)
    if len(node_list) < branching_factor:
        for node in node_list:
            root.children.append(TreeNode(node))
        return
    perm = np.random.permutation(len(node_list))
    nodes_per_branch = len(node_list) // branching_factor
    for branch in range(branching_factor):
        root.children.append(TreeNode("random_node"))
        start, end = branch * nodes_per_branch, (branch + 1) * nodes_per_branch
        if branch == branching_factor - 1:
            end = len(node_list)
        randomizeTree(root.children[-1], node_list[perm[start:end]], branching_factor)

'''
Cluster via heuristic, create a balanced tree with some branching factor b bottom up. 
Methodology:
    1. Cluster to k clusters.
    2. For each class, find its linear weight via each cluster. sum of weights add to 1.
    3. Start with any given class. Find the b - 1 other non-used classes with the LEAST amount
        of shared weights with this given class. Essentially, their dot products is minimized.
        These classes are clustered together.
    4. Repeat step 3 until all classes clustered. Now move one step up, and repeat.
ycounts - dictionary, {label : count}
k - kmeans k
y - TreeNodes, corresponds to each x
'''
def kmeans_cluster_weighted(x, y, k=5, disjoint=False, branching_factor=2, debug=False):
    node_set = set(y)
    n_classes = len(node_set)
    ycounts = {node:0 for node in node_set}
    for node in y:
        ycounts[node] += 1

    if debug:
        print("Clustering top down for %d classes" % n_classes)

    kmeans = KMeans(n_clusters = k).fit(x)
    cluster_labels = kmeans.labels_

    # count occurrences
    cluster_count = {node:[0 for i in range(k)] for node in node_set}
    for cluster_label, node in zip(cluster_labels, y):
        cluster_count[node][cluster_label] += 1

    # find weights
    cluster_weights = {node: np.array(cluster_count[node]) / ycounts[node] for node in node_set}

    # cluster via minimum dot products. Currently manually doing branching factor of 2
    # TODO: implement higher branching factors. think of way to cluster
    node_pool = list(node_set)
    new_nodes = []
    new_node_map = {}  # maps old nodes to their new parent node.

    while len(node_pool) > 0:
        node1 = node_pool[0]
        node2 = node_pool[1]
        for i, node in enumerate(node_pool[2:]):

            if cluster_weights[node1].dot(cluster_weights[node2]) > cluster_weights[node1].dot(cluster_weights[node]):
                if disjoint:
                    node2 = node
            else:
                if not disjoint:
                    node2 = node
        # generate new node
        new_node = TreeNode('node')
        new_node.children.extend([node1, node2])
        new_node_map[node1] = new_node
        new_node_map[node2] = new_node
        new_nodes.append(new_node)

        node_pool.remove(node1)
        node_pool.remove(node2)
        if len(node_pool) == 1:   # if we are left with one element, add it to the previous node split
            new_node_map[node_pool[0]] = new_nodes[-1]
            new_nodes[-1].children.append(node_pool[0])
            node_pool.remove(node_pool[0])

    # get new y
    new_y = []
    for node in y:
        new_y.append(new_node_map[node])

    return new_y


def kmeans_cluster_topdown(x, y, branching_factor=2, debug=False):
    label_set = set(y)
    n_classes = len(label_set)

    if debug:
        print("Clustering top down for %d classes" % n_classes)

    kmeans = KMeans(n_clusters = branching_factor).fit(x)
    cluster_labels = kmeans.labels_

    # count occurrences
    cluster_count = {label:{i: 0 for i in range(branching_factor)} for label in label_set}
    for cluster_label, label in zip(cluster_labels, y):
        cluster_count[label][cluster_label] += 1


    cluster_belong = {} # maps which label does each node belong to. y label -> cluster label
    cluster_contains  = {} # maps new which y labels a cluster contains. cluster label -> [y labels]
    # get cluster to which each node belongs in
    for label in label_set:
        cluster = max(cluster_count[label].items(), key=operator.itemgetter(1))[0]
        cluster_belong[label] = cluster

        if cluster not in cluster_contains:
            cluster_contains[cluster] = []
        cluster_contains[cluster].append(label)

    # create root
    root = TreeNode('node')

    # terminate early? if only cluster to one cluster...
    if len(cluster_contains) < 2:
        if debug:
            print("Terminating early, with %d classes left" % n_classes)
        # generate random tree given nodes
        randomizeTree(root, np.array(list(label_set)), branching_factor=branching_factor)
        return root

    new_x, new_y = {cluster:[] for cluster in cluster_contains}, {cluster:[] for cluster in cluster_contains}
    for x_, y_ in zip(x, y):
        new_x[cluster_belong[y_]].append(x_)
        new_y[cluster_belong[y_]].append(y_)

    for cluster in cluster_contains:
        new_x[cluster], new_y[cluster] = np.array(new_x[cluster]), np.array(new_y[cluster])
        root.children.append(kmeans_cluster_topdown(new_x[cluster], new_y[cluster], branching_factor, debug))

    return root


# feature_set - np array of feature maps from each image
# tree_map - mapping (using list) from feature_set to node, which node does each image belong to
# Note - Looks like etree can only build a tree top down, so we'll use our own structure
def kmeans_cluster_counts(feature_set, tree_map, debug=False):
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


def kmeans_cluster_means(feature_set, tree_map, debug=False):
    """Take the average of each class to obtain a representative sample.

    Cluster these samples iteratively. Tbh, not a very good idea.
    """
    node_set = set(tree_map)   # set of nodes at this level
    n_classes = len(node_set)

    if debug:
        print("Clustering for %d" % n_classes)

    n_clusters = max(2, n_classes // 2)
    kmeans = KMeans(n_clusters = n_clusters).fit(feature_set)

    if debug:
        print("Clustering finished")
    cluster_labels = kmeans.labels_

    cluster_belong = {} # maps which label does each node belong to. treenode -> label
    new_node_dict  = {} # maps new nodes, new node label -> new treenode
    # get cluster to which each node belongs in
    for cluster_label, treenode in zip(cluster_labels, tree_map):
        cluster = cluster_label  # TODO will this cause problems?
        cluster_belong[treenode] = cluster
        if cluster not in new_node_dict:
            new_node_dict[cluster] = TreeNode(cluster)
        new_node_dict[cluster].children.append(treenode)

    new_tree_map = []
    for new_node in set(new_node_dict.values()):
        new_tree_map.append(new_node)
    return new_tree_map


def get_feature_label_counts():
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
    return feature_set, label_set


def get_feature_label_means():
    """
    Instead of obtaining one feature for each sample, obtain an average
    feature for all samples in a class.

    :return:
        feature_set array[num_classes, dim]: representatives from each class
        label_set list: labels, straightforward
    """
    d = len(trainset.classes)
    feature_set = None
    label_count = np.zeros((d, 1))

    i = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        outputs = model_ft(inputs).cpu().detach().numpy()
        labels = labels.cpu().numpy()

        if feature_set is None:
            feature_set = np.zeros((d, outputs.shape[1]))
        for label, output in zip(labels, outputs):  # feature_set[labels] += output doesn't work because labels may repeat
            feature_set[label] += output
            label_count[label] += 1

        i += args.batch_size
        if i > args.sample * len(trainset):
            break

    feature_set = feature_set / label_count
    label_set = list(range(d))
    print("Number of samples: %d" % len(feature_set))
    return feature_set, label_set


import argparse

CHOICES = ('bottom-up-count', 'bottom-up-means', 'top-down', 'weighted-disjoint', 'weighted-joint')

if __name__ == '__main__':

    dataset_choices = ('CIFAR10', 'CIFAR100') + custom_datasets.names

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--dataset', default='CIFAR100', choices=dataset_choices)
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size used for pulling data')
    parser.add_argument('--sample', default=1.0, type=float, help='How much of the dataset to sample')
    parser.add_argument('--branching-factor', default=2, type=int, help='branching factor')
    parser.add_argument('--k', default=5, type=int, help='kmeans k')
    parser.add_argument('--method', choices=CHOICES, default=CHOICES[0])
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

    if args.method == CHOICES[0] or args.method == CHOICES[2] or args.method == CHOICES[3] or args.method == CHOICES[4]:
        feature_set, label_set = get_feature_label_counts()
    elif args.method == CHOICES[1]:
        feature_set, label_set = get_feature_label_means()

    #############################################
    #   Feature maps set up, time to cluster    #
    #############################################
    tree = ET.Element('tree')
    root = ET.SubElement(tree, "synset", {
        "wnid": "-1"
    })
    if args.method == CHOICES[2]: # topdown
        TreeNoderoot = kmeans_cluster_topdown(feature_set, label_set, branching_factor=args.branching_factor, debug=True)
        TreeNoderoot.print()
        extendTree(root, TreeNoderoot)
    else:
        # build list of nodes for each label
        new_node_dict = {}  # label -> treenode
        tree_map = []
        for label in label_set:
            if label not in new_node_dict:
                new_node_dict[label] = TreeNode(label)
            tree_map.append(new_node_dict[label])
        # pass to clustering algo
        while len(set(tree_map)) != 1:
            if args.method == CHOICES[0]:
                tree_map = kmeans_cluster_counts(feature_set, tree_map, debug=True)
            elif args.method == CHOICES[1]:
                tree_map = kmeans_cluster_means(feature_set, tree_map, debug=True)
            elif args.method == 'weighted-joint' or args.method == 'weighted-disjoint':
                tree_map = kmeans_cluster_weighted(feature_set, tree_map, k=args.k,  
                                                            disjoint=args.method == 'weighted-disjoint',
                                                            branching_factor=args.branching_factor,
                                                            debug=True)
            # we now have the tree set up. Recursively create the tree, using our data struct
        for node in set(tree_map):
            node.print()
            extendTree(root, node)

    # set up labels for leaf nodes
    label_to_idx_dict = trainset.class_to_idx
    idx_to_label_dict = {label_to_idx_dict[label]: label for label in label_to_idx_dict}
    with open(os.path.join('data', args.dataset, 'wnids.txt')) as f:
        wnid_dict = [wnid.strip() for wnid in f.readlines()]

    for leaf in get_leaves(root):
        index_cluster = int(leaf.get('cluster'))
        leaf.attrib['label'] = idx_to_label_dict[index_cluster]
        leaf.attrib['wnid'] = wnid_dict[index_cluster]

    directory = os.path.join('data', args.dataset)
    path = os.path.join(directory, 'tree-image-%s.xml' % args.method)
    tree = ET.ElementTree(element=tree)
    tree = prune_single_child_nodes(tree)
    tree = prune_single_child_nodes(tree)
    tree = prune_single_child_nodes(tree)

    # prune duplicate leaves
    tree = prune_duplicate_leaves(tree)
    tree.write(path)

    print('Wrote clustered tree to {}'.format(path))
