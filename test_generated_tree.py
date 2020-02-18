from utils.xmlutils import get_leaves
from utils.utils import DATASETS, METHODS, DATASET_TO_FOLDER_NAME
import xml.etree.ElementTree as ET
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
    help='Must be a folder data/{dataset} containing a wnids.txt',
    choices=DATASETS,
    default='CIFAR10')
parser.add_argument('--method', choices=METHODS,
    help='structure_released.xml apparently is missing many CIFAR100 classes. '
    'As a result, pruning does not work for CIFAR100. Random will randomly '
    'join clusters together, iteratively, to make a roughly-binary tree.',
    default='build')

args = parser.parse_args()

folder = DATASET_TO_FOLDER_NAME[args.dataset]
directory = os.path.join('data', folder)
with open(os.path.join(directory, 'wnids.txt')) as f:
    wnids = f.readlines()


def get_seen_wnids(wnid_set, nodes):
    leaves_seen = set()
    for leaf in nodes:
        id_ = leaf.get('wnid')
        if id_ in wnid_set:
            wnid_set.remove(id_)
        if id_ in leaves_seen:
            pass
        leaves_seen.add(id_)
    return leaves_seen


def match_wnid_leaves(wnids, tree, tree_name):
    wnid_set = set()
    for wnid in wnids:
        wnid_set.add(wnid.strip())

    leaves_seen = get_seen_wnids(wnid_set, get_leaves(tree))
    return leaves_seen, wnid_set


def match_wnid_nodes(wnids, tree, tree_name):
    wnid_set = {wnid.strip() for wnid in wnids}
    leaves_seen = get_seen_wnids(wnid_set, tree.iter())

    return leaves_seen, wnid_set


def print_stats(leaves_seen, wnid_set, tree_name, node_type):
    print(f"[{tree_name}] \t {node_type}: {len(leaves_seen)} \t WNIDs missing from {node_type}: {len(wnid_set)}")
    if len(wnid_set):
        print(f"\033[93m==> Warning: WNIDs in wnid.txt are missing from {tree_name} {node_type}\033[0m")



# tree_name = 'structure_released.xml'
# tree = ET.parse(tree_name)
# match_wnid_leaves(wnids, tree, tree_name)
# match_wnid_nodes(wnids, tree, tree_name)
#
# print('='*30)

tree_name = os.path.join(directory, 'tree-{}.xml'.format(args.method))
tree = ET.parse(tree_name)

leaves_seen, wnid_set1 = match_wnid_leaves(wnids, tree, tree_name)
print_stats(leaves_seen, wnid_set1, tree_name, 'leaves')

leaves_seen, wnid_set2 = match_wnid_nodes(wnids, tree, tree_name)
print_stats(leaves_seen, wnid_set2, tree_name, 'nodes')

if len(wnid_set1) == len(wnid_set2) == 0:
    print("\033[92m==> All checks pass!\033[0m")
