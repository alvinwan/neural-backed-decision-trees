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

# args = parser.parse_args()
# tree = ET.parse('structure_released.xml')

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

    print("Total unique leaves in %s: %d" % (tree_name, len(leaves_seen)))
    print("Number of wnids in wnids.txt not found in leaves of %s: %d" % (tree_name, len(wnid_set)))


def match_wnid_nodes(wnids, tree, tree_name):
    wnid_set = {wnid.strip() for wnid in wnids}
    leaves_seen = get_seen_wnids(wnid_set, tree.iter())

    print("Total unique nodes in %s: %d" % (tree_name, len(leaves_seen)))
    print("Number of wnids in wnids.txt not found in nodes of %s: %d" % (tree_name, len(wnid_set)))

# tree_name = 'structure_released.xml'
# tree = ET.parse(tree_name)
# match_wnid_leaves(wnids, tree, tree_name)
# match_wnid_nodes(wnids, tree, tree_name)
#
# print('='*30)

tree_name = os.path.join(directory, 'tree-{}.xml'.format(args.method))
tree = ET.parse(tree_name)
match_wnid_leaves(wnids, tree, tree_name)
match_wnid_nodes(wnids, tree, tree_name)
