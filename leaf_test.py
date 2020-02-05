from utils.xmlutils import get_leaves
import xml.etree.ElementTree as ET
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
    help='Must be a folder data/{dataset} containing a wnids.txt',
    choices=('tiny-imagenet-200', 'cifar10', 'CIFAR100'),
    default='cifar10')

args = parser.parse_args()
tree = ET.parse('structure_released.xml')

directory = os.path.join('data', args.dataset)
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

tree_name = 'structure_released.xml'
tree = ET.parse(tree_name)
match_wnid_leaves(wnids, tree, tree_name)
match_wnid_nodes(wnids, tree, tree_name)

print('='*30)

tree_name = os.path.join(directory, 'tree.xml')
tree = ET.parse(tree_name)
match_wnid_leaves(wnids, tree, tree_name)
match_wnid_nodes(wnids, tree, tree_name)