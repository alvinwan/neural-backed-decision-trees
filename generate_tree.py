"""Generates subset_structure_released.xml from structured_released.xml

Constructs minimal tree such that all wnids contained in
tiny-imagenet-200/wnids.txt and their ancestors are included.
"""


from utils.xmlutils import keep_matched_nodes_and_ancestors, count_nodes, \
    compute_depth, compute_num_children, prune_single_child_nodes
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

def print_tree_stats(tree, name):
    print(' => {} nodes in {} tree'.format(count_nodes(tree), name))
    print(' => {} depth for {} tree'.format(compute_depth(tree), name))

print_tree_stats(tree, 'original')
tree = keep_matched_nodes_and_ancestors(tree, [
    './/synset[@wnid="{}"]'.format(wnid.strip()) for wnid in wnids
])

print_tree_stats(tree, 'matched')
# lol
tree = prune_single_child_nodes(tree)
tree = prune_single_child_nodes(tree)
tree = prune_single_child_nodes(tree)

print_tree_stats(tree, 'pruned')

num_children = compute_num_children(tree)
print(' => {} max number of children'.format(max(num_children)))

print(num_children)
path = os.path.join(directory, 'tree.xml')
tree.write(path)

print('Wrote final pruned tree to {}'.format(path))
