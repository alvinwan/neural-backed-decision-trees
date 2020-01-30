"""Generates subset_structure_released.xml from structured_released.xml

Constructs minimal tree such that all wnids contained in
tiny-imagenet-200/wnids.txt and their ancestors are included.
"""


from utils.xmlutils import keep_matched_nodes_and_ancestors, count_nodes, \
    compute_depth, compute_num_children, prune_single_child_nodes
import xml.etree.ElementTree as ET


tree = ET.parse('structure_released.xml')

with open('tiny-imagenet-200/wnids.txt') as f:
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
tree.write('subset_structure_released.xml')
