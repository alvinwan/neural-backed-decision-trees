"""Generates subset_structure_released.xml from structured_released.xml

Constructs minimal tree such that all wnids contained in
tiny-imagenet-200/wnids.txt and their ancestors are included.
"""


from utils.xmlutils import keep_matched_nodes_and_ancestors, count_nodes, \
    compute_depth_bfs
import xml.etree.ElementTree as ET


tree = ET.parse('structure_released.xml')

with open('tiny-imagenet-200/wnids.txt') as f:
    wnids = f.readlines()

print(' => {} nodes in original tree'.format(count_nodes(tree)))
print(' => {} depth for original tree'.format(compute_depth_bfs(tree)))
tree = keep_matched_nodes_and_ancestors(tree, [
    './/synset[@wnid="{}"]'.format(wnid.strip()) for wnid in wnids
])
print(' => {} nodes in pruned tree'.format(count_nodes(tree)))
print(' => {} depth for pruned tree'.format(compute_depth_bfs(tree)))
tree.write('subset_structure_released.xml')
