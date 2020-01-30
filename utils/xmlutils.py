"""XML utilities

- prune tree so that only nodes matching certain criteria and their ancestors
  are kept
"""

import xml


def get_ancestors(node, node_to_parent):
    while node in node_to_parent:
        yield node
        node = node_to_parent[node]


def count_nodes(tree):
    n = 0
    for _ in tree.iter():
        n += 1
    return n


def keep_matched_nodes_and_ancestors(tree, criteria):
    node_to_parent = {child: parent for parent in tree.iter() for child in parent}
    keep = set()

    for criterion in criteria:
        node = tree.find(criterion)
        for ancestor in get_ancestors(node, node_to_parent):
            keep.add(ancestor.get('wnid'))

    if not len(keep):
        raise UserWarning('No nodes were matched. Check your criteria.')

    for node in list(tree.iter()):
        if node.get('wnid') not in keep:
            if node not in node_to_parent:
                continue
            node_to_parent[node].remove(node)

    return tree
