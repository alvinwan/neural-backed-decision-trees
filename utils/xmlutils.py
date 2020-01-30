"""XML utilities

- prune tree so that only nodes matching certain criteria and their ancestors
  are kept
"""

import xml
from collections import defaultdict, Counter


def find_attribute_contains(tree, attribute, contains):
    seen = set()
    nodes = []
    for node in tree.iter():
        if attribute not in node.keys():
            continue
        if contains in node.get(attribute):
            wnid = node.get('wnid')
            if wnid not in seen:
                seen.add(wnid)
                nodes.append(node)
    return nodes


def get_ancestors(node, node_to_parent):
    while node in node_to_parent:
        yield node
        node = node_to_parent[node]


def count_nodes(tree):
    n = 0
    for _ in tree.iter():
        n += 1
    return n


def bfs(tree, func):
    root = tree.getroot()
    frontier = [root]

    while frontier:
        node = frontier.pop(0)
        for child in node.getchildren():
            func(node, child)
            frontier.append(child)


def compute_depth(tree):
    root = tree.getroot()
    root.set('depth', 0)

    depth_max = 0
    def func(node, child):
        nonlocal depth_max
        depth = node.get('depth') + 1
        if depth > depth_max:
            depth_max = depth
        child.set('depth', depth)

    bfs(tree, func)

    for node in tree.iter():
        node.attrib.pop('depth')

    return depth_max


def compute_num_children(tree, condition = lambda count: count > 0):
    num_children = []
    for node in tree.iter():
        num_children.append(len(node.getchildren()))
    return Counter(filter(condition, num_children))


def keep_matched_nodes_and_ancestors(tree, criteria):
    node_to_parent = {child: parent for parent in tree.iter() for child in parent}
    keep = defaultdict(lambda: 0)

    for criterion in criteria:
        node = tree.find(criterion)
        for ancestor in get_ancestors(node, node_to_parent):
            keep[ancestor.get('wnid')] += 1

    if not len(keep):
        raise UserWarning('No nodes were matched. Check your criteria.')

    for node in list(tree.iter()):
        if node.get('wnid') not in keep:
            if node not in node_to_parent:
                continue
            node_to_parent[node].remove(node)

    n_redundant_nodes = sum(count == len(criteria) for count in keep.values())
    assert n_redundant_nodes <= 1, \
        'There are multiple nodes that all leaves share. Weird'

    return tree


def prune_single_child_nodes(tree):
    node_to_parent = {child: parent for parent in tree.iter() for child in parent}

    for node in tree.iter():
        if node not in node_to_parent:
            continue
        children = node.getchildren()
        if len(children) == 1:
            parent = node_to_parent[node]
            parent.remove(node)
            parent.insert(0, children[0])

    return tree


def get_leaves(tree):
    leaves = []
    for node in tree.iter():
        if not node.getchildren():
            yield node
