"""Utilities acting directly on networkx objects"""

from nbdt.utils import makeparentdirs
import networkx as nx
import json
import random
from nbdt.utils import DATASETS, METHODS, fwd
from networkx.readwrite.json_graph import node_link_data, node_link_graph
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import nbdt.models as models
import torch
import argparse
import os


def is_leaf(G, node):
    return len(G.succ[node]) == 0


def get_leaves(G, root=None):
    nodes = G.nodes if root is None else nx.descendants(G, root) | {root}
    for node in nodes:
        if is_leaf(G, node):
            yield node


def get_roots(G):
    for node in G.nodes:
        if len(G.pred[node]) == 0:
            yield node


def get_root(G):
    roots = list(get_roots(G))
    assert len(roots) == 1, f"Multiple ({len(roots)}) found"
    return roots[0]


def get_depth(G):
    def _get_depth(node):
        if not G.succ[node]:
            return 1
        return max([_get_depth(child) for child in G.succ[node]]) + 1

    return max([_get_depth(root) for root in get_roots(G)])


def get_leaf_to_path(G):
    leaf_to_path = {}
    for root in get_roots(G):
        frontier = [(root, 0, [])]
        while frontier:
            node, child_index, path = frontier.pop(0)
            path = path + [(child_index, node)]
            if is_leaf(G, node):
                leaf_to_path[node] = path
                continue
            frontier.extend([(child, i, path) for i, child in enumerate(G.succ[node])])
    return leaf_to_path


def write_graph(G, path):
    makeparentdirs(path)
    with open(str(path), "w") as f:
        json.dump(node_link_data(G), f)


def read_graph(path):
    if not os.path.exists(path):
        parent = Path(fwd()).parent
        print(f"No such file or directory: {path}. Looking in {str(parent)}")
        path = parent / path
    with open(path) as f:
        return node_link_graph(json.load(f))
