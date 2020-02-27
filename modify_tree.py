"""Randomly reshuffle leaves"""

from pathlib import Path
from utils.graph import read_graph, get_root, is_leaf, write_graph
from generate_graph import print_graph_stats
from utils.utils import Colors
import networkx as nx
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path-graph', default='./data/CIFAR10/graph-wordnet-single.json')
args = parser.parse_args()


G = read_graph(args.path_graph)

def contract(node, keep=True):
    global G
    if is_leaf(G, node):
        return False
    for child in G.succ[node]:
        contractable = contract(child, not keep)
        if contractable:
            G = nx.contracted_nodes(G, node, child, self_loops=False)
    return keep


print_graph_stats(G, 'original', args)
contract(get_root(G))
print_graph_stats(G, 'contracted', args)

path = Path(args.path_graph)
path = path \
    .with_name(f'{path.stem}-contract') \
    .with_suffix(path.suffix)

write_graph(G, path)
Colors.green(f'==> Wrote modified graph to {path}')
