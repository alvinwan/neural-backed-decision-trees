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
parser.add_argument('--method', choices=('shuffle', 'contract'), default='shuffle')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--contract-every', default=1, type=int,
                    help='(contract) contract every kth node in a dfs')
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


contract(get_root(G))
print_graph_stats(G, 'contracted', args)

# tree = ET.parse(args.path_tree)
# leaves = list(get_leaves(tree))
#
# leaves_data = [
#     {key: leaf.get(key) for key in leaf.keys()}
#     for leaf in leaves
# ]
# random.seed(args.seed)
# random.shuffle(leaves_data)
#
# for leaf, data in zip(leaves, leaves_data):
#     for key, value in data.items():
#         leaf.set(key, value)
#
#
path = Path(args.path_graph)
path = path \
    .with_name(f'{path.stem}-{args.method}') \
    .with_suffix(path.suffix)

# tree.write(str(path))
write_graph(G, path)
Colors.green(f'==> Wrote modified graph to {path}')
