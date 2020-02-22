"""Generates various graphs for independent node training"""

from utils.utils import DATASETS, METHODS, DATASET_TO_FOLDER_NAME
from utils.graph import build_minimal_wordnet_graph, \
    prune_single_successor_nodes, write_graph, get_wnids
from utils.utils import Colors
import xml.etree.ElementTree as ET
import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
        help='Must be a folder data/{dataset} containing a wnids.txt',
        choices=DATASETS,
        default='CIFAR10')
    return parser


def get_wnids_from_directory(directory):
    return get_wnids(os.path.join(directory, 'wnids.txt'))


def generate_fname(**kwargs):
    fname = f'graph-wordnet'
    return fname


def print_graph_stats(G, name, args):
    num_children = [len(succ) for succ in G.succ]
    print('[{}] \t Nodes: {} \t Depth: {} \t Max Children: {}'.format(
        name,
        len(G.nodes),
        0,  # compute_depth(tree),
        max(num_children)))


def assert_all_wnids_in_graph(G, wnids):
    assert all(wnid.strip() in G.nodes for wnid in wnids), [
        wnid for wnid in wnids if wnid not in G.nodes
    ]


def main():
    parser = get_parser()
    args = parser.parse_args()

    folder = DATASET_TO_FOLDER_NAME[args.dataset]
    directory = os.path.join('data', folder)
    wnids = get_wnids_from_directory(directory)

    G = build_minimal_wordnet_graph(wnids)
    print_graph_stats(G, 'matched', args)
    assert_all_wnids_in_graph(G, wnids)

    G = prune_single_successor_nodes(G)
    print_graph_stats(G, 'pruned', args)
    assert_all_wnids_in_graph(G, wnids)

    fname = generate_fname(**vars(args))
    path = os.path.join(directory, f'{fname}.json')
    write_graph(G, path)

    Colors.green('==> Wrote tree to {}'.format(path))


if __name__ == '__main__':
    main()
