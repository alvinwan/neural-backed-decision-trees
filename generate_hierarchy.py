"""Generates various graphs for independent node training"""

from nbdt.utils import DATASETS, METHODS, DATASET_TO_FOLDER_NAME
from nbdt.graph import build_minimal_wordnet_graph, build_random_graph, \
    prune_single_successor_nodes, write_graph, get_wnids, generate_fname, \
    get_parser, get_wnids_from_dataset, get_directory, get_graph_path_from_args, \
    augment_graph, get_depth, build_induced_graph
from nbdt.utils import Colors
import xml.etree.ElementTree as ET
import argparse
import os


def print_graph_stats(G, name, args):
    num_children = [len(succ) for succ in G.succ]
    print('[{}] \t Nodes: {} \t Depth: {} \t Max Children: {}'.format(
        name,
        len(G.nodes),
        get_depth(G),
        max(num_children)))


def assert_all_wnids_in_graph(G, wnids):
    assert all(wnid.strip() in G.nodes for wnid in wnids), [
        wnid for wnid in wnids if wnid not in G.nodes
    ]


def main():
    parser = get_parser()

    args = parser.parse_args()
    wnids = get_wnids_from_dataset(args.dataset)

    if args.method == 'wordnet':
        G = build_minimal_wordnet_graph(wnids, args.single_path)
    elif args.method == 'random':
        G = build_random_graph(wnids, seed=args.seed, branching_factor=args.branching_factor)
    elif args.method == 'induced':
        G = build_induced_graph(wnids,
            checkpoint=args.induced_checkpoint,
            linkage=args.induced_linkage,
            affinity=args.induced_affinity,
            branching_factor=args.branching_factor)
    else:
        raise NotImplementedError(f'Method "{args.method}" not yet handled.')
    print_graph_stats(G, 'matched', args)
    assert_all_wnids_in_graph(G, wnids)

    if not args.no_prune:
        G = prune_single_successor_nodes(G)
        print_graph_stats(G, 'pruned', args)
        assert_all_wnids_in_graph(G, wnids)

    if args.extra > 0:
        G, n_extra, n_imaginary = augment_graph(G, args.extra, True)
        print(f'[extra] \t Extras: {n_extra} \t Imaginary: {n_imaginary}')
        print_graph_stats(G, 'extra', args)
        assert_all_wnids_in_graph(G, wnids)

    path = get_graph_path_from_args(args)
    write_graph(G, path)

    Colors.green('==> Wrote tree to {}'.format(path))


if __name__ == '__main__':
    main()
