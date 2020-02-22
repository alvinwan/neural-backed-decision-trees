"""Generates various graphs for independent node training"""

from utils.utils import DATASETS, METHODS, DATASET_TO_FOLDER_NAME
from utils.graph import build_minimal_wordnet_graph, prune_single_successor_nodes, write_graph
from utils.utils import Colors
import xml.etree.ElementTree as ET
import argparse
import os


def generate_fname(method, seed=0, branching_factor=2, **kwargs):
    fname = f'tree-{method}'
    if method == 'random':
        if seed != 0:
            fname += f'-seed{seed}'
        if branching_factor != 2:
            fname += f'-branch{branching_factor}'
    return fname


def print_graph_stats(G, name, args):
    num_children = [len(succ) for succ in G.succ]
    print('[{}] \t Nodes: {} \t Depth: {} \t Max Children: {}'.format(
        name,
        len(G.nodes),
        0,  # compute_depth(tree),
        max(num_children)))
    if args.verbose:
        print('[{}]'.format(name), num_children)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
        help='Must be a folder data/{dataset} containing a wnids.txt',
        choices=DATASETS,
        default='CIFAR10')
    parser.add_argument('--method', choices=METHODS,
        help='structure_released.xml apparently is missing many CIFAR100 classes. '
        'As a result, pruning does not work for CIFAR100. Random will randomly '
        'join clusters together, iteratively, to make a roughly-binary tree.',
        default='build')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--branching-factor', type=int, default=2)
    parser.add_argument('--extra-roots', action='store_true',
                        help='If should include all parents of each synset '
                        'as extra roots.')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    folder = DATASET_TO_FOLDER_NAME[args.dataset]
    directory = os.path.join('data', folder)
    with open(os.path.join(directory, 'wnids.txt')) as f:
        wnids = [wnid.strip() for wnid in f.readlines()]

    # elif args.method == 'random':
    #     tree = build_random_tree(
    #         wnids, seed=args.seed, branching_factor=args.branching_factor)
    # else:
    #     raise NotImplementedError(f'Method "{args.method}" not yet handled.')

    G = build_minimal_wordnet_graph(wnids)
    print_graph_stats(G, 'matched', args)

    G = prune_single_successor_nodes(G)
    print_graph_stats(G, 'pruned', args)

    fname = generate_fname(**vars(args))
    path = os.path.join(directory, f'{fname}.json')
    write_graph(G, path)

    Colors.green('==> Wrote tree to {}'.format(path))

    # wnids_set = {node.get('wnid') for node in tree.iter()}
    # assert all(wnid.strip() in wnids_set for wnid in wnids), \
    #     [wnid.strip() for wnid in wnids if wnid.strip() not in wnids_set]


if __name__ == '__main__':
    main()
