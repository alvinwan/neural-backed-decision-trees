"""Generates various graphs for independent node training"""

from nbdt.utils import DATASETS, METHODS, Colors
from nbdt.graph import build_minimal_wordnet_graph, build_random_graph, \
    prune_single_successor_nodes, write_graph, get_wnids, generate_fname, \
    get_parser, get_wnids_from_dataset, get_directory, get_graph_path_from_args, \
    augment_graph, get_depth, build_induced_graph, read_graph, get_leaves, \
    get_roots
from nbdt import data
from networkx.readwrite.json_graph import adjacency_data
from pathlib import Path
import os
import json
import torchvision


############
# GENERATE #
############


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


def generate_hierarchy(args):
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


########
# TEST #
########


def get_seen_wnids(wnid_set, nodes):
    leaves_seen = set()
    for leaf in nodes:
        if leaf in wnid_set:
            wnid_set.remove(leaf)
        if leaf in leaves_seen:
            pass
        leaves_seen.add(leaf)
    return leaves_seen


def match_wnid_leaves(wnids, G, tree_name):
    wnid_set = set()
    for wnid in wnids:
        wnid_set.add(wnid.strip())

    leaves_seen = get_seen_wnids(wnid_set, get_leaves(G))
    return leaves_seen, wnid_set


def match_wnid_nodes(wnids, G, tree_name):
    wnid_set = {wnid.strip() for wnid in wnids}
    leaves_seen = get_seen_wnids(wnid_set, G.nodes)

    return leaves_seen, wnid_set


def print_stats(leaves_seen, wnid_set, tree_name, node_type):
    print(f"[{tree_name}] \t {node_type}: {len(leaves_seen)} \t WNIDs missing from {node_type}: {len(wnid_set)}")
    if len(wnid_set):
        Colors.red(f"==> Warning: WNIDs in wnid.txt are missing from {tree_name} {node_type}")


def test_hierarchy(args):
    wnids = get_wnids_from_dataset(args.dataset)
    path = get_graph_path_from_args(args)
    print('==> Reading from {}'.format(path))

    G = read_graph(path)

    G_name = Path(path).stem

    leaves_seen, wnid_set1 = match_wnid_leaves(wnids, G, G_name)
    print_stats(leaves_seen, wnid_set1, G_name, 'leaves')

    leaves_seen, wnid_set2 = match_wnid_nodes(wnids, G, G_name)
    print_stats(leaves_seen, wnid_set2, G_name, 'nodes')

    num_roots = len(list(get_roots(G)))
    if num_roots == 1:
        Colors.green('Found just 1 root.')
    else:
        Colors.red(f'Found {num_roots} roots. Should be only 1.')

    if len(wnid_set1) == len(wnid_set2) == 0 and num_roots == 1:
        Colors.green("==> All checks pass!")
    else:
        Colors.red('==> Test failed')


#######
# VIS #
#######


def build_tree(G, root, parent='null'):
    return {
        'name': root,
        'label': G.nodes[root].get('label', ''),
        'parent': parent,
        'children': [build_tree(G, child, root) for child in G.succ[root]]
    }


def build_graph(G):
    return {
        'nodes': [{
            'name': wnid,
            'label': G.nodes[wnid].get('label', ''),
            'id': wnid
        } for wnid in G.nodes],
        'links': [{
            'source': u,
            'target': v
        } for u, v in G.edges]
    }


def generate_vis(path_template, data, name, fname):
    with open(path_template) as f:
        html = f.read().replace(
            "'TREE_DATA_CONSTANT_TO_BE_REPLACED'",
            json.dumps(data))

    os.makedirs('out', exist_ok=True)
    path_html = f'out/{fname}-{name}.html'
    with open(path_html, 'w') as f:
        f.write(html)

    Colors.green('==> Wrote HTML to {}'.format(path_html))


def generate_hierarchy_vis(args):
    path = get_graph_path_from_args(args)
    print('==> Reading from {}'.format(path))

    G = read_graph(path)

    roots = list(get_roots(G))
    num_roots = len(roots)
    root = next(get_roots(G))
    tree = build_tree(G, root)
    graph = build_graph(G)

    if num_roots > 1:
        Colors.red(f'Found {num_roots} roots! Should be only 1: {roots}')
    else:
        print(f'Found just {num_roots} root.')

    fname = generate_fname(**vars(args)).replace('graph-', '', 1)
    generate_vis('nbdt/templates/tree-template.html', tree, 'tree', fname)
    generate_vis('nbdt/templates/graph-template.html', graph, 'graph', fname)


def main():
    parser = get_parser()
    args = parser.parse_args()

    generate_hierarchy(args)
    test_hierarchy(args)
    generate_hierarchy_vis(args)


if __name__ == '__main__':
    main()
