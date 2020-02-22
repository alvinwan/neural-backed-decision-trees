import xmltodict
import json
import argparse
import torchvision
import os

from pathlib import Path
from utils.utils import Colors, METHODS, DATASET_TO_FOLDER_NAME
from utils.graph import generate_fname, get_parser, read_graph, get_roots, \
    get_wnids_from_dataset
from networkx.readwrite.json_graph import adjacency_data
from utils import data


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


def main():
    parser = get_parser()
    args = parser.parse_args()

    folder = DATASET_TO_FOLDER_NAME[args.dataset]
    directory = os.path.join('data', folder)

    fname = generate_fname(**vars(args))
    path = os.path.join(directory, f'{fname}.json')
    print('==> Reading from {}'.format(path))

    dataset = getattr(data, args.dataset)('./data')
    wnids = get_wnids_from_dataset(args.dataset)
    wnid_to_class = {wnid: cls for wnid, cls in zip(wnids, dataset.classes)}


    G = read_graph(path)

    num_roots = len(list(get_roots(G)))
    root = next(get_roots(G))
    tree = build_tree(G, root)
    graph = build_graph(G)

    if num_roots == 0:
        Colors.red(f'==> Found {num_roots} roots! Should be only 1.')
    else:
        Colors.green(f'==> Found just {num_roots} root.')

    generate_vis('vis/tree-template.html', tree, 'tree', fname)
    generate_vis('vis/graph-template.html', graph, 'graph', fname)


if __name__ == '__main__':
    main()
