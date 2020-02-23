import networkx as nx
import json
from nltk.corpus import wordnet as wn
from utils.utils import DATASETS, DATASET_TO_FOLDER_NAME
from networkx.readwrite.json_graph import node_link_data, node_link_graph
import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
        help='Must be a folder data/{dataset} containing a wnids.txt',
        choices=DATASETS,
        default='CIFAR10')
    return parser


def generate_fname(**kwargs):
    fname = f'graph-wordnet'
    return fname


def get_directory(dataset, root='./data'):
    folder = DATASET_TO_FOLDER_NAME[dataset]
    return os.path.join(root, folder)


def get_wnids_from_dataset(dataset, root='./data'):
    directory = get_directory(dataset, root)
    return get_wnids(os.path.join(directory, 'wnids.txt'))


def get_wnids(path_wnids):
    with open(path_wnids) as f:
        wnids = [wnid.strip() for wnid in f.readlines()]
    return wnids


def get_graph_path_from_args(args):
    fname = generate_fname(**vars(args))
    directory = get_directory(args.dataset)
    path = os.path.join(directory, f'{fname}.json')
    return path


def synset_to_wnid(synset):
    return f'{synset.pos()}{synset.offset():08d}'


def wnid_to_synset(wnid):
    offset = int(wnid[1:])
    pos = wnid[0]
    return wn.synset_from_pos_and_offset(wnid[0], offset)


def synset_to_name(synset):
    return synset.name().split('.')[0]


def get_leaves(G, root=None, include_root=False):
    nodes = G.nodes if root is None else nx.descendants(G, root) | {root}
    for node in nodes:
        if len(G.succ[node]) == 0:
            yield node


def get_non_leaves(G):
    for node in G.nodes:
        if len(G.succ[node]) > 0:
            yield node


def get_roots(G):
    for node in G.nodes:
        if len(G.pred[node]) == 0:
            yield node


def set_node_label(G, synset):
    nx.set_node_attributes(G, {
        synset_to_wnid(synset): synset_to_name(synset)
    }, 'label')


def build_minimal_wordnet_graph(wnids):
    G = nx.DiGraph()

    for wnid in wnids:
        G.add_node(wnid)
        synset = wnid_to_synset(wnid)
        set_node_label(G, synset)

        if wnid == 'n10129825':  # hardcode 'girl' to be a child of 'female', not 'woman'
            G.add_edge('n09619168', 'n10129825')
            continue

        hypernyms = [synset]
        while hypernyms:
            current = hypernyms.pop(0)
            set_node_label(G, current)
            for hypernym in current.hypernyms():
                G.add_edge(synset_to_wnid(hypernym), synset_to_wnid(current))
                hypernyms.append(hypernym)

        children = [(key, wnid_to_synset(key).name()) for key in G.succ[wnid]]
        assert len(children) == 0, \
            f'Node {wnid} ({synset.name()}) is not a leaf. Children: {children}'
    return G


def prune_single_successor_nodes(G):
    for node in G.nodes:
        if len(G.succ[node]) == 1:
            succ = list(G.succ[node])[0]
            G = nx.contracted_nodes(G, succ, node, self_loops=False)
    return G


def write_graph(G, path):
    with open(path, 'w') as f:
        json.dump(node_link_data(G), f)


def read_graph(path):
    with open(path) as f:
        return node_link_graph(json.load(f))
