import networkx as nx
import json
import random
from nltk.corpus import wordnet as wn
from utils.utils import DATASETS, DATASET_TO_FOLDER_NAME
from networkx.readwrite.json_graph import node_link_data, node_link_graph
import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help='Must be a folder data/{dataset} containing a wnids.txt',
        choices=DATASETS,
        default='CIFAR10')
    parser.add_argument(
        '--extra',
        type=int,
        default=0,
        help='Percent extra nodes to add to the tree. If 100, the number of '
        'nodes in tree are doubled. Note this is an integral percent.')
    parser.add_argument(
        '--single-path',
        action='store_true',
        help='Ensure every leaf only has one path from the root.')
    parser.add_argument('--no-prune', action='store_true', help='Do not prune.')
    parser.add_argument('--fname', type=str,
        help='Override all settings and just provide a path to a graph')
    return parser


def generate_fname(extra=0, no_prune=False, fname='', single_path=False, **kwargs):
    if fname:
        return fname

    fname = f'graph-wordnet'
    if extra > 0:
        fname += f'-extra{extra}'
    if no_prune:
        fname += '-noprune'
    if single_path:
        fname += '-single'
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

    try:
        return wn.synset_from_pos_and_offset(wnid[0], offset)
    except:
        return FakeSynset(wnid)


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


def get_root(G):
    roots = list(get_roots(G))
    assert len(roots) == 1
    return roots[0]


def set_node_label(G, synset):
    nx.set_node_attributes(G, {
        synset_to_wnid(synset): synset_to_name(synset)
    }, 'label')


def build_minimal_wordnet_graph(wnids, single_path=False):
    G = nx.DiGraph()

    for wnid in wnids:
        G.add_node(wnid)
        synset = wnid_to_synset(wnid)
        set_node_label(G, synset)

        if wnid == 'n10129825':  # hardcode 'girl' to not be child of 'woman'
            if single_path:
                G.add_edge('n09624168', 'n10129825')  # child of 'male' (sibling to 'male_child')
            else:
                G.add_edge('n09619168', 'n10129825')  # child of 'female'
            G.add_edge('n09619168', 'n10129825')  # child of 'female'
            continue

        hypernyms = [synset]
        while hypernyms:
            current = hypernyms.pop(0)
            set_node_label(G, current)
            for hypernym in current.hypernyms():
                G.add_edge(synset_to_wnid(hypernym), synset_to_wnid(current))
                hypernyms.append(hypernym)

                if single_path:
                    break

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


####################
# AUGMENTING GRAPH #
####################


class FakeSynset:

    def __init__(self, wnid):
        self.wnid = wnid

        assert isinstance(wnid, str)

    @staticmethod
    def create_from_offset(offset):
        return FakeSynset('f{:08d}'.format(offset))

    def offset(self):
        return int(self.wnid[1:])

    def pos(self):
        return 'f'

    def name(self):
        return '(generated)'


def augment_graph(G, extra, allow_imaginary=False, seed=0, max_retries=10000):
    """Augment graph G with extra% more nodes.

    e.g., If G has 100 nodes and extra = 0.5, the final graph will have 150
    nodes.
    """
    n = len(G.nodes)
    n_extra = int(extra / 100. * n)
    random.seed(seed)

    n_imaginary = 0
    for i in range(n_extra):
        candidate, is_imaginary_synset, children = get_new_node(G)
        if not is_imaginary_synset or \
                (is_imaginary_synset and allow_imaginary):
            add_node_to_graph(G, candidate, children)
            n_imaginary += is_imaginary_synset
            continue

        # now, must be imaginary synset AND not allowed
        if n_imaginary > 0:  # hit max retries before, not likely to find real
            return G, i, n_imaginary

        retries, is_imaginary_synset = 0, True
        while is_imaginary_synset:
            candidate, is_imaginary_synset, children = get_new_node(G)
            if retries > max_retries:
                print(f'Exceeded max retries ({max_retries})')
                return G, i, n_imaginary
        add_node_to_graph(G, candidate, children)

    return G, n_extra, n_imaginary


def get_new_node(G):
    """Get new candidate node for the graph"""
    root = get_root(G)
    nodes = list(filter(lambda node: node is not root and not node.startswith('f'), G.nodes))

    children = get_new_adjacency(G, nodes)
    synsets = [wnid_to_synset(wnid) for wnid in children]
    common_hypernyms = get_common_hypernyms(synsets)

    assert len(common_hypernyms) > 0, [synset.name() for synset in synsets]

    candidate = pick_unseen_hypernym(G, common_hypernyms)
    if candidate is None:
        return FakeSynset.create_from_offset(len(G.nodes)), True, children
    return candidate, False, children


def add_node_to_graph(G, candidate, children):
    root = get_root(G)

    wnid = synset_to_wnid(candidate)
    G.add_node(wnid)
    set_node_label(G, candidate)

    for child in children:
        G.add_edge(wnid, child)
    G.add_edge(root, wnid)


def get_new_adjacency(G, nodes):
    adjacency = set(tuple(adj) for adj in G.adj.values())
    children = next(iter(adjacency))

    while children in adjacency:
        k = random.randint(2, 4)
        children = tuple(random.sample(nodes, k=k))
    return children


def get_common_hypernyms(synsets):
    common_hypernyms = set(synsets[0].common_hypernyms(synsets[1]))
    for synset in synsets[2:]:
        common_hypernyms &= set(synsets[0].common_hypernyms(synset))
    return common_hypernyms


def deepest_synset(synsets):
    return max(synsets, key=lambda synset: synset.max_depth())


def pick_unseen_hypernym(G, common_hypernyms):
    candidate = deepest_synset(common_hypernyms)
    wnid = synset_to_wnid(candidate)

    while common_hypernyms and wnid in G.nodes:
        common_hypernyms -= {candidate}
        if not common_hypernyms:
            return None

        candidate = deepest_synset(common_hypernyms)
        wnid = synset_to_wnid(candidate)
    return candidate
