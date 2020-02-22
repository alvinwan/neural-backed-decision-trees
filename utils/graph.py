import networkx as nx
import json
from nltk.corpus import wordnet as wn
from networkx.readwrite.json_graph import node_link_data, node_link_graph


def synset_to_wnid(synset):
    return f'{synset.pos()}{synset.offset():08d}'


def wnid_to_synset(wnid):
    offset = int(wnid[1:])
    pos = wnid[0]
    return wn.synset_from_pos_and_offset(wnid[0], offset)


def build_minimal_wordnet_graph(wnids):
    G = nx.DiGraph()

    for wnid in wnids:
        G.add_node(wnid)
        synset = wnid_to_synset(wnid)

        hypernyms = [synset]
        while hypernyms:
            current = hypernyms.pop(0)
            for hypernym in current.hypernyms():
                G.add_edge(synset_to_wnid(hypernym), wnid)
                hypernyms.append(hypernym)

        assert len(G.succ[wnid]) == 0
    return G


def prune_single_successor_nodes(G):
    for node in G.nodes:
        if len(G.succ[node]) == 1:
            succ = list(G.succ[node])[0]
            G = nx.contracted_nodes(G, node, succ, self_loops=False)
    return G


def write_graph(G, path):
    with open(path, 'w') as f:
        json.dump(node_link_data(G), f)


def read_graph(path):
    with open(path) as f:
        return node_link_graph(json.load(f))
