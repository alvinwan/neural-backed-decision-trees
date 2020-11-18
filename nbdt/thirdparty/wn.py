"""Utilities for NLTK WordNet synsets and WordNet IDs"""
import networkx as nx
import json
import random
from nbdt.utils import DATASETS, METHODS, fwd, get_directory
from networkx.readwrite.json_graph import node_link_data, node_link_graph
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import nbdt.models as models
import torch
import argparse
import os
import nltk


def maybe_install_wordnet():
    try:
        nltk.data.find("corpora/wordnet")
    except Exception as e:
        print(e)
        nltk.download("wordnet")


def get_wnids(path_wnids):
    if not os.path.exists(path_wnids):
        parent = Path(fwd()).parent
        print(f"No such file or directory: {path_wnids}. Looking in {str(parent)}")
        path_wnids = parent / path_wnids
    with open(path_wnids) as f:
        wnids = [wnid.strip() for wnid in f.readlines()]
    return wnids


def get_wnids_from_dataset(dataset, root="./nbdt/wnids"):
    directory = get_directory(dataset, root)
    return get_wnids(f"{directory}.txt")


##########
# SYNSET #
##########


def synset_to_wnid(synset):
    return f"{synset.pos()}{synset.offset():08d}"


def wnid_to_synset(wnid):
    from nltk.corpus import wordnet as wn  # entire script should not depend on wn

    offset = int(wnid[1:])
    pos = wnid[0]

    try:
        return wn.synset_from_pos_and_offset(wnid[0], offset)
    except:
        return FakeSynset(wnid)


def wnid_to_name(wnid):
    return synset_to_name(wnid_to_synset(wnid))


def synset_to_name(synset):
    return synset.name().split(".")[0]


def write_wnids(wnids, path):
    makeparentdirs(path)
    with open(str(path), "w") as f:
        f.write("\n".join(wnids))


class FakeSynset:
    def __init__(self, wnid):
        self.wnid = wnid

        assert isinstance(wnid, str)

    @staticmethod
    def create_from_offset(offset):
        return FakeSynset("f{:08d}".format(offset))

    def offset(self):
        return int(self.wnid[1:])

    def pos(self):
        return "f"

    def name(self):
        return "(generated)"

    def definition(self):
        return "(generated)"
