"""Various graph creation algorithms and utilities for WordNet association.

Builds off of thirdparty utilities in thirdparty/*, interfacing with WordNet
and Networkx for the rest of NBDT code.

- Minimal WordNet graph
- Random graph
- Induced graph
- WordNet association with nodes
"""
import networkx as nx
import json
import random
from nbdt.utils import DATASETS, METHODS, fwd
from networkx.readwrite.json_graph import node_link_data, node_link_graph
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import nbdt.models as models
import torch
import argparse
import os
from nbdt.thirdparty.wn import (
    FakeSynset,
    synset_to_wnid,
    wnid_to_synset,
    synset_to_name,
)
from nbdt.thirdparty.nx import get_roots
from nbdt.utils import get_directory


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="Must be a folder nbdt/wnids/{dataset}.txt containing wnids",
        choices=DATASETS,
        default="CIFAR10",
    )
    parser.add_argument(
        "--extra",
        type=int,
        default=0,
        help="Percent extra nodes to add to the tree. If 100, the number of "
        "nodes in tree are doubled. Note this is an integral percent.",
    )
    parser.add_argument(
        "--multi-path",
        action="store_true",
        help="Allows each leaf multiple paths to the root.",
    )
    parser.add_argument("--no-prune", action="store_true", help="Do not prune.")
    parser.add_argument(
        "--fname", type=str, help="Override all settings and just provide graph name"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Override all settings and just provide a path to a graph",
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        help="structure_released.xml apparently is missing many CIFAR100 classes. "
        "As a result, pruning does not work for CIFAR100. Random will randomly "
        "join clusters together, iteratively, to make a roughly-binary tree.",
        default="induced",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--branching-factor", type=int, default=2)
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="(induced hierarchy) Checkpoint to load into model. The fc weights"
        " are used for clustering.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="ResNet18",
        help="(induced hierarchy) Model name to get pretrained fc weights for.",
        choices=list(models.get_model_choices()),
    )
    parser.add_argument(
        "--induced-linkage",
        type=str,
        default="ward",
        help="(induced hierarchy) Linkage type used for agglomerative clustering",
    )
    parser.add_argument(
        "--induced-affinity",
        type=str,
        default="euclidean",
        help="(induced hierarchy) Metric used for computing similarity",
    )
    parser.add_argument(
        "--vis-out-fname", type=str, help="Base filename for vis output file"
    )
    parser.add_argument(
        "--vis-zoom",
        type=float,
        default=1.0,
        help="How large individual elements are, relative to the whole screen",
    )
    parser.add_argument(
        "--vis-scale",
        type=float,
        default=1.0,
        help="Initial scale for the svg. Like scaling an image.",
    )
    parser.add_argument(
        "--vis-curved", action="store_true", help="Use curved lines for edges"
    )
    parser.add_argument("--vis-sublabels", action="store_true", help="Show sublabels")
    parser.add_argument(
        "--vis-fake-sublabels", action="store_true", help="Show fake sublabels"
    )
    parser.add_argument(
        "--color",
        choices=("blue", "blue-green", "blue-minimal"),
        default="blue",
        help="Color to use, for colored flags. Note this takes NO effect if "
        "nodes are not colored.",
    )
    parser.add_argument(
        "--vis-no-color-leaves",
        action="store_true",
        help="Do NOT highlight leaves with special color.",
    )
    parser.add_argument(
        "--vis-color-path-to",
        type=str,
        help="Vis all nodes on path from leaf to root, as blue. Pass leaf name.",
    )
    parser.add_argument(
        "--vis-color-nodes",
        nargs="*",
        help="Nodes to color. Nodes are identified by label",
    )
    parser.add_argument(
        "--vis-force-labels-left",
        nargs="*",
        help="Labels to force text left of the node.",
    )
    parser.add_argument(
        "--vis-leaf-images",
        action="store_true",
        help="Include sample images for each leaf/class.",
    )
    parser.add_argument(
        "--vis-image-resize-factor",
        type=float,
        default=1.0,
        help="Factor to resize image size by. Default image size is provided "
        "by the original image. e.g., 32 for CIFAR10, 224 for Imagenet",
    )
    parser.add_argument(
        "--vis-height",
        type=int,
        default=750,
        help="Height of the outputted visualization",
    )
    parser.add_argument("--vis-width", type=int, default=3000)
    parser.add_argument(
        "--vis-theme", choices=("dark", "minimal", "regular"), default="regular"
    )
    parser.add_argument("--vis-root", type=str, help="Which node is root")
    parser.add_argument("--vis-margin-top", type=int, default=20)
    parser.add_argument("--vis-margin-left", type=int, default=250)
    parser.add_argument("--vis-hide", nargs="*", help="IDs of nodes to hide")
    parser.add_argument(
        "--vis-node-conf",
        nargs=3,
        action="append",
        help="Key-value pairs to add: <node> <key> <value>",
    )
    parser.add_argument(
        "--vis-above-dy",
        type=int,
        default=325,
        help="Amount to offset images above nodes by",
    )
    parser.add_argument(
        "--vis-below-dy",
        type=int,
        default=200,
        help="Amount to offset images below nodes by",
    )
    parser.add_argument("--vis-colormap", help="Path to colormap image")
    parser.add_argument("--vis-root-y", type=int, help="root position y", default=-1)
    return parser


def generate_graph_fname(
    method,
    seed=0,
    branching_factor=2,
    extra=0,
    no_prune=False,
    fname="",
    path="",
    multi_path=False,
    induced_linkage="ward",
    induced_affinity="euclidean",
    checkpoint=None,
    arch=None,
    **kwargs,
):
    if path:
        return Path(path).stem
    if fname:
        return fname

    fname = f"graph-{method}"
    if method == "random":
        if seed != 0:
            fname += f"-seed{seed}"
    if method == "induced":
        assert (
            checkpoint or arch
        ), "Induced hierarchy needs either `arch` or `checkpoint`"
        if induced_linkage != "ward" and induced_linkage is not None:
            fname += f"-linkage{induced_linkage}"
        if induced_affinity != "euclidean" and induced_affinity is not None:
            fname += f"-affinity{induced_affinity}"
        if checkpoint:
            checkpoint_stem = Path(checkpoint).stem
            if checkpoint_stem.startswith("ckpt-") and checkpoint_stem.count("-") >= 2:
                checkpoint_suffix = "-".join(checkpoint_stem.split("-")[2:])
                checkpoint_fname = checkpoint_suffix.replace("-induced", "")
            else:
                checkpoint_fname = checkpoint_stem
        else:
            checkpoint_fname = arch
        fname += f"-{checkpoint_fname}"
    if method in ("random", "induced"):
        if branching_factor != 2:
            fname += f"-branch{branching_factor}"
    if extra > 0:
        fname += f"-extra{extra}"
    if no_prune:
        fname += "-noprune"
    if multi_path:
        fname += "-multi"
    return fname


def get_graph_path_from_args(
    dataset,
    method,
    seed=0,
    branching_factor=2,
    extra=0,
    no_prune=False,
    fname="",
    path="",
    multi_path=False,
    induced_linkage="ward",
    induced_affinity="euclidean",
    checkpoint=None,
    arch=None,
    **kwargs,
):
    if path:
        return path
    fname = generate_graph_fname(
        method=method,
        seed=seed,
        branching_factor=branching_factor,
        extra=extra,
        no_prune=no_prune,
        fname=fname,
        multi_path=multi_path,
        induced_linkage=induced_linkage,
        induced_affinity=induced_affinity,
        checkpoint=checkpoint,
        arch=arch,
    )
    directory = get_directory(dataset)
    path = os.path.join(directory, f"{fname}.json")
    return path


#################
# MINIMAL GRAPH #
#################


def build_minimal_wordnet_graph(wnids, multi_path=False):
    G = nx.DiGraph()

    for wnid in wnids:
        G.add_node(wnid)
        synset = wnid_to_synset(wnid)
        set_node_label(G, synset)

        if wnid == "n10129825":  # hardcode 'girl' to not be child of 'woman'
            if not multi_path:
                G.add_edge(
                    "n09624168", "n10129825"
                )  # child of 'male' (sibling to 'male_child')
            else:
                G.add_edge("n09619168", "n10129825")  # child of 'female'
            G.add_edge("n09619168", "n10129825")  # child of 'female'
            continue

        hypernyms = [synset]
        while hypernyms:
            current = hypernyms.pop(0)
            set_node_label(G, current)
            for hypernym in current.hypernyms():
                G.add_edge(synset_to_wnid(hypernym), synset_to_wnid(current))
                hypernyms.append(hypernym)

                if not multi_path:
                    break

        children = [(key, wnid_to_synset(key).name()) for key in G.succ[wnid]]
        assert (
            len(children) == 0
        ), f"Node {wnid} ({synset.name()}) is not a leaf. Children: {children}"
    return G


################
# RANDOM GRAPH #
################


def build_random_graph(wnids, seed=0, branching_factor=2):
    random.seed(seed)

    G = nx.DiGraph()

    if seed >= 0:
        random.shuffle(wnids)
    current = None
    remaining = wnids

    # Build the graph from the leaves up
    while len(remaining) > 1:
        current, remaining = remaining, []
        while current:
            nodes, current = current[:branching_factor], current[branching_factor:]
            remaining.append(nodes)

    # Construct networkx graph from root down
    G.add_node("0")
    set_random_node_label(G, "0")
    next = [(remaining[0], "0")]
    i = 1
    while next:
        group, parent = next.pop(0)
        if len(group) == 1:
            if isinstance(group[0], str):
                G.add_node(group[0])
                synset = wnid_to_synset(group[0])
                set_node_label(G, synset)
                G.add_edge(parent, group[0])
            else:
                next.append((group[0], parent))
            continue

        for candidate in group:
            is_leaf = not isinstance(candidate, list)
            wnid = candidate if is_leaf else str(i)
            G.add_node(wnid)
            if is_leaf:
                synset = wnid_to_synset(wnid)
                set_node_label(G, synset)
            else:
                set_random_node_label(G, wnid)
            G.add_edge(parent, wnid)
            i += 1

            if not is_leaf:
                next.append((candidate, wnid))
    return G


################
# INDUCED TREE #
################


MODEL_FC_KEYS = (
    "fc.weight",
    "linear.weight",
    "module.linear.weight",
    "module.net.linear.weight",
    "output.weight",
    "module.output.weight",
    "output.fc.weight",
    "module.output.fc.weight",
    "classifier.weight",
    "model.last_layer.3.weight",
)


def build_induced_graph(
    wnids,
    checkpoint,
    model=None,
    linkage="ward",
    affinity="euclidean",
    branching_factor=2,
    dataset="CIFAR10",
    state_dict=None,
):
    num_classes = len(wnids)
    assert (
        checkpoint or model or state_dict
    ), "Need to specify either `checkpoint` or `method` or `state_dict`."
    if state_dict:
        centers = get_centers_from_state_dict(state_dict)
    elif checkpoint:
        centers = get_centers_from_checkpoint(checkpoint)
    else:
        centers = get_centers_from_model(model, num_classes, dataset)
    assert num_classes == centers.size(0), (
        f"The model FC supports {centers.size(0)} classes. However, the dataset"
        f" {dataset} features {num_classes} classes. Try passing the "
        "`--dataset` with the right number of classes."
    )

    if centers.is_cuda:
        centers = centers.cpu()

    G = nx.DiGraph()

    # add leaves
    for wnid in wnids:
        G.add_node(wnid)
        set_node_label(G, wnid_to_synset(wnid))

    # add rest of tree
    clustering = AgglomerativeClustering(
        linkage=linkage, n_clusters=branching_factor, affinity=affinity,
    ).fit(centers)
    children = clustering.children_
    index_to_wnid = {}

    for index, pair in enumerate(map(tuple, children)):
        child_wnids = []
        child_synsets = []
        for child in pair:
            if child < num_classes:
                child_wnid = wnids[child]
            else:
                child_wnid = index_to_wnid[child - num_classes]
            child_wnids.append(child_wnid)
            child_synsets.append(wnid_to_synset(child_wnid))

        parent = get_wordnet_meaning(G, child_synsets)
        parent_wnid = synset_to_wnid(parent)
        G.add_node(parent_wnid)
        set_node_label(G, parent)
        index_to_wnid[index] = parent_wnid

        for child_wnid in child_wnids:
            G.add_edge(parent_wnid, child_wnid)

    assert len(list(get_roots(G))) == 1, list(get_roots(G))
    return G


def get_centers_from_checkpoint(checkpoint):
    data = torch.load(checkpoint, map_location=torch.device("cpu"))

    for key in ("net", "state_dict"):
        try:
            state_dict = data[key]
            break
        except:
            state_dict = data

    fc = get_centers_from_state_dict(state_dict)
    assert (
        fc is not None
    ), f"Could not find FC weights in checkpoint {checkpoint} with keys: {net.keys()}"
    return fc


def get_centers_from_model(model, num_classes, dataset):
    net = None
    try:
        net = getattr(models, model)(
            pretrained=True, num_classes=num_classes, dataset=dataset
        )
    except TypeError as e:
        print(f"Ignoring TypeError. Retrying without `dataset` kwarg: {e}")
        try:
            net = getattr(models, model)(pretrained=True, num_classes=num_classes)
        except TypeError as e:
            print(e)
    assert net is not None, f"Could not find pretrained model {model}"
    fc = get_centers_from_state_dict(net.state_dict())
    assert (
        fc is not None
    ), f"Could not find FC weights in model {model} with keys: {net.keys()}"
    return fc


def get_centers_from_state_dict(state_dict):
    fc = None
    for key in MODEL_FC_KEYS:
        if key in state_dict:
            fc = state_dict[key].squeeze()
            break
    if fc is not None:
        return fc.detach()


####################
# AUGMENTING GRAPH #
####################


def augment_graph(G, extra, allow_imaginary=False, seed=0, max_retries=10000):
    """Augment graph G with extra% more nodes.

    e.g., If G has 100 nodes and extra = 0.5, the final graph will have 150
    nodes.
    """
    n = len(G.nodes)
    n_extra = int(extra / 100.0 * n)
    random.seed(seed)

    n_imaginary = 0
    for i in range(n_extra):
        candidate, is_imaginary_synset, children = get_new_node(G)
        if not is_imaginary_synset or (is_imaginary_synset and allow_imaginary):
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
                print(f"Exceeded max retries ({max_retries})")
                return G, i, n_imaginary
        add_node_to_graph(G, candidate, children)

    return G, n_extra, n_imaginary


def set_node_label(G, synset):
    nx.set_node_attributes(G, {synset_to_wnid(synset): synset_to_name(synset)}, "label")


def set_random_node_label(G, i):
    nx.set_node_attributes(G, {i: ""}, "label")


def get_new_node(G):
    """Get new candidate node for the graph"""
    root = get_root(G)
    nodes = list(
        filter(lambda node: node is not root and not node.startswith("f"), G.nodes)
    )

    children = get_new_adjacency(G, nodes)
    synsets = [wnid_to_synset(wnid) for wnid in children]

    candidate = get_wordnet_meaning(G, synsets)
    is_fake = candidate.pos() == "f"
    return candidate, is_fake, children


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


def prune_single_successor_nodes(G):
    for node in G.nodes:
        if len(G.succ[node]) == 1:
            succ = list(G.succ[node])[0]
            G = nx.contracted_nodes(G, succ, node, self_loops=False)
    return G


####################
# WORDNET + GRAPHS #
####################


def get_wordnet_meaning(G, synsets):
    hypernyms = get_common_hypernyms(synsets)
    candidate = pick_unseen_hypernym(G, hypernyms) if hypernyms else None
    if candidate is None:
        return FakeSynset.create_from_offset(len(G.nodes))
    return candidate


def deepest_synset(synsets):
    return max(synsets, key=lambda synset: synset.max_depth())


def get_common_hypernyms(synsets):
    if any(synset.pos() == "f" for synset in synsets):
        return set()
    common_hypernyms = set(synsets[0].common_hypernyms(synsets[1]))
    for synset in synsets[2:]:
        common_hypernyms &= set(synsets[0].common_hypernyms(synset))
    return common_hypernyms


def pick_unseen_hypernym(G, common_hypernyms):
    assert len(common_hypernyms) > 0

    candidate = deepest_synset(common_hypernyms)
    wnid = synset_to_wnid(candidate)

    while common_hypernyms and wnid in G.nodes:
        common_hypernyms -= {candidate}
        if not common_hypernyms:
            return None

        candidate = deepest_synset(common_hypernyms)
        wnid = synset_to_wnid(candidate)
    return candidate
