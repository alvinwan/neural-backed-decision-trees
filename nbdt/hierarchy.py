from nbdt.utils import DATASETS, METHODS, Colors, fwd
from nbdt.graph import build_minimal_wordnet_graph, build_random_graph, \
    prune_single_successor_nodes, write_graph, get_wnids, generate_fname, \
    get_parser, get_wnids_from_dataset, get_directory, get_graph_path_from_args, \
    augment_graph, get_depth, build_induced_graph, read_graph, get_leaves, \
    get_roots, synset_to_wnid, wnid_to_name
from nbdt import data
from networkx.readwrite.json_graph import adjacency_data
from pathlib import Path
import os
import json
import torchvision
import base64
from io import BytesIO


############
# GENERATE #
############


def print_graph_stats(G, name):
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


def generate_hierarchy(
        dataset, method, seed=0, branching_factor=2, extra=0,
        no_prune=False, fname='', single_path=False,
        induced_linkage='ward', induced_affinity='euclidean',
        induced_checkpoint=None, induced_model=None, model=None, **kwargs):
    wnids = get_wnids_from_dataset(dataset)

    if method == 'wordnet':
        G = build_minimal_wordnet_graph(wnids, single_path)
    elif method == 'random':
        G = build_random_graph(wnids, seed=seed, branching_factor=branching_factor)
    elif method == 'induced':
        G = build_induced_graph(wnids,
            dataset=dataset,
            checkpoint=induced_checkpoint,
            model=induced_model,
            linkage=induced_linkage,
            affinity=induced_affinity,
            branching_factor=branching_factor,
            state_dict=model.state_dict() if model is not None else None)
    else:
        raise NotImplementedError(f'Method "{method}" not yet handled.')
    print_graph_stats(G, 'matched')
    assert_all_wnids_in_graph(G, wnids)

    if not no_prune:
        G = prune_single_successor_nodes(G)
        print_graph_stats(G, 'pruned')
        assert_all_wnids_in_graph(G, wnids)

    if extra > 0:
        G, n_extra, n_imaginary = augment_graph(G, extra, True)
        print(f'[extra] \t Extras: {n_extra} \t Imaginary: {n_imaginary}')
        print_graph_stats(G, 'extra')
        assert_all_wnids_in_graph(G, wnids)

    path = get_graph_path_from_args(
        dataset=dataset,
        method=method,
        seed=seed,
        branching_factor=branching_factor,
        extra=extra,
        no_prune=no_prune,
        fname=fname,
        single_path=single_path,
        induced_linkage=induced_linkage,
        induced_affinity=induced_affinity,
        induced_checkpoint=induced_checkpoint,
        induced_model=induced_model)
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
    path = get_graph_path_from_args(**vars(args))
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


def build_tree(G, root,
        parent='null',
        color_leaves_blue=True,
        force_labels_left=(),
        include_leaf_images=False,
        dataset=None,
        image_resize_factor=1):
    children = [
        build_tree(G, child, root,
            color_leaves_blue=color_leaves_blue,
            force_labels_left=force_labels_left,
            include_leaf_images=include_leaf_images,
            dataset=dataset,
            image_resize_factor=image_resize_factor)
        for child in G.succ[root]]
    label = G.nodes[root].get('label', '')
    sublabel = root

    if root.startswith('f'):  # WARNING: hacky, ignores fake wnids -- this will have to be changed lol
        sublabel = ''

    node = {
        'sublabel': sublabel,
        'label': label,
        'parent': parent,
        'children': children,
    }

    is_leaf = len(children) == 0
    if is_leaf and color_leaves_blue:
        node['color'] = 'blue'

    if label in force_labels_left:
        node['force_text_on_left'] = True

    if include_leaf_images and is_leaf:
        image = get_class_image_from_dataset(dataset, label)
        base64_encode = image_to_base64_encode(image, format="jpeg")
        image_href = f"data:image/jpeg;base64,{base64_encode.decode('utf-8')}"
        image_height, image_width = image.size
        node['image'] = {
            'href': image_href,
            'width': image_width * image_resize_factor,
            'height': image_height *  image_resize_factor
        }
    return node


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


def get_class_image_from_dataset(dataset, candidate):
    """Returns image for given class `candidate`. Image is PIL."""
    if isinstance(candidate, int):
        candidate = dataset.classes[candidate]
    for sample, label in dataset:
        intersection = compare_wnids(dataset.classes[label], candidate)
        if label == candidate or intersection:
            return sample
    raise UserWarning(f'No samples with label {candidate} found.')


def compare_wnids(label1, label2):
    from nltk.corpus import wordnet as wn  # entire script should not depend on wordnet
    synsets1 = wn.synsets(label1, pos=wn.NOUN)
    synsets2 = wn.synsets(label2, pos=wn.NOUN)
    wnids1 = set(map(synset_to_wnid, synsets1))
    wnids2 = set(map(synset_to_wnid, synsets2))
    return wnids1.intersection(wnids2)


def image_to_base64_encode(image, format="jpeg"):
    """Converts PIL image to base64 encoding, ready for use as data uri."""
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue())


def generate_vis(path_template, data, name, fname, zoom=2, straight_lines=True,
        show_sublabels=False):
    with open(path_template) as f:
        html = f.read() \
        .replace(
            "CONFIG_TREE_DATA",
            json.dumps([data])) \
        .replace(
            "CONFIG_ZOOM",
            str(zoom)) \
        .replace(
            "CONFIG_STRAIGHT_LINES",
            str(straight_lines).lower()) \
        .replace(
            "CONFIG_SHOW_SUBLABELS",
            str(show_sublabels).lower()) \
        .replace(
            "CONFIG_TITLE",
            fname)

    os.makedirs('out', exist_ok=True)
    path_html = f'out/{fname}-{name}.html'
    with open(path_html, 'w') as f:
        f.write(html)

    Colors.green('==> Wrote HTML to {}'.format(path_html))


def generate_hierarchy_vis(args):
    path = get_graph_path_from_args(**vars(args))
    print('==> Reading from {}'.format(path))

    G = read_graph(path)

    roots = list(get_roots(G))
    num_roots = len(roots)
    root = next(get_roots(G))

    dataset = None
    if args.dataset:
        cls = getattr(data, args.dataset)
        dataset = cls(root='./data', train=False, download=False)

    tree = build_tree(G, root,
        color_leaves_blue=not args.vis_gray,
        force_labels_left=args.vis_force_labels_left or [],
        dataset=dataset,
        include_leaf_images=args.vis_leaf_images,
        image_resize_factor=args.vis_image_resize_factor)
    graph = build_graph(G)

    if num_roots > 1:
        Colors.red(f'Found {num_roots} roots! Should be only 1: {roots}')
    else:
        print(f'Found just {num_roots} root.')

    fname = generate_fname(**vars(args)).replace('graph-', f'{args.dataset}-', 1)
    parent = Path(fwd()).parent
    generate_vis(
        str(parent / 'nbdt/templates/tree-template.html'), tree, 'tree', fname,
        zoom=args.vis_zoom,
        straight_lines=not args.vis_curved,
        show_sublabels=args.vis_sublabels)
