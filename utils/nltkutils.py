from xml.etree.ElementTree import Element, SubElement, ElementTree
import nltk
from nltk.corpus import wordnet as wn
import random


def synset_to_wnid(synset):
    return f'{synset.pos()}{synset.offset():08d}'


def wnid_to_synset(wnid):
    offset = int(wnid[1:])
    pos = wnid[0]
    return wn.synset_from_pos_and_offset(wnid[0], offset)


def get_paths(synset, extra_roots=False):
    last = [[synset]]
    current = []
    final = []

    while last:
        while last:
            path = last.pop(0)
            synset = path[0]
            if not synset.hypernyms():
                final.append(path)
            else:
                for hypernym in synset.hypernyms():
                    path = path[:]
                    path.insert(0, hypernym)
                    current.append(path)

                    if not extra_roots:
                        break
        last = current
        current = []
    return final


def build_tree(root, path, wnid_to_node={}, wnid_to_parent={}):
    parent = root
    for synset in path:
        id = synset_to_wnid(synset)
        if id in wnid_to_node:
            parent = wnid_to_node[id]
        elif id == 'n10129825':  # harcode making woman/girl siblings
            wnid_to_parent[id] = parent
            parent = SubElement(wnid_to_parent['n10787470'], 'synset', {'wnid': id})
            wnid_to_node[id] = parent
        else:
            wnid_to_parent[id] = parent
            parent = SubElement(parent, 'synset', {'wnid': id})
            wnid_to_node[id] = parent


def build_minimal_wordnet_tree(wnids, extra_roots=False):
    tree = Element('tree')
    root = SubElement(tree, 'synset', {'wnid': 'fall11'})
    wnid_to_node = {'fall11': root}
    wnid_to_parent = {}

    for wnid in wnids:
        offset = int(wnid[1:])
        synset = original = wnid_to_synset(wnid)

        assert synset is not None
        assert wnid == synset_to_wnid(original), (
            f'Wrong synset may have been used for wnid {wnid} (became '
            f'{synset_to_wnid(original)})'
        )

        paths = get_paths(synset, extra_roots=extra_roots)
        for path in paths:
            build_tree(root, path, wnid_to_node, wnid_to_parent)

        node = root.find(f'.//synset[@wnid="{wnid}"]')
        assert node is not None, (
            f'Could not find {wnid} in built tree, with wnids '
            f'{[node.get("wnid") for node in root.iter()]}'
        )
        assert len(node.getchildren()) == 0, (
            f'{wnid} ({original.definition()}) is not a leaf. Has '
            f'{len(node.getchildren())} children: '
            f'{[child.get("wnid") for child in node.getchildren()]}'
        )

    return ElementTree(tree)


def get_leaves(tree):
    leaves = []
    for node in tree.iter():
        if not node.getchildren():
            yield node


def build_random_tree(wnids, seed=0, branching_factor=2):
    random.seed(seed)

    tree = Element('tree')
    root = SubElement(tree, 'synset', {'wnid': 'fall11'})
    wnid_to_node = {'fall11': root}

    random.shuffle(wnids)
    current = None
    remaining = wnids

    # build the tree from the leaves up
    while len(remaining) > 1:
        current, remaining = remaining, []
        while current:
            nodes, current = current[:branching_factor], current[branching_factor:]
            remaining.append(nodes)

    # construct the xml tree from the root down
    next = [(remaining[0], root)]
    i = 0
    while next:
        group, parent = next.pop(0)
        if len(group) == 1:
            if isinstance(group[0], str):
                leaf = SubElement(parent, 'synset', {'wnid': group[0]})
                i += 1
            else:
                next.append((group[0], parent))
            continue

        for candidate in group:
            is_leaf = not isinstance(candidate, list)
            wnid = candidate if is_leaf else str(i)
            node = SubElement(parent, 'synset', {'wnid': wnid})
            i += 1

            if not is_leaf:
                next.append((candidate, node))

            node = root.find(f'.//synset[@wnid="{wnid}"]')
            assert node is not None, (
                f'Could not find {wnid} in built tree, with wnids '
                f'{[node.get("wnid") for node in root.iter()]}'
            )

    return ElementTree(tree)
