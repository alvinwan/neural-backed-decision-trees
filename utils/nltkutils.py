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


def build_minimal_wordnet_tree(wnids):
    tree = Element('tree')
    root = SubElement(tree, 'synset', {'wnid': 'fall11'})
    wnid_to_node = {'fall11': root}
    wnid_to_parent = {}

    for wnid in wnids:
        offset = int(wnid[1:])
        synset = original = wnid_to_synset(wnid)
        assert synset is not None

        path = [synset_to_wnid(synset)]
        while synset.hypernyms():
            hypernym = synset.hypernyms()[0]
            path.insert(0, synset_to_wnid(hypernym))
            synset = hypernym

        assert wnid == synset_to_wnid(original), (
            f'Wrong synset may have been used for wnid {wnid} (became '
            f'{synset_to_wnid(original)})'
        )

        parent = root
        for id in path:
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
