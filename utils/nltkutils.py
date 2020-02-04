from xml.etree.ElementTree import Element, SubElement, ElementTree
import nltk
from nltk.corpus import wordnet as wn


def synset_to_wnid(synset):
    return f'{synset.pos()}{synset.offset():08d}'


def wnid_to_synset(wnid):
    offset = int(wnid[1:])
    pos = wnid[0]
    return wn.synset_from_pos_and_offset(wnid[0], offset)


def build_minimal_wordnet_tree(wnids):
    root = Element('synset', {'wnid': 'fall11'})
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

        node = root.find('.//synset[@wnid="{}"]'.format(wnid))
        assert node is not None, (
            f'Could not find {wnid} in built tree, with wnids '
            f'{[node.get("wnid") for node in root.iter()]}'
        )
        assert len(node.getchildren()) == 0, (
            f'{wnid} ({original.definition()}) is not a leaf. Has '
            f'{len(node.getchildren())} children: '
            f'{[child.get("wnid") for child in node.getchildren()]}'
        )

    return ElementTree(root)


def get_leaves(tree):
    leaves = []
    for node in tree.iter():
        if not node.getchildren():
            yield node
