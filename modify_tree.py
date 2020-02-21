"""Randomly reshuffle leaves"""

import xml.etree.ElementTree as ET
from pathlib import Path
from utils.xmlutils import get_leaves
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path-tree', default='./data/CIFAR10/tree-build.xml')
parser.add_argument('--method', choices=('shuffle',), default='shuffle')
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()


tree = ET.parse(args.path_tree)
leaves = list(get_leaves(tree))

leaves_data = [
    {key: leaf.get(key) for key in leaf.keys()}
    for leaf in leaves
]
random.seed(args.seed)
random.shuffle(leaves_data)

for leaf, data in zip(leaves, leaves_data):
    for key, value in data.items():
        leaf.set(key, value)


path = Path(args.path_tree)
path = path \
    .with_name(f'{path.stem}-{args.method}') \
    .with_suffix(path.suffix)

tree.write(str(path))
print('\033[92m==> Wrote modified tree to {}\033[0m'.format(path))
