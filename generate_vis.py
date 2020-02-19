import xmltodict
import json
import argparse
import torchvision
import os

from pathlib import Path
from utils.utils import DATASETS, METHODS, DATASET_TO_FOLDER_NAME
from utils import custom_datasets
from generate_tree import generate_fname


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
    help='Must be a folder data/{dataset} containing a wnids.txt',
    choices=DATASETS,
    default='CIFAR10')
parser.add_argument('--method', choices=METHODS,
    help='structure_released.xml apparently is missing many CIFAR100 classes. '
    'As a result, pruning does not work for CIFAR100. Random will randomly '
    'join clusters together, iteratively, to make a roughly-binary tree.',
    default='build')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--branching-factor', type=int, default=2)
parser.add_argument('--path-tree', type=str, default='')

args = parser.parse_args()

folder = DATASET_TO_FOLDER_NAME[args.dataset]
directory = os.path.join('data', folder)

fname = generate_fname(**vars(args))
if args.path_tree:
    file = args.path_tree
else:
    file = os.path.join(directory, f'{fname}.xml')
print('==> Reading from {}'.format(file))


if args.dataset in custom_datasets.names:
    dataset = getattr(custom_datasets, args.dataset)
else:
    dataset = getattr(torchvision.datasets, args.dataset)

dataset = dataset(root='./data', download=True)


def format_tree(tree_dict, parent_wnid):
    '''
    Format json tree
    '''
    tree = {}

    if 'synset' in tree_dict.keys():
        children = []
        if type(tree_dict['synset']) != list:
            tree_dict['synset'] = [tree_dict['synset']]

        for c in tree_dict['synset']:
            new_child = format_tree(c, tree_dict['@wnid'])
            children.append(new_child)

        tree['children'] = children
        tree['parent'] = parent_wnid
        tree['name'] = tree_dict['@wnid']
        tree['leaf'] = "False"
        return tree
    else:
        child = {
            'name': tree_dict['@wnid'],
            'parent': parent_wnid,
            'leaf': "True"
        }
        return child

# put wnid classes-index in json
wnid_file = os.path.join(directory, 'wnids.txt')
index = 0
wnid_names = {}
wnid_index = {}
with open(wnid_file) as fp:
    line = fp.readline().strip()
    while line:
        wnid_index[index] = line
        wnid_names[line] = dataset.classes[index]
        index += 1
        line = fp.readline().strip()

outfile_index = os.path.join(directory, '{0}-{1}_index_to_wnid.json'.format(args.dataset, args.method))
json.dump(wnid_index, open(outfile_index, 'w'))

outfile_names = os.path.join(directory, '{0}-{1}_class_index.json'.format(args.dataset, args.method))
json.dump(wnid_names, open(outfile_names, 'w'))

# put wnid classes-names in json



# convert from xml to json
with open(file) as inFh:
    tree_json = xmltodict.parse(inFh.read())

root = tree_json['tree']

if isinstance(root['synset'], dict):
    root = root['synset']

tree_data = format_tree(root, 'root')
# put new tree in file
fname = Path(file).stem
path_json = os.path.join(directory, f'{fname}-d3.json')
json.dump(tree_data, open(path_json, 'w'))

print('\033[92m==> Wrote JSON tree to {}\033[0m'.format(path_json))

with open('vis/tree-template.html') as f:
    html = f.read().replace(
        "'TREE_DATA_CONSTANT_TO_BE_REPLACED'",
        json.dumps(tree_data))

os.makedirs('out', exist_ok=True)
path_html = f'out/{fname}.html'
with open(path_html, 'w') as f:
    f.write(html)

print('\033[92m==> Wrote HTML tree to {}\033[0m'.format(path_html))
