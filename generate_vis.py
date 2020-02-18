import xmltodict
import json
import argparse
import torchvision
import os
        
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
    help='Must be a folder data/{dataset} containing a wnids.txt',
    choices=('tiny-imagenet-200', 'CIFAR10', 'CIFAR100'),
    default='CIFAR10')
parser.add_argument('--method', choices=('prune', 'build', 'random'),
    help='structure_released.xml apparently is missing many CIFAR100 classes. '
    'As a result, pruning does not work for CIFAR100. Random will randomly '
    'join clusters together, iteratively, to make a roughly-binary tree.',
    default='build')

args = parser.parse_args()

directory = os.path.join('data', args.dataset)
file = os.path.join(directory, 'tree-{}.xml'.format(args.method))
print('reading from ', file)

dataset = getattr(torchvision.datasets, args.dataset)(root='./data', download=True)


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

root = tree_json['tree']['synset']

tree_data = format_tree(root, 'root')
# put new tree in file
outfile = os.path.join(directory, 'new_{0}-{1}_d3.json'.format(args.dataset, args.method))
json.dump(tree_data, open(outfile, 'w'))

print('saved json tree to ', directory)