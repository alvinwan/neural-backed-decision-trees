from torchvision import datasets
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from utils import custom_datasets, nmn_datasets
import numpy as np
import operator
import xml.etree.ElementTree as ET

from sklearn.cluster import KMeans

# hacky way of "removing" layer in pytorch
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


# n_classes - number of nodes at current level, since we are building bottom up
# feature_set - np array of feature maps from each image
# tree_map - mapping (using list) from feature_set to node, which node does each image belong to
# tree_structure - list of nodes at this level -> tuple of (node label, [children])
def kmeans_cluster(n_classes, feature_set, tree_map, tree_structure):
	if n_classes == 1:
		return tree_structure

	n_clusters = max(2, n_classes // 2)

	kmeans = KMeans(n_clusters = n_clusters).fit(feature_set)
	cluster_labels = kmeans.labels_

	# count occurrences
	cluster_count = {treenode:{i: 0 for i in range(n_clusters)} for treenode in set(tree_map)}
	for cluster_label, treenode in zip(cluster_labels, tree_map):
		cluster_count[treenode][cluster_label] += 1

	cluster_belong = {}
	new_tree_structure  = {}
	# get cluster to which each node belongs in
	for treenode in tree_structure:
		cluster = max(cluster_count[treenode[0]].items(), key=operator.itemgetter(1))[0]
		cluster_belong[treenode[0]] = cluster

		if cluster not in new_tree_cluster:
			new_tree_structure[cluster] = [treenode]
		else:
			new_tree_structure[cluster].append(treenode)

	return kmeans_cluster()


import argparse

if __name__ == '__main__':

	dataset_choices = ('CIFAR10', 'CIFAR100') + custom_datasets.names + nmn_datasets.names

	parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
	parser.add_argument('--dataset', default='CIFAR100', choices=dataset_choices)
	parser.add_argument('--model', default='ResNet18')
	parser.add_argument('--batch-size', default=64, type=int,
	                    help='Batch size used for pulling data')
	parser.add_argument('--wnid', help='wordnet id for cifar10node dataset',
	                    default='fall11')
	parser.add_argument('--sample', default=1.0, type=float, help='How much of the dataset to sample')
	args = parser.parse_args()

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# load datasets
	print("==> Preparing data...")

	# Currently only using one set with no data augmentation for feeding into model for feature clustering
	transform = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	if args.dataset in nmn_datasets.names:
	    dataset = getattr(nmn_datasets, args.dataset)
	elif args.dataset in custom_datasets.names:
	    dataset = getattr(custom_datasets, args.dataset)
	else:
	    dataset = getattr(datasets, args.dataset)

	dataset_args = ()
	if getattr(dataset, 'needs_wnid', False):
	    dataset_args = (args.wnid,)

	trainset = dataset(*dataset_args, root='./data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)


	# load pretrained model
	print("==> Preparing model...")

	model_ft = getattr(models, args.model.lower())(pretrained=True)
	# hacky way to remove last layer
	model_ft.fc = Identity()

	if device == 'cuda':
		print("Using CUDA")
		model_ft = torch.nn.DataParallel(model_ft)

	feature_set = []
	label_set = []
	
	i = 0
	for batch_idx, (inputs, labels) in enumerate(trainloader):
		inputs = inputs.to(device)
		outputs = model_ft(inputs).cpu().detach().numpy()
		labels = labels.cpu().numpy()

		feature_set.append(outputs)
		label_set.append(labels)

		i += args.batch_size
		if i > args.sample * len(trainset):
			break

	feature_set = np.concatenate(feature_set)
	label_set = np.concatenate(label_set)
	print("Number of samples: %d" % len(feature_set))

	#############################################
	#	Feature maps set up, time to cluster    #
	#############################################
	label_to_leaf_map = {}

	# pass to clustering algo
	kmeans_cluster(len(np.unique(label_set)), feature_set, label_set)

	


		



