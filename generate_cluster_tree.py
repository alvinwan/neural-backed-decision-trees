from torchvision import datasets
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from utils import custom_datasets, nmn_datasets
import numpy as np


# hacky way of "removing" layer in pytorch
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


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
		i += 1
		if i == 10:
			break

	feature_set = np.concatenate(feature_set)
	label_set = np.concatenate(label_set)

	print(feature_set.shape)
	print(label_set.shape)


		



