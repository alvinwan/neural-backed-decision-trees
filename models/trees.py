from utils.datasets import CIFAR10Node
import torch
import torch.nn as nn


__all__ = ('CIFAR10Tree',)

class CIFAR10Tree(nn.Module):
    """returns samples from all node classifiers"""

    def __init__(self, *args,
            path_tree='./data/cifar10/tree.xml',
            path_wnids='./data/cifar10/wnids.txt',
            pretrained=True,
            num_classes=10,
            **kwargs):
        super().__init__()

        wnid_to_node = CIFAR10Node.get_wnid_to_node(path_tree, path_wnids)
        wnids = sorted(wnid_to_node)
        self.nodes = [wnid_to_node[wnid] for wnid in wnids]
        self.nets = nn.ModuleList([
            self.get_net_for_node(node, pretrained) for node in self.nodes])

        self.linear = nn.Linear(self.get_input_dim(), num_classes)

    def get_net_for_node(self, node, pretrained):
        import models
        # TODO: WARNING: the model and paths are hardcoded
        net = models.ResNet10(num_classes=len(node.classes))

        if pretrained:
            checkpoint = torch.load(f'./checkpoint/ckpt-CIFAR10node-ResNet10-{node.wnid}.pth')
            # hacky fix lol
            state_dict = {key.replace('module.', '', 1): value for key, value in checkpoint['net'].items()}
            net.load_state_dict(state_dict)
        return net

    # WARNING: copy-pasta from above
    def get_input_dim(self):
        return sum([len(dataset.classes) for dataset in self.nodes])

    def forward(self, old_sample):
        with torch.no_grad():
            sample = []
            for net in self.nets:
                sample.extend(net(old_sample))
            sample = torch.cat(sample, 0)
        return self.linear(sample)
