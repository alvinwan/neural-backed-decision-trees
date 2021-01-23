from classy_vision.losses import ClassyLoss, register_loss
import torch.nn as nn
from nbdt.loss import SoftTreeSupLoss


@register_loss("NBDTTreeSupLoss")
class NBDTTreeSupLoss(SoftTreeSupLoss, ClassyLoss):
    def __init__(self):
        super().__init__(
            criterion=nn.CrossEntropyLoss().cuda(),
            dataset='Imagenet1000',
            tree_supervision_weight=10,
            hierarchy='induced-efficientnet_b7b'
        )

    @classmethod
    def from_config(cls, config):
        # We don't need anything from the config
        return cls()
