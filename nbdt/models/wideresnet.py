from pytorchcv.models.wrn_cifar import (
    wrn28_10_cifar10,
    wrn28_10_cifar100,
    get_wrn_cifar,
)
from nbdt.models.utils import get_pretrained_model
import torch.nn as nn


__all__ = ("wrn28_10", "wrn28_10_cifar10", "wrn28_10_cifar100")


model_urls = {
    (
        "wrn28_10",
        "TinyImagenet200",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-TinyImagenet200-wrn28_10.pth"
}


def _wrn(arch, model, pretrained=False, progress=True, dataset="CIFAR10"):
    model = get_pretrained_model(
        arch, dataset, model, model_urls, pretrained=pretrained, progress=progress
    )
    return model


def wrn28_10(pretrained=False, progress=True, dataset="CIFAR10", **kwargs):
    """Replace `final_pool` (8x8 average pooling) with a global average pooling.

    If this gets crappy accuracy for TinyImagenet200, it's probably because the
    final pooled feature map is 16x16 instead of 8x8. So needs another stride 2
    stage, technically.
    """
    model = get_wrn_cifar(blocks=28, width_factor=10, model_name="wrn28_10", **kwargs)
    model.features.final_pool = nn.AdaptiveAvgPool2d((1, 1))
    model = _wrn(
        "wrn28_10", model, pretrained=pretrained, progress=progress, dataset=dataset
    )
    return model
