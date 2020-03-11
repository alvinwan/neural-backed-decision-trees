from pytorchcv.models.wrn import get_wrn
import torch.nn as nn

__all__ = ('wrn28_10',)


def wrn28_10(num_classes=10, **kwargs):
    """Replace `final_pool` (8x8 average pooling) with a global average pooling.

    If this gets crappy accuracy for TinyImagenet200, it's probably because the
    final pooled feature map is 16x16 instead of 8x8. So needs another stride 2
    stage, technically.
    """
    net = get_wrn(num_classes=num_classes, blocks=28, width_factor=10, model_name="wrn28_10", **kwargs)
    net.features.final_pool = nn.AdaptiveAvgPool2d((1, 1))
    return net
