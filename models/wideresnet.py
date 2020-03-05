from pytorchcv.models.wrn_cifar import get_wrn_cifar

__all__ = ('wrn28_10',)


def wrn28_10(num_classes=10, **kwargs):
    return get_wrn_cifar(num_classes=num_classes, blocks=28, width_factor=10, model_name="wrn28_10", **kwargs)
