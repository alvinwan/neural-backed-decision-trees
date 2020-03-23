from .resnet import *
from .wideresnet import *
from pytorchcv.models.efficientnet import *
from torchvision.models import *


def get_model_choices():
    from types import ModuleType

    for key, value in globals().items():
        if not key.startswith('__') and not isinstance(value, ModuleType) and callable(value):
            yield key
