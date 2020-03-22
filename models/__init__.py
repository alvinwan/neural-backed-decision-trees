from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .preact_resnet import *
from .efficientnet import *
from torchvision.models import *
from .resnet import *
from .wideresnet import *
from pytorchcv.models.efficientnet import *


def get_model_choices():
    from types import ModuleType

    for key, value in globals().items():
        if not key.startswith('__') and not isinstance(value, ModuleType) and callable(value):
            yield key
