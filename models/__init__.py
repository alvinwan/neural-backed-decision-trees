from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .linear import *
from .trees import *
from .wideresnet import *


def get_model_choices():
    from types import ModuleType

    for key in globals():
        if not key.startswith('__') and not isinstance(key, ModuleType):
            yield key
