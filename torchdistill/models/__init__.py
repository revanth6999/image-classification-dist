from . import adaptation, classification, densenet, wide_resnet
from .custom import CUSTOM_MODEL_DICT
from .registry import ADAPTATION_MODULE_DICT, AUXILIARY_MODEL_WRAPPER_DICT

MODEL_DICT = dict()

MODEL_DICT.update(ADAPTATION_MODULE_DICT)
MODEL_DICT.update(AUXILIARY_MODEL_WRAPPER_DICT)
MODEL_DICT.update(CUSTOM_MODEL_DICT)
