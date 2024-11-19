from torch.nn import SyncBatchNorm
from torchvision import models

OFFICIAL_MODEL_DICT = dict()
OFFICIAL_MODEL_DICT.update(models.__dict__)
OFFICIAL_MODEL_DICT.update(models.detection.__dict__)
OFFICIAL_MODEL_DICT.update(models.segmentation.__dict__)


def get_image_classification_model(model_config, distributed=False):
    """
    Gets an image classification model from torchvision.

    :param model_config: image classification model configuration.
    :type model_config: dict
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :return: image classification model.
    :rtype: nn.Module
    """
    model_key = model_config['key']
    quantized = model_config.get('quantized', False)
    if not quantized and model_key in models.__dict__:
        model = models.__dict__[model_key](**model_config['kwargs'])
    elif quantized and model_key in models.quantization.__dict__:
        model = models.quantization.__dict__[model_key](**model_config['kwargs'])
    else:
        return None

    sync_bn = model_config.get('sync_bn', False)
    if distributed and sync_bn:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def get_object_detection_model(model_config):
    """
    Gets an object detection model from torchvision.

    :param model_config: object detection model configuration.
    :type model_config: dict
    :return: object detection model.
    :rtype: nn.Module
    """

    model_key = model_config['key']
    if model_key not in models.detection.__dict__:
        return None
    return models.detection.__dict__[model_key](**model_config['kwargs'])


def get_semantic_segmentation_model(model_config):
    """
    Gets a semantic segmentation model from torchvision.

    :param model_config: semantic segmentation model configuration.
    :type model_config: dict
    :return: semantic segmentation model.
    :rtype: nn.Module
    """
    model_key = model_config['key']
    if model_key not in models.segmentation.__dict__:
        return None
    return models.segmentation.__dict__[model_key](**model_config['kwargs'])


def get_vision_model(model_config):
    """
    Gets a computer vision model from torchvision.

    :param model_config: model configuration.
    :type model_config: dict
    :return: computer vision model.
    :rtype: nn.Module
    """
    model_key = model_config['key']
    return OFFICIAL_MODEL_DICT[model_key](**model_config['kwargs'])
