import torch
import torch.nn as nn

from .misc_util import *

MODEL_DICT = dict()
ADAPTATION_MODULE_DICT = dict()
AUXILIARY_MODEL_WRAPPER_DICT = dict()
MODULE_DICT = get_classes_as_dict('torch.nn')


def register_model(arg=None, **kwargs):
    """
    Registers a model class or function to instantiate it.

    :param arg: class or function to be registered as a model.
    :type arg: class or typing.Callable or None
    :return: registered model class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The model will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.models.registry import register_model
        >>>
        >>> @register_model(key='my_custom_model')
        >>> class CustomModel(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom model class')

        In the example, ``CustomModel`` class is registered with a key "my_custom_model".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomModel`` class by
        "my_custom_model".
    """
    def _register_model(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        MODEL_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_model(arg)
    return _register_model


def register_adaptation_module(arg=None, **kwargs):
    """
    Registers an adaptation module class or function to instantiate it.

    :param arg: class or function to be registered as an adaptation module.
    :type arg: class or typing.Callable or None
    :return: registered adaptation module class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The adaptation module will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.models.registry import register_adaptation_module
        >>>
        >>> @register_adaptation_module(key='my_custom_adaptation_module')
        >>> class CustomAdaptationModule(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom adaptation module class')

        In the example, ``CustomAdaptationModule`` class is registered with a key "my_custom_adaptation_module".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomAdaptationModule`` class by
        "my_custom_adaptation_module".
    """
    def _register_adaptation_module(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        ADAPTATION_MODULE_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_adaptation_module(arg)
    return _register_adaptation_module


def register_auxiliary_model_wrapper(arg=None, **kwargs):
    """
    Registers an auxiliary model wrapper class or function to instantiate it.

    :param arg: class or function to be registered as an auxiliary model wrapper.
    :type arg: class or typing.Callable or None
    :return: registered auxiliary model wrapper class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The auxiliary model wrapper will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.models.registry import register_auxiliary_model_wrapper
        >>>
        >>> @register_auxiliary_model_wrapper(key='my_custom_auxiliary_model_wrapper')
        >>> class CustomAuxiliaryModelWrapper(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom auxiliary model wrapper class')

        In the example, ``CustomAuxiliaryModelWrapper`` class is registered with a key "my_custom_auxiliary_model_wrapper".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomAuxiliaryModelWrapper`` class by
        "my_custom_auxiliary_model_wrapper".
    """
    def _register_auxiliary_model_wrapper(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        AUXILIARY_MODEL_WRAPPER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_auxiliary_model_wrapper(arg)
    return _register_auxiliary_model_wrapper


def get_model(key, repo_or_dir=None, *args, **kwargs):
    """
    Gets a model from the model registry.

    :param key: model key.
    :type key: str
    :param repo_or_dir: ``repo_or_dir`` for torch.hub.load.
    :type repo_or_dir: str or None
    :return: model.
    :rtype: nn.Module
    """
    if key in MODEL_DICT:
        return MODEL_DICT[key](*args, **kwargs)
    elif repo_or_dir is not None:
        return torch.hub.load(repo_or_dir, key, *args, **kwargs)
    raise ValueError('model_name `{}` is not expected'.format(key))


def get_adaptation_module(key, *args, **kwargs):
    """
    Gets an adaptation module from the adaptation module registry.

    :param key: model key.
    :type key: str
    :return: adaptation module.
    :rtype: nn.Module
    """
    if key in ADAPTATION_MODULE_DICT:
        return ADAPTATION_MODULE_DICT[key](*args, **kwargs)
    elif key in MODULE_DICT:
        return MODULE_DICT[key](*args, **kwargs)
    raise ValueError('No adaptation module `{}` registered'.format(key))


def get_auxiliary_model_wrapper(key, *args, **kwargs):
    """
    Gets an auxiliary model wrapper from the auxiliary model wrapper registry.

    :param key: model key.
    :type key: str
    :return: auxiliary model wrapper.
    :rtype: nn.Module
    """
    if key in AUXILIARY_MODEL_WRAPPER_DICT:
        return AUXILIARY_MODEL_WRAPPER_DICT[key](*args, **kwargs)
    raise ValueError('No auxiliary model wrapper `{}` registered'.format(key))


DATASET_DICT = dict()
COLLATE_FUNC_DICT = dict()
SAMPLE_LOADER_DICT = dict()
BATCH_SAMPLER_DICT = dict()
TRANSFORM_DICT = dict()
DATASET_WRAPPER_DICT = dict()

DATASET_DICT.update(get_classes_as_dict('torchvision.datasets'))
BATCH_SAMPLER_DICT.update(get_classes_as_dict('torch.utils.data.sampler'))


def register_dataset(arg=None, **kwargs):
    """
    Registers a dataset class or function to instantiate it.

    :param arg: class or function to be registered as a dataset.
    :type arg: class or typing.Callable or None
    :return: registered dataset class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The dataset will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch.utils.data import Dataset
        >>> from torchdistill.datasets.registry import register_dataset
        >>> @register_dataset(key='my_custom_dataset')
        >>> class CustomDataset(Dataset):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom dataset class')

        In the example, ``CustomDataset`` class is registered with a key "my_custom_dataset".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomDataset`` class by
        "my_custom_dataset".
    """
    def _register_dataset(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        DATASET_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_dataset(arg)
    return _register_dataset


def register_collate_func(arg=None, **kwargs):
    """
    Registers a collate function.

    :param arg: function to be registered as a collate function.
    :type arg: typing.Callable or None
    :return: registered function.
    :rtype: typing.Callable

    .. note::
        The collate function will be registered as an option.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.datasets.registry import register_collate_func
        >>>
        >>> @register_collate_func(key='my_custom_collate')
        >>> def custom_collate(batch, label):
        >>>     print('This is my custom collate function')
        >>>     return batch, label

        In the example, ``custom_collate`` function is registered with a key "my_custom_collate".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``custom_collate`` function by
        "my_custom_collate".
    """
    def _register_collate_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__ if isinstance(func, (BuiltinMethodType, BuiltinFunctionType, FunctionType)) \
                else type(func).__name__

        COLLATE_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_collate_func(arg)
    return _register_collate_func


def register_sample_loader(arg=None, **kwargs):
    """
    Registers a sample loader class or function to instantiate it.

    :param arg: class or function to be registered as a sample loader.
    :type arg: class or typing.Callable or None
    :return: registered sample loader class or function to instantiate it.
    :rtype: class

    .. note::
        The sample loader will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch.utils.data import Sampler
        >>> from torchdistill.datasets.registry import register_sample_loader
        >>> @register_sample_loader(key='my_custom_sample_loader')
        >>> class CustomSampleLoader(Sampler):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom dataset class')

        In the example, ``CustomSampleLoader`` class is registered with a key "my_custom_sample_loader".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomSampleLoader`` class by
        "my_custom_sample_loader".
    """
    def _register_sample_loader_class(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        SAMPLE_LOADER_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_sample_loader_class(arg)
    return _register_sample_loader_class


def register_batch_sampler(arg=None, **kwargs):
    """
    Registers a batch sampler or function to instantiate it.

    :param arg: function to be registered as a batch sample loader.
    :type arg: typing.Callable or None
    :return: registered batch sample loader function.
    :rtype: typing.Callable

    .. note::
        The batch sampler will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch.utils.data import Sampler
        >>> from torchdistill.datasets.registry import register_batch_sampler
        >>> @register_batch_sampler(key='my_custom_batch_sampler')
        >>> class CustomSampleLoader(Sampler):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom dataset class')

        In the example, ``CustomSampleLoader`` class is registered with a key "my_custom_batch_sampler".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomSampleLoader`` class by
        "my_custom_batch_sampler".
    """
    def _register_batch_sampler(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        BATCH_SAMPLER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_batch_sampler(arg)
    return _register_batch_sampler


def register_transform(arg=None, **kwargs):
    """
    Registers a transform class or function to instantiate it.

    :param arg: class/function to be registered as a transform.
    :type arg: class or typing.Callable or None
    :return: registered transform class/function.
    :rtype: typing.Callable

    .. note::
        The transform will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.datasets.registry import register_transform
        >>> @register_transform(key='my_custom_transform')
        >>> class CustomTransform(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom transform class')

        In the example, ``CustomTransform`` class is registered with a key "my_custom_transform".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomTransform`` class by
        "my_custom_transform".
    """
    def _register_transform(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        TRANSFORM_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_transform(arg)
    return _register_transform


def register_dataset_wrapper(arg=None, **kwargs):
    """
    Registers a dataset wrapper class or function to instantiate it.

    :param arg: class/function to be registered as a dataset wrapper.
    :type arg: class or typing.Callable or None
    :return: registered dataset wrapper class/function.
    :rtype: typing.Callable

    .. note::
        The dataset wrapper will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch.utils.data import Dataset
        >>> from torchdistill.datasets.registry import register_dataset_wrapper
        >>> @register_transform(key='my_custom_dataset_wrapper')
        >>> class CustomDatasetWrapper(Dataset):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom dataset wrapper class')

        In the example, ``CustomDatasetWrapper`` class is registered with a key "my_custom_dataset_wrapper".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomDatasetWrapper`` class by
        "my_custom_dataset_wrapper".
    """
    def _register_dataset_wrapper(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        DATASET_WRAPPER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_dataset_wrapper(arg)
    return _register_dataset_wrapper


def get_dataset(key):
    """
    Gets a registered dataset class or function to instantiate it.

    :param key: unique key to identify the registered dataset class/function.
    :type key: str
    :return: registered dataset class or function to instantiate it.
    :rtype: class or typing.Callable
    """
    if key is None:
        return None
    elif key in DATASET_DICT:
        return DATASET_DICT[key]
    raise ValueError('No dataset `{}` registered'.format(key))


def get_collate_func(key):
    """
    Gets a registered collate function.

    :param key: unique key to identify the registered collate function.
    :type key: str or None
    :return: registered collate function.
    :rtype: typing.Callable
    """
    if key is None:
        return None
    elif key in COLLATE_FUNC_DICT:
        return COLLATE_FUNC_DICT[key]
    raise ValueError('No collate function `{}` registered'.format(key))


def get_sample_loader(key):
    """
    Gets a registered sample loader class or function to instantiate it.

    :param key: unique key to identify the registered sample loader class or function to instantiate it.
    :type key: str
    :return: registered sample loader class or function to instantiate it.
    :rtype: class or typing.Callable
    """
    if key is None:
        return None
    elif key in SAMPLE_LOADER_DICT:
        return SAMPLE_LOADER_DICT[key]
    raise ValueError('No sample loader `{}` registered.'.format(key))


def get_batch_sampler(key):
    """
    Gets a registered batch sampler class or function to instantiate it.

    :param key: unique key to identify the registered batch sampler class or function to instantiate it.
    :type key: str
    :return: registered batch sampler class or function to instantiate it.
    :rtype: class or typing.Callable
    """
    if key is None:
        return None

    if key not in BATCH_SAMPLER_DICT and key != 'BatchSampler':
        raise ValueError('No batch sampler `{}` registered.'.format(key))
    return BATCH_SAMPLER_DICT[key]


def get_transform(key):
    """
    Gets a registered transform class or function to instantiate it.

    :param key: unique key to identify the registered transform class or function to instantiate it.
    :type key: str
    :return: registered transform class or function to instantiate it.
    :rtype: class or typing.Callable
    """
    if key in TRANSFORM_DICT:
        return TRANSFORM_DICT[key]
    raise ValueError('No transform `{}` registered.'.format(key))


def get_dataset_wrapper(key):
    """
    Gets a registered dataset wrapper class or function to instantiate it.

    :param key: unique key to identify the registered dataset wrapper class or function to instantiate it.
    :type key: str
    :return: registered dataset wrapper class or function to instantiate it.
    :rtype: class or typing.Callable
    """
    if key in DATASET_WRAPPER_DICT:
        return DATASET_WRAPPER_DICT[key]
    raise ValueError('No dataset wrapper `{}` registered.'.format(key))


OPTIM_DICT = get_classes_as_dict('torch.optim')
SCHEDULER_DICT = get_classes_as_dict('torch.optim.lr_scheduler')


def register_optimizer(arg=None, **kwargs):
    """
    Registers an optimizer class or function to instantiate it.

    :param arg: class or function to be registered as an optimizer.
    :type arg: class or typing.Callable or None
    :return: registered optimizer class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The optimizer will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch.optim import Optimizer
        >>> from torchdistill.optim.registry import register_optimizer
        >>>
        >>> @register_optimizer(key='my_custom_optimizer')
        >>> class CustomOptimizer(Optimizer):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom optimizer class')

        In the example, ``CustomOptimizer`` class is registered with a key "my_custom_optimizer".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomOptimizer`` class by
        "my_custom_optimizer".
    """
    def _register_optimizer(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        OPTIM_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_optimizer(arg)
    return _register_optimizer


def register_scheduler(arg=None, **kwargs):
    """
    Registers a scheduler class or function to instantiate it.

    :param arg: class or function to be registered as a scheduler.
    :type arg: class or typing.Callable or None
    :return: registered scheduler class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The scheduler will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch.optim.lr_scheduler import LRScheduler
        >>> from torchdistill.optim.registry import register_scheduler
        >>>
        >>> @register_scheduler(key='my_custom_scheduler')
        >>> class CustomScheduler(LRScheduler):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom scheduler class')

        In the example, ``CustomScheduler`` class is registered with a key "my_custom_scheduler".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomScheduler`` class by
        "my_custom_scheduler".
    """
    def _register_scheduler(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        SCHEDULER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_scheduler(arg)
    return _register_scheduler


def get_optimizer(module, key, filters_params=True, *args, **kwargs):
    """
    Gets an optimizer from the optimizer registry.

    :param module: module to be added to optimizer.
    :type module: nn.Module
    :param key: optimizer key.
    :type key: str
    :param filters_params: if True, filers out parameters whose `required_grad = False`.
    :type filters_params: bool
    :return: optimizer.
    :rtype: Optimizer
    """
    is_module = isinstance(module, nn.Module)
    if key in OPTIM_DICT:
        optim_cls_or_func = OPTIM_DICT[key]
        if is_module and filters_params:
            params = module.parameters() if is_module else module
            updatable_params = [p for p in params if p.requires_grad]
            return optim_cls_or_func(updatable_params, *args, **kwargs)
        return optim_cls_or_func(module, *args, **kwargs)
    raise ValueError('No optimizer `{}` registered'.format(key))


def get_scheduler(optimizer, key, *args, **kwargs):
    """
    Gets a scheduler from the scheduler registry.

    :param optimizer: optimizer to be added to scheduler.
    :type optimizer: Optimizer
    :param key: scheduler key.
    :type key: str
    :return: scheduler.
    :rtype: LRScheduler
    """
    if key in SCHEDULER_DICT:
        return SCHEDULER_DICT[key](optimizer, *args, **kwargs)
    raise ValueError('No scheduler `{}` registered'.format(key))

LOSS_DICT = get_classes_as_dict('torch.nn.modules.loss')
LOW_LEVEL_LOSS_DICT = dict()
MIDDLE_LEVEL_LOSS_DICT = dict()
HIGH_LEVEL_LOSS_DICT = dict()
LOSS_WRAPPER_DICT = dict()
FUNC2EXTRACT_MODEL_OUTPUT_DICT = dict()


def register_low_level_loss(arg=None, **kwargs):
    """
    Registers a low-level loss class or function to instantiate it.

    :param arg: class or function to be registered as a low-level loss.
    :type arg: class or typing.Callable or None
    :return: registered low-level loss class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The low-level loss will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.losses.registry import register_low_level_loss
        >>>
        >>> @register_low_level_loss(key='my_custom_low_level_loss')
        >>> class CustomLowLevelLoss(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom low-level loss class')

        In the example, ``CustomLowLevelLoss`` class is registered with a key "my_custom_low_level_loss".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomLowLevelLoss`` class by
        "my_custom_low_level_loss".
    """
    def _register_low_level_loss(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        LOW_LEVEL_LOSS_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_low_level_loss(arg)
    return _register_low_level_loss


def register_mid_level_loss(arg=None, **kwargs):
    """
    Registers a middle-level loss class or function to instantiate it.

    :param arg: class or function to be registered as a middle-level loss.
    :type arg: class or typing.Callable or None
    :return: registered middle-level loss class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The middle-level loss will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.losses.registry import register_mid_level_loss
        >>>
        >>> @register_mid_level_loss(key='my_custom_mid_level_loss')
        >>> class CustomMidLevelLoss(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom middle-level loss class')

        In the example, ``CustomMidLevelLoss`` class is registered with a key "my_custom_mid_level_loss".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomMidLevelLoss`` class by
        "my_custom_mid_level_loss".
    """
    def _register_mid_level_loss(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        MIDDLE_LEVEL_LOSS_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_mid_level_loss(arg)
    return _register_mid_level_loss


def register_high_level_loss(arg=None, **kwargs):
    """
    Registers a high-level loss class or function to instantiate it.

    :param arg: class or function to be registered as a high-level loss.
    :type arg: class or typing.Callable or None
    :return: registered high-level loss class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The high-level loss will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.losses.registry import register_high_level_loss
        >>>
        >>> @register_high_level_loss(key='my_custom_high_level_loss')
        >>> class CustomHighLevelLoss(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom high-level loss class')

        In the example, ``CustomHighLevelLoss`` class is registered with a key "my_custom_high_level_loss".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomHighLevelLoss`` class by
        "my_custom_high_level_loss".
    """
    def _register_high_level_loss(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        HIGH_LEVEL_LOSS_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_high_level_loss(arg)
    return _register_high_level_loss


def register_loss_wrapper(arg=None, **kwargs):
    """
    Registers a loss wrapper class or function to instantiate it.

    :param arg: class or function to be registered as a loss wrapper.
    :type arg: class or typing.Callable or None
    :return: registered loss wrapper class or function to instantiate it.
    :rtype: class or typing.Callable

    .. note::
        The loss wrapper will be registered as an option.
        You can choose the registered class/function by specifying the name of the class/function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the class/function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torch import nn
        >>> from torchdistill.losses.registry import register_loss_wrapper
        >>>
        >>> @register_loss_wrapper(key='my_custom_loss_wrapper')
        >>> class CustomLossWrapper(nn.Module):
        >>>     def __init__(self, **kwargs):
        >>>         print('This is my custom loss wrapper class')

        In the example, ``CustomHighLevelLoss`` class is registered with a key "my_custom_loss_wrapper".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``CustomLossWrapper`` class by
        "my_custom_loss_wrapper".
    """
    def _register_loss_wrapper(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        LOSS_WRAPPER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_loss_wrapper(arg)
    return _register_loss_wrapper


def register_func2extract_model_output(arg=None, **kwargs):
    """
    Registers a function to extract model output.

    :param arg: function to be registered for extracting model output.
    :type arg: typing.Callable or None
    :return: registered function.
    :rtype: typing.Callable

    .. note::
        The function to extract model output will be registered as an option.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.losses.registry import register_func2extract_model_output
        >>>
        >>> @register_func2extract_model_output(key='my_custom_function2extract_model_output')
        >>> def custom_func2extract_model_output(batch, label):
        >>>     print('This is my custom collate function')
        >>>     return batch, label

        In the example, ``custom_func2extract_model_output`` function is registered with a key "my_custom_function2extract_model_output".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``custom_func2extract_model_output`` function by
        "my_custom_function2extract_model_output".
    """
    def _register_func2extract_model_output(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        FUNC2EXTRACT_MODEL_OUTPUT_DICT[key] = func
        return func

    if callable(arg):
        return _register_func2extract_model_output(arg)
    return _register_func2extract_model_output


def get_low_level_loss(key, **kwargs):
    """
    Gets a registered (low-level) loss module.

    :param key: unique key to identify the registered loss class/function.
    :type key: str
    :return: registered loss class or function to instantiate it.
    :rtype: nn.Module
    """
    if key in LOSS_DICT:
        return LOSS_DICT[key](**kwargs)
    elif key in LOW_LEVEL_LOSS_DICT:
        return LOW_LEVEL_LOSS_DICT[key](**kwargs)
    raise ValueError('No loss `{}` registered'.format(key))


def get_mid_level_loss(mid_level_criterion_config, criterion_wrapper_config=None):
    """
    Gets a registered middle-level loss module.

    :param mid_level_criterion_config: middle-level loss configuration to identify and instantiate the registered middle-level loss class.
    :type mid_level_criterion_config: dict
    :param criterion_wrapper_config: middle-level loss configuration to identify and instantiate the registered middle-level loss class.
    :type criterion_wrapper_config: dict
    :return: registered middle-level loss class or function to instantiate it.
    :rtype: nn.Module
    """
    loss_key = mid_level_criterion_config['key']
    mid_level_loss = MIDDLE_LEVEL_LOSS_DICT[loss_key](**mid_level_criterion_config['kwargs']) \
        if loss_key in MIDDLE_LEVEL_LOSS_DICT else get_low_level_loss(loss_key, **mid_level_criterion_config['kwargs'])
    if criterion_wrapper_config is None or len(criterion_wrapper_config) == 0:
        return mid_level_loss
    return get_loss_wrapper(mid_level_loss, criterion_wrapper_config)


def get_high_level_loss(criterion_config):
    """
    Gets a registered high-level loss module.

    :param criterion_config: high-level loss configuration to identify and instantiate the registered high-level loss class.
    :type criterion_config: dict
    :return: registered high-level loss class or function to instantiate it.
    :rtype: nn.Module
    """
    criterion_key = criterion_config['key']
    args = criterion_config.get('args', None)
    kwargs = criterion_config.get('kwargs', None)
    if args is None:
        args = list()
    if kwargs is None:
        kwargs = dict()
    if criterion_key in HIGH_LEVEL_LOSS_DICT:
        return HIGH_LEVEL_LOSS_DICT[criterion_key](*args, **kwargs)
    raise ValueError('No high-level loss `{}` registered'.format(criterion_key))


def get_loss_wrapper(mid_level_loss, criterion_wrapper_config):
    """
    Gets a registered loss wrapper module.

    :param mid_level_loss: middle-level loss module.
    :type mid_level_loss: nn.Module
    :param criterion_wrapper_config: loss wrapper configuration to identify and instantiate the registered loss wrapper class.
    :type criterion_wrapper_config: dict
    :return: registered loss wrapper class or function to instantiate it.
    :rtype: nn.Module
    """
    wrapper_key = criterion_wrapper_config['key']
    args = criterion_wrapper_config.get('args', None)
    kwargs = criterion_wrapper_config.get('kwargs', None)
    if args is None:
        args = list()
    if kwargs is None:
        kwargs = dict()
    if wrapper_key in LOSS_WRAPPER_DICT:
        return LOSS_WRAPPER_DICT[wrapper_key](mid_level_loss, *args, **kwargs)
    raise ValueError('No loss wrapper `{}` registered'.format(wrapper_key))


def get_func2extract_model_output(key):
    """
    Gets a registered function to extract model output.

    :param key: unique key to identify the registered function to extract model output.
    :type key: str
    :return: registered function to extract model output.
    :rtype: typing.Callable
    """
    if key is None:
        key = 'extract_model_loss_dict'
    if key in FUNC2EXTRACT_MODEL_OUTPUT_DICT:
        return FUNC2EXTRACT_MODEL_OUTPUT_DICT[key]
    raise ValueError('No function to extract original output `{}` registered'.format(key))


PRE_EPOCH_PROC_FUNC_DICT = dict()
PRE_FORWARD_PROC_FUNC_DICT = dict()
FORWARD_PROC_FUNC_DICT = dict()
POST_FORWARD_PROC_FUNC_DICT = dict()
POST_EPOCH_PROC_FUNC_DICT = dict()


def register_pre_epoch_proc_func(arg=None, **kwargs):
    """
    Registers a pre-epoch process function for :class:`torchdistill.core.distillation.DistillationBox` and
    :class:`torchdistill.core.training.TrainingBox`.

    :param arg: function to be registered as a pre-epoch process function.
    :type arg: typing.Callable or None
    :return: registered pre-epoch process function.
    :rtype: typing.Callable

    .. note::
        The function will be registered as an option of the pre-epoch process function.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.core.interfaces.registry import register_pre_epoch_proc_func
        >>> @register_pre_epoch_proc_func(key='my_custom_pre_epoch_proc_func')
        >>> def new_pre_epoch_proc(self, epoch=None, **kwargs):
        >>>     print('This is my custom pre-epoch process function')

        In the example, ``new_pre_epoch_proc`` function is registered with a key "my_custom_pre_epoch_proc_func".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``new_pre_epoch_proc`` function by
        "my_custom_pre_epoch_proc_func".
    """
    def _register_pre_epoch_proc_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        PRE_EPOCH_PROC_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_pre_epoch_proc_func(arg)
    return _register_pre_epoch_proc_func


def register_pre_forward_proc_func(arg=None, **kwargs):
    """
    Registers a pre-forward process function for :class:`torchdistill.core.distillation.DistillationBox` and
    :class:`torchdistill.core.training.TrainingBox`.

    :param arg: function to be registered as a pre-forward process function.
    :type arg: typing.Callable or None
    :return: registered pre-forward process function.
    :rtype: typing.Callable

    .. note::
        The function will be registered as an option of the pre-forward process function.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.core.interfaces.registry import register_pre_forward_proc_func
        >>> @register_pre_forward_proc_func(key='my_custom_pre_forward_proc_func')
        >>> def new_pre_forward_proc(self, *args, **kwargs):
        >>>     print('This is my custom pre-forward process function')

        In the example, ``new_pre_forward_proc`` function is registered with a key "my_custom_pre_forward_proc_func".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``new_pre_forward_proc`` function by
        "my_custom_pre_forward_proc_func".
    """
    def _register_pre_forward_proc_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        PRE_FORWARD_PROC_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_pre_forward_proc_func(arg)
    return _register_pre_forward_proc_func


def register_forward_proc_func(arg=None, **kwargs):
    """
    Registers a forward process function for :class:`torchdistill.core.distillation.DistillationBox` and
    :class:`torchdistill.core.training.TrainingBox`.

    :param arg: function to be registered as a forward process function.
    :type arg: typing.Callable or None
    :return: registered forward process function.
    :rtype: typing.Callable

    .. note::
        The function will be registered as an option of the forward process function.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.core.interfaces.registry import register_forward_proc_func
        >>> @register_forward_proc_func(key='my_custom_forward_proc_func')
        >>> def new_forward_proc(model, sample_batch, targets=None, supp_dict=None, **kwargs):
        >>>     print('This is my custom forward process function')

        In the example, ``new_forward_proc`` function is registered with a key "my_custom_forward_proc_func".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``new_forward_proc`` function by
        "my_custom_forward_proc_func".
    """
    def _register_forward_proc_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        FORWARD_PROC_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_forward_proc_func(arg)
    return _register_forward_proc_func


def register_post_forward_proc_func(arg=None, **kwargs):
    """
    Registers a post-forward process function for :class:`torchdistill.core.distillation.DistillationBox` and
    :class:`torchdistill.core.training.TrainingBox`.

    :param arg: function to be registered as a post-forward process function.
    :type arg: typing.Callable or None
    :return: registered post-forward process function.
    :rtype: typing.Callable

    .. note::
        The function will be registered as an option of the post-forward process function.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.core.interfaces.registry import register_post_forward_proc_func
        >>> @register_post_forward_proc_func(key='my_custom_post_forward_proc_func')
        >>> def new_post_forward_proc(self, loss, metrics=None, **kwargs):
        >>>     print('This is my custom post-forward process function')

        In the example, ``new_post_forward_proc`` function is registered with a key "my_custom_post_forward_proc_func".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``new_post_forward_proc`` function by
        "my_custom_post_forward_proc_func".
    """
    def _register_post_forward_proc_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        POST_FORWARD_PROC_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_post_forward_proc_func(arg)
    return _register_post_forward_proc_func


def register_post_epoch_proc_func(arg=None, **kwargs):
    """
    Registers a post-epoch process function for :class:`torchdistill.core.distillation.DistillationBox` and
    :class:`torchdistill.core.training.TrainingBox`.

    :param arg: function to be registered as a post-epoch process function.
    :type arg: typing.Callable or None
    :return: registered post-epoch process function.
    :rtype: typing.Callable

    .. note::
        The function will be registered as an option of the post-epoch process function.
        You can choose the registered function by specifying the name of the function or ``key``
        you used for the registration, in a training configuration used for
        :class:`torchdistill.core.distillation.DistillationBox` or :class:`torchdistill.core.training.TrainingBox`.

        If you want to register the function with a key of your choice, add ``key`` to the decorator as below:

        >>> from torchdistill.core.interfaces.registry import register_post_epoch_proc_func
        >>> @register_post_epoch_proc_func(key='my_custom_post_epoch_proc_func')
        >>> def new_post_epoch_proc(self, metrics=None, **kwargs):
        >>>     print('This is my custom post-epoch process function')

        In the example, ``new_post_epoch_proc`` function is registered with a key "my_custom_post_epoch_proc_func".
        When you configure :class:`torchdistill.core.distillation.DistillationBox` or
        :class:`torchdistill.core.training.TrainingBox`, you can choose the ``new_post_epoch_proc`` function by
        "my_custom_post_epoch_proc_func".
    """
    def _register_post_epoch_proc_func(func):
        key = kwargs.get('key')
        if key is None:
            key = func.__name__

        POST_EPOCH_PROC_FUNC_DICT[key] = func
        return func

    if callable(arg):
        return _register_post_epoch_proc_func(arg)
    return _register_post_epoch_proc_func


def get_pre_epoch_proc_func(key):
    """
    Gets a registered pre-epoch process function.

    :param key: unique key to identify the registered pre-epoch process function.
    :type key: str
    :return: registered pre-epoch process function.
    :rtype: typing.Callable
    """
    if key in PRE_EPOCH_PROC_FUNC_DICT:
        return PRE_EPOCH_PROC_FUNC_DICT[key]
    raise ValueError('No pre-epoch process function `{}` registered'.format(key))


def get_pre_forward_proc_func(key):
    """
    Gets a registered pre-forward process function.

    :param key: unique key to identify the registered pre-forward process function.
    :type key: str
    :return: registered pre-forward process function.
    :rtype: typing.Callable
    """
    if key in PRE_FORWARD_PROC_FUNC_DICT:
        return PRE_FORWARD_PROC_FUNC_DICT[key]
    raise ValueError('No pre-forward process function `{}` registered'.format(key))


def get_forward_proc_func(key):
    """
    Gets a registered forward process function.

    :param key: unique key to identify the registered forward process function.
    :type key: str
    :return: registered forward process function.
    :rtype: typing.Callable
    """
    if key in FORWARD_PROC_FUNC_DICT:
        return FORWARD_PROC_FUNC_DICT[key]
    raise ValueError('No forward process function `{}` registered'.format(key))


def get_post_forward_proc_func(key):
    """
    Gets a registered post-forward process function.

    :param key: unique key to identify the registered post-forward process function.
    :type key: str
    :return: registered post-forward process function.
    :rtype: typing.Callable
    """
    if key in POST_FORWARD_PROC_FUNC_DICT:
        return POST_FORWARD_PROC_FUNC_DICT[key]
    raise ValueError('No post-forward process function `{}` registered'.format(key))


def get_post_epoch_proc_func(key):
    """
    Gets a registered post-epoch process function.

    :param key: unique key to identify the registered post-epoch process function.
    :type key: str
    :return: registered post-epoch process function.
    :rtype: typing.Callable
    """
    if key in POST_EPOCH_PROC_FUNC_DICT:
        return POST_EPOCH_PROC_FUNC_DICT[key]
    raise ValueError('No post-epoch process function `{}` registered'.format(key))