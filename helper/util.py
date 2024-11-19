from collections import OrderedDict

import torch
from torch import nn
from torch.nn import Module, Sequential
from torch.nn.parallel import DistributedDataParallel

from .registry import get_adaptation_module
from .constant import def_logger
from .file_util import make_parent_dirs
from .main_util import is_main_process, save_on_master
from .module_util import check_if_wrapped, get_module, get_frozen_param_names, get_updatable_param_names,\
    freeze_module_params

logger = def_logger.getChild(__name__)


def wrap_if_distributed(module, device, device_ids, distributed, find_unused_parameters=None, **kwargs):
    """
    Wraps ``module`` with DistributedDataParallel if ``distributed`` = True and ``module`` has any updatable parameters.

    :param module: module to be wrapped.
    :type module: nn.Module
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    :return: wrapped module if ``distributed`` = True and it contains any updatable parameters.
    :rtype: nn.Module
    """
    module.to(device)
    if distributed and len(get_updatable_param_names(module)) > 0:
        any_frozen = len(get_frozen_param_names(module)) > 0
        if find_unused_parameters is None:
            find_unused_parameters = any_frozen
        return DistributedDataParallel(module, device_ids=device_ids, find_unused_parameters=find_unused_parameters,
                                       **kwargs)
    return module


def load_module_ckpt(module, map_location, ckpt_file_path):
    """
    Loads checkpoint for ``module``.

    :param module: module to load checkpoint.
    :type module: nn.Module
    :param map_location: ``map_location`` for torch.load.
    :type map_location: torch.device or str or dict or typing.Callable
    :param ckpt_file_path: file path to load checkpoint.
    :type ckpt_file_path: str
    """
    state_dict = torch.load(ckpt_file_path, map_location=map_location)
    if check_if_wrapped(module):
        module.module.load_state_dict(state_dict)
    else:
        module.load_state_dict(state_dict)


def save_module_ckpt(module, ckpt_file_path):
    """
    Saves checkpoint of ``module``'s state dict.

    :param module: module to load checkpoint.
    :type module: nn.Module
    :param ckpt_file_path: file path to save checkpoint.
    :type ckpt_file_path: str
    """
    if is_main_process():
        make_parent_dirs(ckpt_file_path)
    state_dict = module.module.state_dict() if check_if_wrapped(module) else module.state_dict()
    save_on_master(state_dict, ckpt_file_path)


def add_submodule(module, module_path, module_dict):
    """
    Recursively adds submodules to `module_dict`.

    :param module: module.
    :type module: nn.Module
    :param module_path: module path.
    :type module_path: str
    :param module_dict: module dict.
    :type module_dict: nn.ModuleDict or dict
    """
    module_names = module_path.split('.')
    module_name = module_names.pop(0)
    if len(module_names) == 0:
        if module_name in module_dict:
            raise KeyError('module_name `{}` is already used.'.format(module_name))

        module_dict[module_name] = module
        return

    next_module_path = '.'.join(module_names)
    sub_module_dict = module_dict.get(module_name, None)
    if module_name not in module_dict:
        sub_module_dict = OrderedDict()
        module_dict[module_name] = sub_module_dict
    add_submodule(module, next_module_path, sub_module_dict)


def build_sequential_container(module_dict):
    """
    Builds sequential container (nn.Sequential) from ``module_dict``.

    :param module_dict: module dict to build sequential to build a sequential container.
    :type module_dict: nn.ModuleDict or collections.OrderedDict
    :return: sequential container.
    :rtype: nn.Sequential
    """
    for key in module_dict.keys():
        value = module_dict[key]
        if isinstance(value, OrderedDict):
            value = build_sequential_container(value)
            module_dict[key] = value
        elif not isinstance(value, Module):
            raise ValueError('module type `{}` is not expected'.format(type(value)))
    return Sequential(module_dict)


def redesign_model(org_model, model_config, model_label, model_type='original'):
    """
    Redesigns ``org_model`` and returns a new separate model e.g.,

    * prunes some modules from ``org_model``,
    * freezes parameters of some modules in ``org_model``, and
    * adds adaptation module(s) to ``org_model`` as a new separate model.

    .. note::
        The parameters and states of modules in ``org_model`` will be kept in a new redesigned model.

    :param org_model: original model to be redesigned.
    :type org_model: nn.Module
    :param model_config: configuration to redesign ``org_model``.
    :type model_config: dict
    :param model_label: model label (e.g., 'teacher', 'student') to be printed just for debugging purpose.
    :type model_label: str
    :param model_type: model type (e.g., 'original', name of model class, etc) to be printed just for debugging purpose.
    :type model_type: str
    :return: redesigned model.
    :rtype: nn.Module
    """
    frozen_module_path_set = set(model_config.get('frozen_modules', list()))
    module_paths = model_config.get('sequential', list())
    if not isinstance(module_paths, list) or len(module_paths) == 0:
        logger.info('Using the {} model'.format(model_type))
        if len(frozen_module_path_set) > 0:
            logger.info('Frozen module(s): {}'.format(frozen_module_path_set))

        isinstance_str = 'instance('
        for frozen_module_path in frozen_module_path_set:
            if frozen_module_path.startswith(isinstance_str) and frozen_module_path.endswith(')'):
                target_cls = nn.__dict__[frozen_module_path[len(isinstance_str):-1]]
                for m in org_model.modules():
                    if isinstance(m, target_cls):
                        freeze_module_params(m)
            else:
                module = get_module(org_model, frozen_module_path)
                freeze_module_params(module)
        return org_model

    logger.info('Redesigning the {} model with {}'.format(model_label, module_paths))
    if len(frozen_module_path_set) > 0:
        logger.info('Frozen module(s): {}'.format(frozen_module_path_set))

    module_dict = OrderedDict()
    adaptation_dict = model_config.get('adaptations', dict())

    for frozen_module_path in frozen_module_path_set:
        module = get_module(org_model, frozen_module_path)
        freeze_module_params(module)

    for module_path in module_paths:
        if module_path.startswith('+'):
            module_path = module_path[1:]
            adaptation_config = adaptation_dict[module_path]
            module = get_adaptation_module(adaptation_config['key'], **adaptation_config['kwargs'])
        else:
            module = get_module(org_model, module_path)

        if module_path in frozen_module_path_set:
            freeze_module_params(module)

        add_submodule(module, module_path, module_dict)
    return build_sequential_container(module_dict)


import copy

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from .constant import def_logger
from .registry import get_collate_func, get_batch_sampler, get_dataset_wrapper
from .wrapper import default_idx2subpath, BaseDatasetWrapper, CacheableDataset

logger = def_logger.getChild(__name__)


def split_dataset(dataset, lengths=None, generator_seed=None, sub_splits_configs=None, dataset_id=None):
    """
    Randomly splits ``dataset`` into sub datasets.

    :param dataset: dataset to be split.
    :type dataset: torch.utils.data.Dataset
    :param lengths: length ratios e.g., (9, 1) by default (if None).
    :type lengths: list[int]
    :param generator_seed: random seed for :meth:`torch.Generator().manual_seed`.
    :type generator_seed: int or None
    :param sub_splits_configs: sub-split configurations.
    :type sub_splits_configs: list[dict] or None
    :param dataset_id: dataset ID to be printed just for debugging purpose.
    :type dataset_id: str or None
    :return: sub-splits of ``dataset``.
    :rtype: list[torch.utils.data.Subset]
    """
    org_dataset_length = len(dataset)
    if dataset_id is not None:
        logger.info('Splitting `{}` dataset ({} samples in total)'.format(dataset_id, org_dataset_length))
    if lengths is None:
        lengths = (9, 1)

    total_length = sum(lengths)
    if total_length != org_dataset_length:
        lengths = [int((l / total_length) * org_dataset_length) for l in lengths]
        if len(lengths) > 1 and sum(lengths) != org_dataset_length:
            lengths[-1] = org_dataset_length - sum(lengths[:-1])

    sub_datasets = random_split(dataset, lengths) if generator_seed is None \
        else random_split(dataset, lengths, generator=torch.Generator().manual_seed(generator_seed))
    if sub_splits_configs is None:
        return sub_datasets

    # Deep-copy dataset to configure transforms independently as dataset in Subset class is shallow-copied
    for sub_dataset in sub_datasets:
        sub_dataset.dataset = copy.deepcopy(sub_dataset.dataset)

    assert len(sub_datasets) == len(sub_splits_configs), \
        'len(lengths) `{}` should be equal to len(sub_splits_configs) `{}`'.format(len(sub_datasets),
                                                                                   len(sub_splits_configs))
    for sub_dataset, sub_split_kwargs in zip(sub_datasets, sub_splits_configs):
        sub_split_kwargs = sub_split_kwargs.copy()
        transform = sub_split_kwargs.pop('transform', None)
        target_transform = sub_split_kwargs.pop('target_transform', None)
        transforms = sub_split_kwargs.pop('transforms', None)
        if hasattr(sub_dataset.dataset, 'transform') and transform is not None:
            sub_dataset.dataset.transform = transform
        if hasattr(sub_dataset.dataset, 'target_transform') and target_transform is not None:
            sub_dataset.dataset.target_transform = target_transform
        if hasattr(sub_dataset.dataset, 'transforms') and transforms is not None:
            sub_dataset.dataset.transforms = transforms
    return sub_datasets


def build_data_loader(dataset, data_loader_config, distributed, accelerator=None):
    """
    Builds a data loader for ``dataset``.

    :param dataset: dataset.
    :type dataset: torch.utils.data.Dataset
    :param data_loader_config: data loader configuration.
    :type data_loader_config: dict
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param accelerator: Hugging Face accelerator.
    :type accelerator: accelerate.Accelerator or None
    :return: data loader.
    :rtype: torch.utils.data.DataLoader
    """
    cache_dir_path = data_loader_config.get('cache_output', None)
    dataset_wrapper_config = data_loader_config.get('dataset_wrapper', None)
    if isinstance(dataset_wrapper_config, dict) and len(dataset_wrapper_config) > 0:
        dataset_wrapper_args = dataset_wrapper_config.get('args', None)
        dataset_wrapper_kwargs = dataset_wrapper_config.get('kwargs', None)
        if dataset_wrapper_args is None:
            dataset_wrapper_args = list()
        if dataset_wrapper_kwargs is None:
            dataset_wrapper_kwargs = dict()
        dataset_wrapper_cls_or_func = get_dataset_wrapper(dataset_wrapper_config['key'])
        dataset = dataset_wrapper_cls_or_func(dataset, *dataset_wrapper_args, **dataset_wrapper_kwargs)
    elif cache_dir_path is not None:
        dataset = CacheableDataset(dataset, cache_dir_path, idx2subpath_func=default_idx2subpath)
    elif data_loader_config.get('requires_supp', False):
        dataset = BaseDatasetWrapper(dataset)

    sampler_config = data_loader_config.get('sampler', dict())
    sampler_kwargs = sampler_config.get('kwargs', None)
    if sampler_kwargs is None:
        sampler_kwargs = dict()

    if distributed and accelerator is None:
        sampler = DistributedSampler(dataset, **sampler_kwargs)
    else:
        sampler_cls_or_func = sampler_config['class_or_func']
        sampler = sampler_cls_or_func(dataset, **sampler_kwargs)

    batch_sampler_config = data_loader_config.get('batch_sampler', None)
    batch_sampler_cls_or_func = None if batch_sampler_config is None else get_batch_sampler(batch_sampler_config['key'])
    batch_sampler = None if batch_sampler_cls_or_func is None \
        else batch_sampler_cls_or_func(sampler, **batch_sampler_config['kwargs'])
    collate_fn = get_collate_func(data_loader_config.get('collate_fn', None))
    data_loader_kwargs = data_loader_config['kwargs']
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, **data_loader_kwargs)
    return DataLoader(dataset, sampler=sampler, collate_fn=collate_fn, **data_loader_kwargs)


def build_data_loaders(dataset_dict, data_loader_configs, distributed, accelerator=None):
    """
    Builds data loaders for ``dataset_dict``.

    :param dataset_dict: dict of dataset tied with dataset ID as a key.
    :type dataset_dict: dict
    :param data_loader_configs: data loader configurations.
    :type data_loader_configs: list[dict]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param accelerator: Hugging Face accelerator.
    :type accelerator: accelerate.Accelerator or None
    :return: data loaders.
    :rtype: list[torch.utils.data.DataLoader]
    """
    data_loader_list = list()
    for data_loader_config in data_loader_configs:
        dataset_id = data_loader_config.get('dataset_id', None)
        data_loader = None if dataset_id is None or dataset_id not in dataset_dict \
            else build_data_loader(dataset_dict[dataset_id], data_loader_config, distributed, accelerator)
        data_loader_list.append(data_loader)
    return data_loader_list

from collections import abc

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.scatter_gather import gather

from .constant import def_logger
from .module_util import get_module, check_if_wrapped
from .constant import SELF_MODULE_PATH
from .forward_hook import register_forward_hook_with_dict

logger = def_logger.getChild(__name__)


def add_kwargs_to_io_dict(io_dict, module_path, **kwargs):
    """
    Adds kwargs to an I/O dict.

    :param io_dict: I/O dict.
    :type io_dict: dict
    :param module_path: module path.
    :type module_path: str
    :param kwargs: kwargs to be stored in ``io_dict``.
    :type kwargs: dict
    """
    io_dict[module_path] = kwargs


def _extract_module(org_model, sub_model, module_path):
    if module_path.startswith('+'):
        return get_module(sub_model, module_path[1:])
    return get_module(org_model, module_path)


def set_hooks(model, unwrapped_org_model, model_config, io_dict):
    """
    Sets forward hooks for target modules in model.

    :param model: model.
    :type model: nn.Module
    :param unwrapped_org_model: unwrapped original model.
    :type unwrapped_org_model: nn.Module
    :param model_config: model configuration.
    :type model_config: dict
    :param io_dict: I/O dict.
    :type io_dict: dict
    :return: list of pairs of module path and removable forward hook handle.
    :rtype: list[(str, torch.utils.hook.RemovableHandle)]
    """
    pair_list = list()
    forward_hook_config = model_config.get('forward_hook', dict())
    if len(forward_hook_config) == 0:
        return pair_list

    input_module_path_set = set(forward_hook_config.get('input', list()))
    output_module_path_set = set(forward_hook_config.get('output', list()))
    for target_module_path in input_module_path_set.union(output_module_path_set):
        requires_input = target_module_path in input_module_path_set
        requires_output = target_module_path in output_module_path_set
        add_kwargs_to_io_dict(io_dict, target_module_path)
        target_module = _extract_module(unwrapped_org_model, model, target_module_path)
        handle = register_forward_hook_with_dict(target_module, target_module_path,
                                                 requires_input, requires_output, io_dict)
        pair_list.append((target_module_path, handle))
    return pair_list


def wrap_model(model, model_config, device, device_ids=None, distributed=False,
               find_unused_parameters=False, any_updatable=True):
    """
    Wraps ``model`` with either DataParallel or DistributedDataParallel if specified.

    :param model: model.
    :type model: nn.Module
    :param model_config: model configuration.
    :type model_config: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool
    :param any_updatable: True if ``model`` contains any updatable parameters.
    :type any_updatable: bool
    :return: wrapped model (or ``model`` if wrapper is not specified).
    :rtype: nn.Module
    """
    wrapper = model_config.get('wrapper', None) if model_config is not None else None
    wrapper_kwargs = dict()
    if isinstance(wrapper, dict):
        wrapper_key = wrapper.get('key', None)
        wrapper_kwargs = wrapper.get('kwargs', wrapper_kwargs)
    else:
        wrapper_key = wrapper

    wrapper_kwargs['device_ids'] = device_ids
    model.to(device)
    if wrapper_key is not None and device.type.startswith('cuda') and not check_if_wrapped(model):
        if wrapper_key == 'DistributedDataParallel' and distributed and any_updatable:
            if 'find_unused_parameters' not in wrapper_kwargs:
                wrapper_kwargs['find_unused_parameters'] = find_unused_parameters

            model = DistributedDataParallel(model, **wrapper_kwargs)
        elif wrapper_key in {'DataParallel', 'DistributedDataParallel'}:
            model = DataParallel(model, **wrapper_kwargs)
    return model


def change_device(data, device):
    """
    Updates the device of tensor(s) stored in ``data``  with a new ``device``.

    :param data: data that contain tensor(s).
    :type data: Any
    :param device: new device.
    :type device: torch.device or str
    :return: ``data`` on the new ``device``.
    :rtype: Any
    """
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(change_device(samples, device) for samples in zip(*data)))
    elif isinstance(data, (list, tuple)):
        return elem_type(*(change_device(d, device) for d in data))
    elif isinstance(data, abc.Mapping):
        return {key: change_device(data[key], device) for key in data}
    elif isinstance(data, abc.Sequence):
        transposed = zip(*data)
        return [change_device(samples, device) for samples in transposed]
    return data


def tensor2numpy2tensor(data, device):
    """
    Converts tensor to numpy data and re-converts the numpy data to tensor.

    :param data: data that contain tensor(s).
    :type data: Any
    :param device: new device.
    :type device: torch.device or str
    :return: data that contain recreated tensor(s).
    :rtype: Any
    """
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return torch.Tensor(data.to(device).data.numpy())
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(tensor2numpy2tensor(samples, device) for samples in zip(*data)))
    elif isinstance(data, (list, tuple)):
        return elem_type(*(tensor2numpy2tensor(d, device) for d in data))
    elif isinstance(data, abc.Mapping):
        return {key: tensor2numpy2tensor(data[key], device) for key in data}
    elif isinstance(data, abc.Sequence):
        transposed = zip(*data)
        return [tensor2numpy2tensor(samples, device) for samples in transposed]
    return data


def clear_io_dict(model_io_dict):
    """
    Clears a model I/O dict's sub dict(s).

    :param model_io_dict: model I/O dict.
    :type model_io_dict: dict
    """
    for module_io_dict in model_io_dict.values():
        for sub_dict in list(module_io_dict.values()):
            sub_dict.clear()


def extract_io_dict(model_io_dict, target_device):
    """
    Extracts I/O dict, gathering tensors on ``target_device``.

    :param model_io_dict: model I/O dict.
    :type model_io_dict: dict
    :param target_device: target device.
    :type target_device: torch.device or str
    :return: extracted I/O dict.
    :rtype: dict
    """
    uses_cuda = target_device.type == 'cuda'
    gathered_io_dict = {SELF_MODULE_PATH: dict()}
    for module_path, module_io_dict in model_io_dict.items():
        gathered_io_dict[module_path] = dict()
        for io_type in list(module_io_dict.keys()):
            sub_dict = module_io_dict.pop(io_type)
            values = [sub_dict[key] for key in sorted(sub_dict.keys())]
            gathered_obj = gather(values, target_device) if uses_cuda and len(values) > 1 else values[-1]
            gathered_io_dict[module_path][io_type] = gathered_obj
    return gathered_io_dict


def update_io_dict(main_io_dict, sub_io_dict):
    """
    Updates an I/O dict with a sub I/O dict.

    :param main_io_dict: main I/O dict to be updated.
    :type main_io_dict: dict
    :param sub_io_dict: sub I/O dict.
    :type sub_io_dict: dict
    """
    for key, module_io_dict in sub_io_dict.items():
        for io_type, value in module_io_dict.items():
            if len(value) > 0:
                main_io_dict[key][io_type] = value


def extract_sub_model_io_dict(model_io_dict, index):
    """
    Extracts sub I/O dict from ``model_io_dict``.

    :param model_io_dict: model I/O dict.
    :type model_io_dict: dict
    :param index: sample index.
    :type index: int
    :return: extracted sub I/O dict.
    :rtype: dict
    """
    sub_model_output_dict = dict()
    for module_path, sub_model_io_dict in model_io_dict.items():
        tmp_dict = dict()
        for key, value in sub_model_io_dict.items():
            tmp_dict[key] = value[index]
        sub_model_output_dict[module_path] = tmp_dict
    return sub_model_output_dict


from .registry import register_func2extract_model_output


@register_func2extract_model_output
def extract_model_loss_dict(student_outputs, targets, **kwargs):
    """
    Extracts model's loss dict.

    :param student_outputs: student model's output.
    :type student_outputs: Amy
    :param targets: training targets (won't be used).
    :type targets: Amy
    :return: registered function to extract model output.
    :rtype: dict
    """
    model_loss_dict = dict()
    if isinstance(student_outputs, dict):
        model_loss_dict.update(student_outputs)
    return model_loss_dict
