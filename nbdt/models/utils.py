from torch.hub import load_state_dict_from_url
from pathlib import Path
import torch


def get_pretrained_model(
        arch, dataset, model, model_urls,
        pretrained=False,
        progress=True,
        root='.cache/torch/checkpoints'):
    if pretrained:
        state_dict = load_state_dict_from_key(
            [(arch, dataset)], model_urls, pretrained, progress, root,
            device=get_model_device(model))
        state_dict = coerce_state_dict(state_dict, model.state_dict())
        model.load_state_dict(state_dict)
    return model

def coerce_state_dict(state_dict, reference_state_dict):
    if 'net' in state_dict:
        state_dict = state_dict['net']
    has_reference_module = list(reference_state_dict)[0].startswith('module.')
    has_module = list(state_dict)[0].startswith('module.')
    if not has_reference_module and has_module:
        state_dict = {
            key.replace('module.', '', 1): value
            for key, value in state_dict.items()
        }
    elif has_reference_module and not has_module:
        state_dict = {
            'module.' + key: value
            for key, value in state_dict.items()
        }
    return state_dict

def get_model_device(model):
    return next(model.parameters()).device

def load_state_dict_from_key(
        keys, model_urls,
        pretrained=False,
        progress=True,
        root='.cache/torch/checkpoints',
        device='cpu'):
    valid_keys = [key for key in keys if key in model_urls]
    if not valid_keys:
        raise UserWarning(
            f'None of the keys {keys} correspond to a pretrained model.'
        )
    return load_state_dict_from_url(
        model_urls[valid_keys[-1]],
        Path.home() / root,
        progress=progress,
        check_hash=False,
        map_location=torch.device(device))
