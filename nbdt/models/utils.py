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
            device=model.device)
        model.load_state_dict(state_dict)
    return model

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
