from torch.hub import load_state_dict_from_url
from pathlib import Path


def get_pretrained_model(
        arch, dataset, model, model_urls,
        pretrained=False,
        progress=True,
        root='.cache/torch/checkpoints'):
    if pretrained:
        state_dict = load_state_dict_from_key(
            arch, dataset, model_urls, pretrained, progress, root)
        model.load_state_dict(state_dict)
    return model

def load_state_dict_from_arch_dataset(
        arch, dataset, model_urls,
        pretrained=False,
        progress=True,
        root='.cache/torch/checkpoints'):
    if (arch, dataset) not in model_urls:
        raise UserWarning(
            f'The architecture {arch} for dataset {dataset} does not have a'
            ' pretrained model.'
        )
    return load_state_dict_from_url(
        model_urls[(arch, dataset)],
        Path.home() / root,
        progress=progress,
        check_hash=False)
