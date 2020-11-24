"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import os
import sys
import time
import math
import numpy as np

from urllib.request import urlopen, Request
from PIL import Image
import torch.nn as nn
import torch.nn.init as init
from pathlib import Path
import io

# tree-generation consntants
METHODS = ("wordnet", "random", "induced")
DATASETS = (
    "CIFAR10",
    "CIFAR100",
    "TinyImagenet200",
    "Imagenet1000",
    "Cityscapes",
    "PascalContext",
    "LookIntoPerson",
    "ADE20K",
)
DATASET_TO_NUM_CLASSES = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "TinyImagenet200": 200,
    "Imagenet1000": 1000,
    "Cityscapes": 19,
    "PascalContext": 59,
    "LookIntoPerson": 20,
    "ADE20K": 150,
}
DATASET_TO_CLASSES = {
    "CIFAR10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
}


def fwd():
    """Get file's working directory"""
    return Path(__file__).parent.absolute()


def dataset_to_default_path_graph(dataset):
    return hierarchy_to_path_graph(dataset, "induced")


def hierarchy_to_path_graph(dataset, hierarchy):
    return os.path.join(fwd(), f"hierarchies/{dataset}/graph-{hierarchy}.json")


def dataset_to_default_path_wnids(dataset):
    return os.path.join(fwd(), f"wnids/{dataset}.txt")


def get_directory(dataset, root="./nbdt/hierarchies"):
    return os.path.join(root, dataset)


def generate_kwargs(args, object, name="Dataset", globals={}, kwargs=None):
    kwargs = kwargs or {}

    for key in dir(object):
        accepts_key = getattr(object, key, False)
        if not key.startswith("accepts_") or not accepts_key:
            continue
        key = key.replace("accepts_", "", 1)
        assert key in args or callable(accepts_key)

        value = getattr(args, key, None)
        if callable(accepts_key):
            kwargs[key] = accepts_key(**globals)
            Colors.cyan(f"{key}:\t(callable)")
        elif accepts_key and value is not None:
            kwargs[key] = value
            Colors.cyan(f"{key}:\t{value}")
        elif value is not None:
            Colors.red(f"Warning: {name} does not support custom " f"{key}: {value}")
    return kwargs


def load_image_from_path(path):
    """Path can be local or a URL"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"
    }
    if "http" in path:
        request = Request(path, headers=headers)
        file = io.BytesIO(urlopen(request).read())
    else:
        file = path
    return Image.open(file)


def makeparentdirs(path):
    dir = Path(path).parent
    os.makedirs(dir, exist_ok=True)


class Colors:
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\x1b[36m"

    @classmethod
    def red(cls, *args):
        print(cls.RED + args[0], *args[1:], cls.ENDC)

    @classmethod
    def green(cls, *args):
        print(cls.GREEN + args[0], *args[1:], cls.ENDC)

    @classmethod
    def cyan(cls, *args):
        print(cls.CYAN + args[0], *args[1:], cls.ENDC)

    @classmethod
    def bold(cls, *args):
        print(cls.BOLD + args[0], *args[1:], cls.ENDC)


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
    _, term_width = os.popen("stty size", "r").read().split()
    term_width = int(term_width)
except Exception as e:
    print(e)
    term_width = 50

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def set_np_printoptions():
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def generate_checkpoint_fname(
    dataset,
    arch,
    path_graph,
    wnid=None,
    name="",
    trainset=None,
    include_labels=(),
    exclude_labels=(),
    include_classes=(),
    num_samples=0,
    tree_supervision_weight=1,
    fine_tune=False,
    loss="CrossEntropyLoss",
    lr=0.1,
    tree_supervision_weight_end=None,
    tree_supervision_weight_power=1,
    xent_weight=1,
    xent_weight_end=None,
    xent_weight_power=1,
    tree_start_epochs=None,
    tree_update_every_epochs=None,
    tree_update_end_epochs=None,
    **kwargs,
):
    fname = "ckpt"
    fname += "-" + dataset
    fname += "-" + arch
    if lr != 0.1:
        fname += f"-lr{lr}"
    if name:
        fname += "-" + name
    if path_graph and "TreeSupLoss" in loss:
        path = Path(path_graph)
        fname += "-" + path.stem.replace("graph-", "", 1)
    if include_labels:
        labels = ",".join(map(str, include_labels))
        fname += f"-incl{labels}"
    if exclude_labels:
        labels = ",".join(map(str, exclude_labels))
        fname += f"-excl{labels}"
    if include_classes:
        labels = ",".join(map(str, include_classes))
        fname += f"-incc{labels}"
    if num_samples != 0 and num_samples is not None:
        fname += f"-samples{num_samples}"
    if len(loss) > 1 or loss[0] != "CrossEntropyLoss":
        fname += f'-{",".join(loss)}'
        if tree_supervision_weight not in (None, 1):
            fname += f"-tsw{tree_supervision_weight}"
        if tree_supervision_weight_end not in (tree_supervision_weight, None):
            fname += f"-tswe{tree_supervision_weight_end}"
        if tree_supervision_weight_power not in (None, 1):
            fname += f"-tswp{tree_supervision_weight_power}"
        if xent_weight not in (None, 1):
            fname += f"-xw{xent_weight}"
        if xent_weight_end not in (xent_weight, None):
            fname += f"-xwe{xent_weight_end}"
        if xent_weight_power not in (None, 1):
            fname += f"-xwp{xent_weight_power}"
    if "SoftTreeLoss" in loss:
        if tree_start_epochs is not None:
            fname += f"-tse{tree_start_epochs}"
        if tree_update_every_epochs is not None:
            fname += f"-tueve{tree_update_every_epochs}"
        if tree_update_end_epochs is not None:
            fname += f"-tuene{tree_update_end_epochs}"
    return fname


def coerce_tensor(x, is_label=False):
    if is_label:
        return x.reshape(-1, 1)
    else:
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


def uncoerce_tensor(x, original_shape):
    n, c, h, w = original_shape
    return x.reshape(n, h, w, c).permute(0, 3, 1, 2)

