import os
import json
from PIL import Image

import cv2
import numpy as np
import random

import torch
from torch.nn import functional as F
from torch.utils import data

__all__ = names = ("ADE20K",)


class BaseDataset(data.Dataset):
    def __init__(
        self,
        ignore_label=-1,
        base_size=2048,
        crop_size=(512, 1024),
        downsample_rate=1,
        scale_factor=16,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1.0 / downsample_rate

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        return np.array(label).astype("int32")

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=padvalue
            )

        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y : y + self.crop_size[0], x : x + self.crop_size[1]]
        label = label[y : y + self.crop_size[0], x : x + self.crop_size[1]]

        return image, label

    def center_crop(self, image, label):
        h, w = image.shape[:2]
        x = int(round((w - self.crop_size[1]) / 2.0))
        y = int(round((h - self.crop_size[0]) / 2.0))
        image = image[y : y + self.crop_size[0], x : x + self.crop_size[1]]
        label = label[y : y + self.crop_size[0], x : x + self.crop_size[1]]

        return image, label

    def image_resize(self, image, long_size, label=None):
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image

        return image, label

    def multi_scale_aug(self, image, label=None, rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        if label is not None:
            image, label = self.image_resize(image, long_size, label)
            if rand_crop:
                image, label = self.rand_crop(image, label)
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image

    def gen_sample(
        self, image, label, multi_scale=True, is_flip=True, center_crop_test=False
    ):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)

        if center_crop_test:
            image, label = self.image_resize(image, self.base_size, label)
            image, label = self.center_crop(image, label)

        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(
                label,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST,
            )

        return image, label


class ADE20K(BaseDataset):
    def __init__(
        self,
        root="./data/",
        list_path="ADE20K/training.odgt",
        num_samples=None,
        num_classes=150,
        multi_scale=True,
        flip=True,
        ignore_label=-1,
        base_size=512,
        crop_size=(512, 512),
        center_crop_test=False,
        downsample_rate=1,
        scale_factor=16,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):

        super(ADE20K, self).__init__(
            ignore_label, base_size, crop_size, downsample_rate, scale_factor, mean, std
        )

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.center_crop_test = center_crop_test

        self.img_list = [
            json.loads(x.rstrip()) for x in open(os.path.join(root, list_path), "r")
        ]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.classes = [
            "wall",
            "building",
            "sky",
            "floor",
            "tree",
            "ceiling",
            "road",
            "bed",
            "windowpane",
            "grass",
            "cabinet",
            "sidewalk",
            "person",
            "earth",
            "door",
            "table",
            "mountain",
            "plant",
            "curtain",
            "chair",
            "car",
            "water",
            "painting",
            "sofa",
            "shelf",
            "house",
            "sea",
            "mirror",
            "rug",
            "field",
            "armchair",
            "seat",
            "fence",
            "desk",
            "rock",
            "wardrobe",
            "lamp",
            "bathtub",
            "railing",
            "cushion",
            "pedestal",
            "box",
            "column",
            "signboard",
            "chest_of_drawers",
            "counter",
            "sand",
            "sink",
            "skyscraper",
            "fireplace",
            "refrigerator",
            "grandstand",
            "path",
            "stairs",
            "runway",
            "case",
            "pool_table",
            "pillow",
            "screen_door",
            "stairway",
            "river",
            "bridge",
            "bookcase",
            "blind",
            "coffee_table",
            "toilet",
            "flower",
            "book",
            "hill",
            "bench",
            "countertop",
            "stove",
            "palm_tree",
            "kitchen_island",
            "computer",
            "swivel_chair",
            "boat",
            "bar",
            "arcade_machine",
            "hovel",
            "bus",
            "towel",
            "light_source",
            "truck",
            "tower",
            "chandelier",
            "awning",
            "streetlight",
            "booth",
            "television_receiver",
            "airplane",
            "dirt_track",
            "apparel",
            "pole",
            "land",
            "handrail",
            "escalator",
            "ottoman",
            "bottle",
            "buffet",
            "poster",
            "stage",
            "van",
            "ship",
            "fountain",
            "conveyer_belt",
            "canopy",
            "washer",
            "toy",
            "swimming_pool",
            "stool",
            "barrel",
            "basket",
            "waterfall",
            "tent",
            "bag",
            "minibike",
            "cradle",
            "oven",
            "ball",
            "food",
            "step",
            "storage_tank",
            "brand",
            "microwave",
            "flowerpot",
            "animal",
            "bicycle",
            "lake",
            "dishwasher",
            "screen",
            "blanket",
            "sculpture",
            "exhaust_hood",
            "sconce",
            "vase",
            "traffic_light",
            "tray",
            "trash_can",
            "fan",
            "pier",
            "crt_screen",
            "plate",
            "monitor",
            "bulletin_board",
            "shower",
            "radiator",
            "drinking_glass",
            "clock",
            "flag",
        ]

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path = item["fpath_img"].replace("ADEChallengeData2016", "ADE20K")
            label_path = item["fpath_segm"].replace("ADEChallengeData2016", "ADE20K")
            name = os.path.splitext(os.path.basename(image_path))[0]
            files.append(
                {"img": image_path, "label": label_path, "name": name,}
            )
        return files

    def resize_image_label(self, image, label, size):
        scale = size / min(image.shape[0], image.shape[1])
        image = cv2.resize(
            image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        label = cv2.resize(
            label, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        return image, label

    def convert_label(self, label):
        # Convert labels to -1 to 149
        return np.array(label).astype("int32") - 1

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, item["img"]), cv2.IMREAD_COLOR)
        size = image.shape
        label = cv2.imread(os.path.join(self.root, item["label"]), cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        if "validation" in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            label = self.label_transform(label)
        else:
            image, label = self.resize_image_label(image, label, self.base_size)
            image, label = self.gen_sample(
                image, label, self.multi_scale, self.flip, self.center_crop_test
            )

        return image.copy(), label.copy(), np.array(size), name
