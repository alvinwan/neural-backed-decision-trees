# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
import random

import torch
from torch.nn import functional as F
from torch.utils import data

__all__ = names = ("LookIntoPerson",)


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


class LookIntoPerson(BaseDataset):
    def __init__(
        self,
        root="./data/",
        list_path="LookIntoPerson/trainList.txt",
        num_samples=None,
        num_classes=20,
        multi_scale=True,
        flip=True,
        ignore_label=-1,
        base_size=473,
        crop_size=(473, 473),
        downsample_rate=1,
        scale_factor=11,
        center_crop_test=False,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):

        super(LookIntoPerson, self).__init__(
            ignore_label, base_size, crop_size, downsample_rate, scale_factor, mean, std
        )

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None
        self.classes = [
            "background",
            "hat",
            "hair",
            "glove",
            "sunglasses",
            "upper-clothes",
            "dress",
            "coat",
            "socks",
            "pants",
            "jumpsuits",
            "scarf",
            "skirt",
            "face",
            "left-arm",
            "right-arm",
            "left-leg",
            "right-leg",
            "left-shoe",
            "right-shoe",
        ]

        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [
            line.strip().split() for line in open(os.path.join(root, list_path))
        ]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item[:2]
            name = os.path.splitext(os.path.basename(label_path))[0]
            sample = {
                "img": image_path,
                "label": label_path,
                "name": name,
            }
            files.append(sample)
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(
            os.path.join(self.root, "LookIntoPerson/TrainVal_images/", item["img"]),
            cv2.IMREAD_COLOR,
        )
        label = cv2.imread(
            os.path.join(
                self.root, "LookIntoPerson/TrainVal_parsing_annotations/", item["label"]
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        size = label.shape

        if "testval" in self.list_path:
            image = cv2.resize(image, self.crop_size, interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :]
            label = label[:, ::flip]

            if flip == -1:
                right_idx = [15, 17, 19]
                left_idx = [14, 16, 18]
                for i in range(0, 3):
                    right_pos = np.where(label == right_idx[i])
                    left_pos = np.where(label == left_idx[i])
                    label[right_pos[0], right_pos[1]] = left_idx[i]
                    label[left_pos[0], left_pos[1]] = right_idx[i]

        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label, self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), name
