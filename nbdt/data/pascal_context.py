###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

from PIL import Image, ImageOps, ImageFilter
import os
import math
import random
import numpy as np
from tqdm import trange

import torch
import torch.utils.data as data

__all__ = names = ("PascalContext",)


class BaseDataset(data.Dataset):
    def __init__(
        self,
        root,
        split,
        mode=None,
        transform=None,
        target_transform=None,
        base_size=520,
        crop_size=480,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        if self.mode == "train":
            print(
                "BaseDataset: base_size {}, crop_size {}".format(base_size, crop_size)
            )

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def make_pred(self, x):
        return x + self.pred_offset

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.0))
        y1 = int(round((h - outsize) / 2.0))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        w, h = img.size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # final transform
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()


class PascalContext(BaseDataset):
    NUM_CLASS = 59

    def __init__(
        self,
        root="./data",
        split="train",
        mode=None,
        transform=None,
        target_transform=None,
        **kwargs
    ):
        super(PascalContext, self).__init__(
            root, split, mode, transform, target_transform, **kwargs
        )
        from detail import Detail

        # from detail import mask
        root = os.path.join(root, "PascalContext")
        annFile = os.path.join(root, "trainval_merged.json")
        imgDir = os.path.join(root, "JPEGImages")
        # training mode
        self.detail = Detail(annFile, imgDir, split)
        self.transform = transform
        self.target_transform = target_transform
        self.ids = self.detail.getImgs()
        # generate masks
        self._mapping = np.sort(
            np.array(
                [
                    0,
                    2,
                    259,
                    260,
                    415,
                    324,
                    9,
                    258,
                    144,
                    18,
                    19,
                    22,
                    23,
                    397,
                    25,
                    284,
                    158,
                    159,
                    416,
                    33,
                    162,
                    420,
                    454,
                    295,
                    296,
                    427,
                    44,
                    45,
                    46,
                    308,
                    59,
                    440,
                    445,
                    31,
                    232,
                    65,
                    354,
                    424,
                    68,
                    326,
                    72,
                    458,
                    34,
                    207,
                    80,
                    355,
                    85,
                    347,
                    220,
                    349,
                    360,
                    98,
                    187,
                    104,
                    105,
                    366,
                    189,
                    368,
                    113,
                    115,
                ]
            )
        )
        self.classes = [
            "background",
            "aeroplane",
            "mountain",
            "mouse",
            "track",
            "road",
            "bag",
            "motorbike",
            "fence",
            "bed",
            "bedclothes",
            "bench",
            "bicycle",
            "diningtable",
            "bird",
            "person",
            "floor",
            "boat",
            "train",
            "book",
            "bottle",
            "tree",
            "window",
            "plate",
            "platform",
            "tvmonitor",
            "building",
            "bus",
            "cabinet",
            "shelves",
            "light",
            "pottedplant",
            "wall",
            "car",
            "ground",
            "cat",
            "sidewalk",
            "truck",
            "ceiling",
            "rock",
            "chair",
            "wood",
            "food",
            "horse",
            "cloth",
            "sign",
            "computer",
            "sheep",
            "keyboard",
            "flower",
            "sky",
            "cow",
            "grass",
            "cup",
            "curtain",
            "snow",
            "water",
            "sofa",
            "dog",
            "door",
        ]
        self._key = np.array(range(len(self._mapping))).astype("uint8")
        mask_file = os.path.join(root, self.split + ".pth")
        print("mask_file:", mask_file)
        if os.path.exists(mask_file):
            self.masks = torch.load(mask_file)
        else:
            self.masks = self._preprocess(mask_file)

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert values[i] in self._mapping
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _preprocess(self, mask_file):
        masks = {}
        tbar = trange(len(self.ids))
        print(
            "Preprocessing mask, this will take a while."
            + "But don't worry, it only run once for each split."
        )
        for i in tbar:
            img_id = self.ids[i]
            mask = Image.fromarray(self._class_to_index(self.detail.getMask(img_id)))
            masks[img_id["image_id"]] = mask
            tbar.set_description("Preprocessing masks {}".format(img_id["image_id"]))
        torch.save(masks, mask_file)
        return masks

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = img_id["file_name"]
        iid = img_id["image_id"]
        img = Image.open(os.path.join(self.detail.img_folder, path)).convert("RGB")
        if self.mode == "test":
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(path)
        # convert mask to 60 categories
        mask = self.masks[iid]
        # synchrosized transform
        if self.mode == "train":
            img, mask = self._sync_transform(img, mask)
        elif self.mode == "val":
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == "testval"
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype("int32") - 1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.ids)

    @property
    def pred_offset(self):
        return 1
