import random

import numpy as np
import torch
from PIL import Image
import math
import numpy as np
import random
from PIL import Image
import torchvision.transforms.functional as F

import torch
from torchvision import transforms
from torchvision.transforms import RandomApply

class RandomTransformCompose(transforms.Compose):
    def __init__(self, transforms):
        super(RandomTransformCompose, self).__init__(transforms)

    def __call__(self, sample):
        # img = sample['image']
        # mask = sample['label']
        for t in self.transforms:
            sample = t(sample)

        return sample

# class RandomTransformCompose(transforms.Compose):
#     def __init__(self, transforms):
#         super(RandomTransformCompose, self).__init__(transforms)

#     def __call__(self, image, label):
#         for t in self.transforms:
#             image, label = t(image, label)
#         return image, label


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__(p)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if self.p < random.random():
            return img, mask
        img = F.hflip(img)
        mask = F.hflip(mask)

        return {'image': img,
                'label': mask}


class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __init__(self, p=0.5):
        super(RandomVerticalFlip, self).__init__(p)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if self.p < random.random():
            return img, mask
        img = F.vflip(img)
        mask = F.vflip(mask)

        return {'image': img,
                'label': mask}


class RandomRotation(transforms.RandomRotation):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        super(RandomRotation, self).__init__(degrees, expand=expand, center=center)

    def __call__(self, image, label):
        angle = self.get_params(self.degrees)
        return F.rotate(image, angle, expand=self.expand, center=self.center), F.rotate(label, angle, expand=self.expand, center=self.center)


class RandomCrop(transforms.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super(RandomCrop, self).__init__(size, padding=padding, pad_if_needed=pad_if_needed, fill=fill,
                                         padding_mode=padding_mode)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        return {'image': img,
                'label': mask}


class Resize(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super(Resize, self).__init__(size, interpolation=interpolation)

    def __call__(self, image, label):
        return F.resize(image, self.size, self.interpolation), F.resize(label, self.size, self.interpolation)


# 示例使用
# image_transform = RandomTransformCompose([
#     RandomHorizontalFlip(),
#     RandomVerticalFlip(),
#     RandomRotation(60),
#     RandomCrop(64),
#     Resize((128, 128))
# ])
#
