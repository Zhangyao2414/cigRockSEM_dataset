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

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__(p)

    def __call__(self, image, label):
        if self.p < random.random():
            return image, label
        return F.hflip(image), F.hflip(label)


class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __init__(self, p=0.5):
        super(RandomVerticalFlip, self).__init__(p)

    def __call__(self, image, label):
        if self.p < random.random():
            return image, label
        return F.vflip(image), F.vflip(label)


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

    def __call__(self, image, label):
        i, j, h, w = self.get_params(image, self.size)
        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class Resize(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super(Resize, self).__init__(size, interpolation=interpolation)

    def __call__(self, image, label):
        return F.resize(image, self.size, self.interpolation), F.resize(label, self.size, self.interpolation)

#
# # 示例使用
# # image_transform = RandomTransformCompose([
# #     RandomHorizontalFlip(),
# #     RandomVerticalFlip(),
# #     RandomRotation(60),
# #     RandomCrop(64),
# #     Resize((128, 128))
# # ])
# #
# # img = Image.open('/Users/zhangyao2414/Downloads/Pytorch-UNet-master/data/imgs/image_1.jpg')
# # label = Image.open('/Users/zhangyao2414/Downloads/Pytorch-UNet-master/data/labels/image_1_label.jpg')
# # print(img.size)
# # img, label = image_transform(img, label)
# # img.show()
# # label.show()
# # print(img.size)
#
#
# # img = Image.open('/Users/zhangyao2414/Downloads/media_images_pred_84_5c450eda12857d37d933.png')
# # img = np.asarray(img)
# # # img = np.unique(img)
# # print(img)
#

# import albumentations as A
# import numpy as np
# from PIL import Image

# import cv2
# from albumentations.augmentations import transforms
# from albumentations.core.composition import Compose, OneOf
# from matplotlib import pyplot as plt

# pic1 = cv2.imread(
#     '/Users/zhangyao2414/Downloads/Pytorch-UNet-master/data/imgs/B5-1HF-ID-2_1_001-Mag-200x-Scale-500um-Depth-3729.2m-crop.png')
# pic2 = cv2.imread(
#     '/Users/zhangyao2414/Downloads/Pytorch-UNet-master/data/masks/B5-1HF-ID-2_1_001-Mag-200x-Scale-500um-Depth-3729.2m-crop-label.png')
# # pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
# # pic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2RGB)

# # print(pic1)

# # pic1 = pic1.astype(float) / 255.0
# # pic2 = pic2.astype(float) / 255.0

# # train_transform = A.RandomRotate90(p=1)

# train_transform = A.Compose([
#     A.Rotate(limit=45),  # 旋转角度限制在-45到45度之间
#     A.HorizontalFlip(p=0.5),  # 水平翻转概率为0.5
#     A.RandomCrop(width=1024, height=610),  # 随机裁剪为1024*610大小的图像
#     A.RandomBrightnessContrast(p=1),  # 随机改变亮度和对比度
#     OneOf([
#         A.ElasticTransform(p=0.5, alpha=120, sigma=120*0.05, alpha_affine=120*0.03),
#         A.GridDistortion(p=0.5),
#         A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
#     ], p=0.8)
# ])

# # 进行数据增强
# transformed = train_transform(image=pic1, mask=pic2)

# # 获取增强后的图像和遮罩
# pic1_augmented = transformed['image']
# pic2_augmented = transformed['mask']

# # pic1 = train_transform(image=pic1)['image']
# # pic2 = train_transform(image=pic2)['mask']

# image1 = Image.fromarray(pic1_augmented.astype(np.uint8))
# image2 = Image.fromarray(pic2_augmented.astype(np.uint8))
# # 显示图像
# image1.show()
# image2.show()
