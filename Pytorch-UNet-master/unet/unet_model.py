""" Full assembly of the parts to form the complete network """
import numpy as np
import torch
from PIL import Image
from torch.nn.modules.module import T

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # print('picture_size', x.size())
        x1 = self.inc(x)
        # print('the 1 layer size', x1.size())
        x2 = self.down1(x1)
        # print('the 2 layer size', x2.size())
        x3 = self.down2(x2)
        # print('the 3 layer size', x3.size())
        x4 = self.down3(x3)
        # print('the 4 layer size', x4.size())
        x5 = self.down4(x4)
        # print('the 5 layer size', x5.size())
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

model = UNet(3, 2, bilinear=False)
image = Image.open('/home/yaozhang/research/pytorch-deeplab-xception-master/dataset_madstone_clahe/PV/JPEGImages/2B_10R_3_48_gen_view_pores_15KX_crop_siltpores_renew_down_left_1.png')
image = np.asarray(image)
image = image.reshape((1, 3, 512, 512))
image = torch.Tensor(image)
output = model(image)
print('picture_size', image.size())
print('output_size', output.size())


# import torch
# import torch.nn as nn
#
#
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#
#         # 编码器（Encoder）部分
#         self.encoder = nn.Sequential(
#             nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         # 中间层（Bottleneck）
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         # 解码器（Decoder）部分
#         self.decoder = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, n_classes, kernel_size=2, stride=2),
#             nn.ConvTranspose2d(n_classes, n_classes, kernel_size=2, stride=2)
#         )
#
#     def forward(self, x):
#         # 编码器
#         x1 = self.encoder(x)
#
#         # 中间层
#         x2 = self.bottleneck(x1)
#
#         # 解码器
#         x3 = self.decoder(x2)
#
#         return x3

