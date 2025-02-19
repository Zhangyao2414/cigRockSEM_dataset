# import torch
# import torch.nn.functional as F
#
# from utils.data_loading import CarvanaDataset
# from torch.utils.data import DataLoader
# from utils.dice_score import *
#
# # 导入训练好的模型
# net = torch.load("/Users/zhangyao2414/Downloads/Pytorch-UNet-master/checkpoints/checkpoint_epoch1.pth")
#
# # 准备训练数据集
# test_dataset = CarvanaDataset(
#     images_dir="/Users/zhangyao2414/research/test/image",
#     mask_dir="/Users/zhangyao2414/research/test/label"
# )
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# # print(test_loader)
#
# def evaluate():
#     net.eval()
#     dice_score = 0
#     num_data = len(test_loader)
#     with torch.no_grad():
#         for data in test_loader:
#             img, label = data
#             label_pred = net(img)
#             assert label.min() >= 0 and label.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
#             # convert to one-hot format
#             label = F.one_hot(label, net.n_classes).permute(0, 3, 1, 2).float()
#             label_pred = F.one_hot(label_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
#             # compute the Dice score, ignoring background
#             dice_score += multiclass_dice_coeff(label_pred[:, 1:], label[:, 1:], reduce_batch_first=False)
#     return dice_score / max(num_data, 1)
#
# correct = evaluate()
#
# print(correct)
#

from PIL import Image
import numpy as np
img = Image.open('/Users/zhangyao2414/Downloads/Pytorch-UNet-master/data/labels/image_9_label.jpg')
# 将图像转换为灰度图像
gray_image = img.convert("L")

# 将灰度图像转换为NumPy数组
gray_array = np.array(gray_image)

# 在图像中随机选取100个点，并将它们的像素值赋为255
indices = np.random.choice(gray_array.size, size=50, replace=False)
gray_array.flat[indices] = 125

# 创建带有修改像素值的新图像
new_image = Image.fromarray(gray_array)

# 显示新图像
new_image.show()
