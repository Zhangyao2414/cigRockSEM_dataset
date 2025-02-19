import os
import numpy as np

root = r"/home/yaozhang/research/pytorch-deeplab-xception-master/dataset/PV/JPEGImages"
output = r"/home/yaozhang/research/pytorch-deeplab-xception-master/dataset/PV/ImageSets/Segmentation"
filename = []

# 从存放原图的目录中遍历所有图像文件
for root, dir, files in os.walk(root):
    for file in files:
        # 排除非图像文件
        if file.endswith('.png'):
            filename.append(file[:-4])  # 去除后缀，存储
# print(len(filename))

# 打乱文件名列表
np.random.shuffle(filename)

# 划分训练集、测试集，默认比例6:2:2
train_len = int(np.floor(len(filename) * 0.8))
trainval_len = int(np.floor(len(filename) * 0.0))
val_len = len(filename) - train_len - trainval_len

train = filename[:train_len]
trainval = filename[train_len:train_len + trainval_len]
val = filename[train_len + trainval_len:]

# 分别写入train.txt, test.txt
with open(os.path.join(output, 'train.txt'), 'w') as f1, open(os.path.join(output, 'trainval.txt'), 'w') as f2, open(
        os.path.join(output, 'val.txt'), 'w') as f3:
    for i in train:
        f1.write(i + '\n')
    for i in trainval:
        f2.write(i + '\n')
    for i in val:
        f3.write(i + '\n')

print('成功！')

