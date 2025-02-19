# import os
# import shutil

# # 设置目标文件夹路径
# folder_path = '/home/yaozhang/research/pytorch-deeplab-xception-master/dataset_mudstone/PV/JPEGImages'

# # 获取目标文件夹中的所有文件名
# file_names = os.listdir(folder_path)
# print(len(file_names))

# # 遍历所有文件名
# for file_name in file_names:
#     # 检查文件名是否包含"-label"
#     if '-label' in file_name:
#         # 使用str.replace()方法去掉"-label"
#         new_file_name = file_name.replace('-label', '')
#         # 使用shutil.move()方法重命名文件
#         shutil.move(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))


import os
import shutil

# 设置目标文件夹路径
folder_path = '/home/yaozhang/research/pytorch-deeplab-xception-master/dataset_mudstone/PV/SegmentationClass'

# 获取目标文件夹中的所有文件名
file_names = [file for file in os.listdir(folder_path) if file.endswith('.png')]


# 遍历所有文件名
for file_name in file_names:
    # 分割文件名和后缀名
    name, ext = os.path.splitext(file_name)
    # 在文件名末尾添加"-label"
    new_name = name + '-label'
    # 重新连接文件名和后缀名
    new_file_name = new_name + ext
    # 使用shutil.move()方法重命名文件
    shutil.move(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))


# from PIL import Image
# import numpy as np

# img = Image.open('/home/yaozhang/research/pytorch-deeplab-xception-master/dataset_sandstone/PV/SegmentationClass')
# img = np.asarray(img)
# print(np.unique(img))
# # img = img*255
# img = Image.fromarray(img)
# img.show()