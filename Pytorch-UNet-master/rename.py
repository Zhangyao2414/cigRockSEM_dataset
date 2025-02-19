# import os

# # 指定文件夹路径
# folder_path = "/Users/zhangyao2414/Downloads/Pytorch-UNet-master/data/imgs"

# # 获取文件夹下所有以'-image'结尾的文件的名称
# file_names = [f for f in os.listdir(folder_path) if f.endswith('-image')]

# # 如果file_names为空，则尝试使用更宽松的匹配方式
# if not file_names:
#     file_names = [f for f in os.listdir(folder_path) if '-image' in f]

# # 对每个文件名进行处理，修改文件名
# for file_name in file_names:
#     # 构造新的文件名，将'-image'替换为空字符串
#     new_file_name = file_name.replace('-image', '')
#     # 构造原始文件路径和新文件路径
#     original_file_path = os.path.join(folder_path, file_name)
#     new_file_path = os.path.join(folder_path, new_file_name)
#     # 使用os.rename()函数修改文件名
#     os.rename(original_file_path, new_file_path)



# import cv2
# import numpy as np
# from PIL import Image

# # 读取灰度图
# gray_img = cv2.imread('/home/yaozhang/research/Pytorch-UNet-master/data/masks/SLICE3_SEM_RESIZE_8bit-label.png', cv2.IMREAD_GRAYSCALE)

# # 扩展维度，将灰度图转换为三通道的形式
# rgb_img = np.expand_dims(gray_img, axis=2)
# rgb_img = np.repeat(rgb_img, 3, axis=2)

# # 保存 RGB 图像
# cv2.imwrite('/home/yaozhang/research/Pytorch-UNet-master/data/masks/SLICE3_SEM_RESIZE_8bit-label.png', rgb_img)

# gray_img = Image.open('/home/yaozhang/research/Pytorch-UNet-master/data/imgs/F1-1HF-ID-93-017_036-Mag-30000x-Scale-4um-Depth-3545.15m.png')
# print(gray_img.size)


# from PIL import Image
# import os

# def invert_colors(image_path):
#     # 打开图像
#     img = Image.open(image_path)

#     # 将图像转换为灰度模式（如果不是的话）
#     img = img.convert('L')

#     # 获取图像的大小
#     width, height = img.size

#     # 遍历每个像素并进行颜色交换
#     for y in range(height):
#         for x in range(width):
#             # 获取当前像素的颜色值
#             pixel = img.getpixel((x, y))

#             # 计算反色
#             inverted_pixel = 255 - pixel

#             # 将反色应用到当前像素上
#             img.putpixel((x, y), inverted_pixel)

#     # 将处理后的图像覆盖原图像
#     img.save(image_path)

# # 指定输入文件夹路径
# input_folder = '/home/yaozhang/research/Pytorch-UNet-master/data_3/masks'

# # 遍历输入文件夹中的所有图像文件
# for filename in os.listdir(input_folder):
#     if filename.endswith('.png') or filename.endswith('.jpg'):
#         input_path = os.path.join(input_folder, filename)

#         # 对每张图像进行颜色反转并直接覆盖原图像
#         invert_colors(input_path)

# print("Color inversion completed for all images.")

# import numpy as np
# from PIL import Image
# # 读取数组
# array_2d = np.loadtxt('array.txt')

# # 将二维数组恢复为三维数组
# array_3d = array_2d.reshape([256, 256, 3])
# array_3d = array_3d.astype(np.uint8)
# # print(array_3d)
# img = Image.fromarray(array_3d)
# img.show()



# import os
# import shutil

# # 定义图像的目录
# image_dir = '/home/yaozhang/research/Pytorch-UNet-master/data_3/masks'

# # 列出目录中的所有文件
# image_files = os.listdir(image_dir)
# # 对每个图像进行处理
# for image_file in image_files:
#     # 检查文件名是否包含'-label'
#     if '-label' in image_file:
#         # 创建新的文件名，将'-label'替换为''
#         new_image_file = image_file.replace('-label', '')

#         # 创建原始文件和新文件的完整路径
#         old_image_path = os.path.join(image_dir, image_file)
#         new_image_path = os.path.join(image_dir, new_image_file)

#         # 重命名文件
#         shutil.move(old_image_path, new_image_path)




import os
import shutil

def rename_files_in_folder(folder_path):
    # 遍历文件夹下的所有文件
    for filename in os.listdir(folder_path):
        # 构造文件的完整路径
        file_path = os.path.join(folder_path, filename)
        # 构造新的文件名
        base, ext = os.path.splitext(filename)
        new_filename = base + '-label' + ext
        new_file_path = os.path.join(folder_path, new_filename)
        # 重命名文件
        shutil.move(file_path, new_file_path)

# 指定要处理的文件夹路径
folder_path = '/home/yaozhang/research/Pytorch-UNet-master/data_3/masks'

# 重命名文件夹下的所有文件
rename_files_in_folder(folder_path)