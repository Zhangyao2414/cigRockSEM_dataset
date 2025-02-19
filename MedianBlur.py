import cv2
import numpy as np
import os

# 输入文件夹路径和输出文件夹路径
input_folder = '/Users/zhangyao2414/research/article/data_processed_up/shale/imgs'
output_folder = '/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3_1'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 核大小
kernel_size = 3

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # 读取输入文件夹中的图像
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if image is not None:
            # 水平方向中值滤波（0度）
            horizontal_filtered = cv2.medianBlur(image, kernel_size)

            # # 垂直方向中值滤波（90度）
            # vertical_filtered = cv2.medianBlur(np.rot90(image, 1), kernel_size)
            # vertical_filtered = np.rot90(vertical_filtered, -1)  # 旋转回原方向

            # # 45度方向（主对角线）
            # diag45_filtered = cv2.medianBlur(np.rot90(image, 2), kernel_size)
            # diag45_filtered = np.rot90(diag45_filtered, -2)  # 旋转回原方向

            # # 135度方向（副对角线）
            # diag135_filtered = cv2.medianBlur(np.rot90(image, 3), kernel_size)
            # diag135_filtered = np.rot90(diag135_filtered, -3)  # 旋转回原方向

            # # 合并结果，选择每个像素的最小值（你也可以使用其他策略，如取平均值）
            # combined_filtered = np.minimum(np.minimum(horizontal_filtered, vertical_filtered),
            #                                np.minimum(diag45_filtered, diag135_filtered))

            # 构建输出文件路径
            output_path = os.path.join(output_folder, filename)

            # 保存滤波后的图像
            cv2.imwrite(output_path, horizontal_filtered)
            
            print(f'Processed and saved: {output_path}')
        else:
            print(f'Failed to load image: {image_path}')