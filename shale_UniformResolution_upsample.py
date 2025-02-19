import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from numpy import mean
from collections import Counter

# 设置目录路径
directory_path = "/Users/zhangyao2414/research/artical/data_processed_up/shale/imgs"
destination_directory = "/Users/zhangyao2414/research/artical/data_processed_up/shale/masks"

# 初始化数组
numbers = []
image_magnifications = []
resolutions = []
file = [filename for filename in os.listdir(directory_path) if filename.endswith(".png")]
filenames = []

# 定义匹配模式
pattern = r"-(\d+(\.\d+)?)([xX]|[kK][xX])"
# pattern = r'(\d+)um'

# for filename in os.listdir(directory_path):
#     if filename.endswith('.png'):
#         match = re.search(pattern, filename)
#         if match:
#             num_str = match.group(1)
#             if num_str == '1':
#                 print(filename)
#         numbers.append(num_str)

# print(len(numbers))
# print(max(numbers))
# print(min(numbers))

# 遍历目录中的所有文件
for filename in os.listdir(directory_path):
    # 只处理 .png 文件
    if filename.endswith(".png"):
        # 使用正则表达式提取Kx之前的两位数字
        match = re.search(pattern, filename)
        if match:
            num_str = match.group(1)
            shffix = match.group(3)
            if shffix.lower() == "x":
                magnifications = float(num_str)
            elif shffix.lower() == "kx":
                magnifications = float(num_str) * 1000
            numbers.append(magnifications)
            image_magnifications.append((filename, magnifications))
            filenames.append(filename)
        
        # 读取图像以获取宽和高
        img_path = os.path.join(directory_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            height, width, _ = img.shape  # 提取图像的高度和宽度

            resolution = 100000/(magnifications*width)
            resolutions.append(resolution)

# 打印分辨率的最大值和最小值
if resolutions:
    print("Maximum resolution (µm/pixel):", max(resolutions))
    print("Minimum resolution (µm/pixel):", min(resolutions))
else:
    print("No valid images found with specified magnifications.")



# # 统计每个数的频次
# counter = Counter(numbers)

# # 获取列表中所有不同的数和它们的频次
# number = sorted(counter.keys())  # 按顺序排列数字
# frequencies = [counter[num] for num in number]

# # 找出频次最高的5个数字
# top_5_numbers = [num for num, freq in counter.most_common(5)]
# top_5_frequencies = [counter[num] for num in top_5_numbers]

# # 绘制折线图
# plt.plot(number, frequencies, marker='o', color='blue', linestyle='-')

# # 将频次最高的5个数字用红色虚线表示
# for num, freq in zip(top_5_numbers, top_5_frequencies):
#     plt.axvline(x=num, color='red', linestyle='--', linewidth=1)  # 在横坐标上画红色虚线
#     plt.text(num, max(frequencies) * 1.05, f'{int(num/1000)}', color='red', ha='center')  # 在顶部标注数字

# # 添加标题和标签
# plt.title('Frequency of Numbers in magnification List')
# plt.xlabel('Numbers')
# plt.ylabel('Frequency')

# # 显示图形
# plt.show()

# # 计算集中程度
# magnification_array = np.array(numbers)
# mean = np.mean(magnification_array)
# median = np.median(magnification_array)
# mode = stats.mode(magnification_array)

# # 计算离散程度
# variance = np.var(magnification_array)
# std_dev = np.std(magnification_array)
# data_range = np.ptp(magnification_array)
# iqr = stats.iqr(magnification_array)

# # 计算分布形状
# skewness = stats.skew(magnification_array)
# kurtosis = stats.kurtosis(magnification_array)

# # 计算极值
# max_value = np.max(magnification_array)
# min_value = np.min(magnification_array)
# q1 = np.percentile(magnification_array, 25)
# q3 = np.percentile(magnification_array, 75)

# # 输出结果
# print(f"Mean: {mean}, Median: {median}, Mode: {mode}")
# print(f"Variance: {variance}, Standard Deviation: {std_dev}, Range: {data_range}, IQR: {iqr}")
# print(f"Skewness: {skewness}, Kurtosis: {kurtosis}")
# print(f"Max: {max_value}, Min: {min_value}, Q1: {q1}, Q3: {q3}")

# 使用 Counter 统计每个数字出现的次数
# counts = Counter(numbers)

# # 获取横坐标和纵坐标
# x = list(counts.keys())  # numbers中的每个唯一数字
# y = list(counts.values())  # 每个数字的出现次数

# def fourth_largest_with_index(numbers):
#     if len(numbers) < 4:
#         return None  # 如果列表中少于四个元素，返回None

#     # 初始化最大值、第二大值、第三大值、第四大值及其索引
#     first = second = third = fourth = float('-inf')
#     first_index = second_index = third_index = fourth_index = -1
    
#     for i, num in enumerate(numbers):
#         if num > first:
#             fourth = third
#             fourth_index = third_index
#             third = second
#             third_index = second_index
#             second = first
#             second_index = first_index
#             first = num
#             first_index = i
#         elif first > num > second:
#             fourth = third
#             fourth_index = third_index
#             third = second
#             third_index = second_index
#             second = num
#             second_index = i
#         elif second > num > third:
#             fourth = third
#             fourth_index = third_index
#             third = num
#             third_index = i
#         elif third > num > fourth:
#             fourth = num
#             fourth_index = i
    
#     return fourth, fourth_index if fourth != float('-inf') else None

# result = fourth_largest_with_index(y)
# print(f'倍率为：{x[result[1]]},出现的次数为：{result[0]}')

# max_index = y.index(max(y))
# print(max_index)
# print(max(y))
# print(f'出现次数最多的放大倍数为：{x[max_index]}')


# # 输出结果
# print(f"一共有{len(numbers)}张图片")
# print(f"最大的放大倍数为：{max(numbers)}")
# print(f"最小的放大倍数为：{min(numbers)}")
# print(f"一共有{len(set(numbers))}种不同的放大倍数")

# for file, magnification in image_magnifications:
#     if magnification == 120000:
#         print(file)


# -------------------------------------data processed-------------------------------------------

# 定义放大倍数的区间和目标倍数
# magnification_ranges = [
#     (0, 1000, 1000),
#     (1000, 2000, 2000),
#     (2000, 5000, 5000),
#     (5000, 20000, 20000),
#     (20000, 50000, 50000),
#     (50000, 100000, 100000),
#     (100000, 120000, 120000)
# ]

# # 遍历 image_magnifications 列表
# for file, magnification in image_magnifications:
#     # 找到对应的目标放大倍数
#     target_magnification = None
#     for lower, upper, target in magnification_ranges:
#         if lower < magnification <= upper:
#             target_magnification = target
#             break
    
#     if target_magnification is None:
#         print(f"Can not find the right magnification: {file} (magnification: {magnification})")
#         continue
    
#     # 读取图像
#     image = cv2.imread(os.path.join(directory_path, file))
    
#     if image is not None:
#         if magnification != target_magnification:
#             # 计算缩放比例
#             scale_factor = target_magnification / magnification
            
#             # 获取新的尺寸
#             new_dimensions = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
            
#             # 调整图像大小
#             resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC)
            
#             # 构造新文件名，标识调整后的版本
#             file_name, file_extension = os.path.splitext(file)
#             new_file_name = f"{file_name}_resizeTo_{target_magnification/1000}KX{file_extension}"
            
#             # 构造完整的输出路径
#             output_path = os.path.join(destination_directory, new_file_name)
            
#             # 保存调整后的图像
#             cv2.imwrite(output_path, resized_image)
#         else:
#             # 构造完整的输出路径
#             output_path = os.path.join(destination_directory, file)
            
#             # 直接保存没有变化的图像
#             cv2.imwrite(output_path, image)
#     else:
#         print(f"Can not read the image: {file}")

# print("PROCESS IS OVER!")