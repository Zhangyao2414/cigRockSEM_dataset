# import os
# import re
# import cv2

# # 设置目录路径
# directory_path = "/Users/zhangyao2414/research/data_raw/mudstone/imgs_clahe"
# destination_directory = "/Users/zhangyao2414/research/data_raw/mudstone/img_clahe_upsample"

# # 初始化数组
# numbers = []
# image_magnifications = []

# # 定义匹配模式
# pattern = r'_(\d{2})[Kk][xX]|(\d{2})K[xX]|(\d{2})_[Kk][Xx]'

# # 遍历目录中的所有文件
# for filename in os.listdir(directory_path):
#     # 只处理 .png 文件
#     if filename.endswith(".png"):
#         # 使用正则表达式提取Kx之前的两位数字
#         match = re.search(pattern, filename)
#         if match:
#             # 将提取到的数字添加到数组中
#             if match.group(1) is not None:
#                 magnifications = int(match.group(1))
#             elif match.group(2) is not None:
#                 magnifications = int(match.group(2))
#             elif match.group(3) is not None:
#                 magnifications = int(match.group(3))
#             numbers.append(magnifications)
#             image_magnifications.append((filename, magnifications))




# # # 计算数组中15和40的个数
# # count_15 = numbers.count(15)
# # count_40 = numbers.count(40)

# # # 输出结果
# print(f"一共有{len(numbers)}张图片")
# # print(f"数组中15的个数: {count_15}")
# # print(f"数组中40的个数: {count_40}")

# # for image, magnification in image_magnifications:
# #     print(f'The pic {image} has the magnification of {magnification}Kx')

# # -------------------------------------data processed-------------------------------------------

# # 处理图像
# for filename, magnification in image_magnifications:
#     # 构造源文件路径
#     source_path = os.path.join(directory_path, filename)

#     # 加载图像
#     image = cv2.imread(source_path)

#     # 判断放大倍数
#     if magnification == 15:
#         # 上采样图像使其达到40kx的效果
#         scale_factor = 40 / 15  # 放大比例
#         new_dimensions = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
#         upsampled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_NEAREST)

#         # 修改文件名，在文件名末尾加上_renew
#         file_name_without_extension, file_extension = os.path.splitext(filename)
#         new_filename = f"{file_name_without_extension}_renew{file_extension}"
#         destination_path = os.path.join(destination_directory, new_filename)

#         # 保存图像
#         cv2.imwrite(destination_path, upsampled_image)
#         print(f"The upsampled images has been saved: {new_filename}")

#     elif magnification == 40:
#         # 直接保存原始图像
#         destination_path = os.path.join(destination_directory, filename)
#         cv2.imwrite(destination_path, image)
#         print(f"The original images has been saved: {filename}")

# print("THE PROCESS IS OVER!")



import os
import re
import cv2

# 设置目录路径
directory_path = "/Users/zhangyao2414/research/artical/data_processed_up/mudstone/imgs"
# destination_directory = "/Users/zhangyao2414/research/data_raw/mudstone/img_clahe_upsample"

# 初始化数组
numbers = []
image_magnifications = []
resolutions = []

# 定义匹配模式
pattern = r'_(\d{2})[Kk][xX]|(\d{2})K[xX]|(\d{2})_[Kk][Xx]'

# 遍历目录中的所有文件
for filename in os.listdir(directory_path):
    # 只处理 .png 文件
    if filename.endswith(".png"):
        # 使用正则表达式提取Kx之前的两位数字
        match = re.search(pattern, filename)
        if match:
            # 将提取到的数字添加到数组中
            if match.group(1) is not None:
                magnifications = int(match.group(1))
            elif match.group(2) is not None:
                magnifications = int(match.group(2))
            elif match.group(3) is not None:
                magnifications = int(match.group(3))
            numbers.append(magnifications)
            image_magnifications.append((filename, magnifications))
            
            # 读取图像以获取宽和高
            img_path = os.path.join(directory_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                height, width, _ = img.shape  # 提取图像的高度和宽度

                # 根据 magnifications 赋值水平视野宽度
                if magnifications == 40:
                    hfw = 7.5  # 7.5 µm
                elif magnifications == 15:
                    hfw = 20.0  # 20 µm
                else:
                    hfw = None  # 如果 magnifications 不符合要求，跳过

                if hfw is not None:
                    # 计算分辨率（µm/pixel）
                    resolution = hfw / width
                    resolutions.append(resolution)

# 打印分辨率的最大值和最小值
if resolutions:
    print("Maximum resolution (µm/pixel):", max(resolutions))
    print("Minimum resolution (µm/pixel):", min(resolutions))
else:
    print("No valid images found with specified magnifications.")