import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from matplotlib import pyplot as plt

# 图像文件夹路径
image_folder = '/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3/imgs_clahe/Cropped_512'
output_folder = '/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3/kmeans_seg'

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取文件夹下所有图像文件
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# 定义KMeans分割函数，并自动调整前景和背景
def kmeans_segmentation(image, k=2):
    # 将图像转换为二维数组
    pixel_values = image.reshape((-1, 1))  # 将图像展平
    pixel_values = np.float32(pixel_values)

    # 使用KMeans聚类
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(pixel_values)

    # 获取每个像素的标签
    labels = kmeans.labels_

    # 将标签转换为图像
    segmented_image = labels.reshape(image.shape)

    # 自动调整前景和背景
    cluster_0_mean = np.mean(image[segmented_image == 0])  # 类别0的像素均值
    cluster_1_mean = np.mean(image[segmented_image == 1])  # 类别1的像素均值
    
    if cluster_0_mean > cluster_1_mean:
        # 如果类别0的平均像素值较大，类别0为前景（255），类别1为背景（0）
        segmented_image = np.where(segmented_image == 0, 255, 0).astype(np.uint8)
    else:
        # 否则类别1为前景（255），类别0为背景（0）
        segmented_image = np.where(segmented_image == 1, 255, 0).astype(np.uint8)

    return segmented_image

# 遍历文件夹下的每个图像并进行处理
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    # 读取图像为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 预处理：高斯模糊减少噪声
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 使用KMeans进行图像分割，并自动调整前景和背景
    segmented_image = kmeans_segmentation(image_blurred)

    # 保存分割结果
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, segmented_image)

print("所有图像分割完成，结果已保存到文件夹:", output_folder)