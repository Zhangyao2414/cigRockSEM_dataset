import cv2
import os
import numpy as np

# 定义输入和输出文件夹路径
input_folder = "/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3/imgs_clahe/Cropped_512"
output_folder = "/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3/watered"

# 创建输出文件夹，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有 .png 文件
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # 读取图像
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 应用 Otsu 阈值处理
        oust_value, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # _, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)

        # # 显示阈值处理后的图像
        # cv2.imshow('Thresholded Image', thresh)

        # # 等待用户按下任意键后关闭窗口
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 去除噪点
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=2)

        # 寻找前景区域
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

        # 确定未知区域
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 标记连接区域
        _, markers = cv2.connectedComponents(sure_fg)

        # 增加1以确保背景是1
        markers = markers + 1

        # 标记未知区域为0
        markers[unknown == 255] = 0

        # 应用分水岭算法
        img_color = cv2.imread(img_path)
        markers = cv2.watershed(img_color, markers)

        # 分割结果：前景标记为255，背景标记为0
        segmented_img = np.zeros_like(img)
        segmented_img[markers > 1] = 255  # 前景
        segmented_img = cv2.bitwise_not(segmented_img)

        # 保存处理后的图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, segmented_img)

print("所有图像已使用分水岭算法分割为0和255两类，并保存到输出文件夹。")