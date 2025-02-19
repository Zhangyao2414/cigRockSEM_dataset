import cv2
import os

def otsu_thresholding(input_folder, output_folder):
    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png'))]

    for image_file in image_files:
        # 构造图像文件的完整路径
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)
        
        # 读取图像
        image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"无法读取图像文件: {input_image_path}")
            continue
        
        # 使用大津法进行阈值分割
        _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 保存分割后的图像
        cv2.imwrite(output_image_path, thresholded_image)
        # print(f"处理并保存图像: {output_image_path}")

    print("所有图像处理完成！")

# 示例用法
input_folder = "/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3/imgs_clahe/Cropped_512"   # 替换为你的输入文件夹路径
output_folder = "/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3/oust_segmentation" # 替换为你的输出文件夹路径

otsu_thresholding(input_folder, output_folder)