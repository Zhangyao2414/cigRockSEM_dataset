import cv2
import os

def process_images(input_folder, output_folder):
    """
    Process all images in the input folder using adaptive mean thresholding
    and save the results to the output folder.
    
    Parameters:
    - input_folder: The folder containing input images.
    - output_folder: The folder where processed images will be saved.
    """
    # 创建输出文件夹（如果不存在的话）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否是图像文件
        if filename.endswith(('.png')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Error loading image: {image_path}")
                continue
            
            # 应用均值自适应阈值方法进行分割
            adaptive_thresh_image = cv2.adaptiveThreshold(
                image, 
                maxValue=255, 
                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
                thresholdType=cv2.THRESH_BINARY, 
                blockSize=205, 
                C=2
            )
            
            # 构造输出路径并保存处理后的图像
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, adaptive_thresh_image)
            # print(f"Processed image saved to: {output_path}")

# 示例用法
input_folder = '/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3/imgs_clahe/Cropped_512'  # 替换为你的输入文件夹路径
output_folder = '/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3/adaptive_thresh'  # 替换为你的输出文件夹路径

# 处理图像
process_images(input_folder, output_folder)