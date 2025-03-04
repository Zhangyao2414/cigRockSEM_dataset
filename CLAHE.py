import cv2
import os
from matplotlib import pyplot as plt

save_path = '/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3/imgs_clahe'

# 应用CLAHE到灰度图像的函数
def apply_clahe(grayscale_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # 创建一个CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # 对灰度图像应用CLAHE
    return clahe.apply(grayscale_img)


def apply_clahe_to_images(image_paths, clip_limit=2.0, tile_grid_size=(8, 8)):
    # List to hold the resulting CLAHE images
    clahe_images = []
    files_path = [image_path for image_path in os.listdir(image_paths) if image_path.endswith('.png')]
    # Apply CLAHE to each image in the list
    for images_path in files_path:
        # Load the image in grayscale
        print(images_path)
        image_path = os.path.join(image_paths, images_path)
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Apply CLAHE
        clahe_img = apply_clahe(gray_img, clip_limit, tile_grid_size)
        # Save the CLAHE image
        cv2.imwrite(os.path.join(save_path, images_path), clahe_img)
        # Add the resulting image to the list
        clahe_images.append(clahe_img)
    
    # Plot the histograms of the original and CLAHE images
    # plt.figure(figsize=(15, 5 * len(image_paths)))
    
    # for i, clahe_img in enumerate(clahe_images):
    #     plt.subplot(len(files_path), 2, 2*i + 1)
    #     plt.hist(clahe_img.ravel(), 256, [0,256])
    #     plt.title(f'CLAHE Histogram {i+1}')
    #     plt.xlabel('Pixel Intensity')
    #     plt.ylabel('Frequency')

    # plt.tight_layout()
    # plt.show()
    
    return clahe_images


apply_clahe_to_images('/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3') # Uncomment this line when you have actual image paths




# 使用示例：
# gray_img = cv2.imread('/Users/zhangyao2414/research/岩石电镜图片分割/sem_dataset_temp_20231106/image_3channel/image_3channel/F1-1HF-ID-138_018-Mag-30000x-Scale-4um-Depth-3639.1m.png', cv2.IMREAD_GRAYSCALE)
# clahe_img = apply_clahe(gray_img)
# cv2.imshow('CLAHE Image', clahe_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 计算灰度直方图
# histogram = cv2.calcHist([clahe_img], [0], None, [256], [0, 256])
# # print(histogram)

# # 绘制gray_img的灰度直方图
# gray_histogram = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
# plt.plot(gray_histogram, color='blue', label='Gray Image')

# # 绘制clahe_img的灰度直方图
# clahe_histogram = cv2.calcHist([clahe_img], [0], None, [256], [0, 256])
# plt.plot(clahe_histogram, color='red', label='CLAHE Image')

# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')
# plt.title('Grayscale Histogram')
# plt.legend()
# plt.show()
