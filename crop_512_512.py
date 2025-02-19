import cv2
import os

# # 指定输入和输出文件夹路径
input_directory = "/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3/imgs_clahe"
output_directory = "/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3/imgs_clahe/Cropped_512"

# 确保输出目录存在
os.makedirs(output_directory, exist_ok=True)

# 仅处理图像文件的扩展名
valid_extensions = (".png")

# 定义裁剪尺寸
crop_size = 512

files = [file for file in os.listdir(input_directory) if file.lower().endswith(valid_extensions)]

# 遍历输入目录中的所有图像文件
for filename in files:
    if filename.endswith(".png"):
        file_path = os.path.join(input_directory, filename)
        
        # 读取图像
        image = cv2.imread(file_path)
        if image is None:
            print(f"Can not read the image: {filename}")
            continue
        
        height, width = image.shape[:2]
        crop_count_1 = 0
        crop_count_2 = 0
        crop_count_3 = 0

        # 判断长和宽是否是crop_size的整数倍
        height_multiple = height % crop_size == 0
        width_multiple = width % crop_size == 0

        if height_multiple and width_multiple:
            print(f'The height and width of {filename} can be divided by an integer')
            # 从左上角开始裁剪
            for y in range(0, height, crop_size):
                for x in range(0, width, crop_size):
                    if y + crop_size <= height and x + crop_size <= width:
                        # 裁剪图像
                        crop = image[y:y + crop_size, x:x + crop_size]
                        crop_count_1 += 1
                        # 构造输出文件路径
                        output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_up_left_{crop_count_1}.png")
                        # 保存裁剪后的图像
                        cv2.imwrite(output_path, crop)

        elif height_multiple and not width_multiple:
            print(f'The height of {filename} can be divided by an integer, neither the width')
            # 从左上角开始裁剪
            # 循环裁剪图像，从左上角开始
            for y in range(0, height, crop_size):
                for x in range(0, width, crop_size):
                    if y + crop_size <= height and x + crop_size <= width:
                        # 裁剪图像
                        crop = image[y:y + crop_size, x:x + crop_size]
                        crop_count_1 += 1
                        # 构造输出文件路径
                        output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_up_left_{crop_count_1}.png")
                        # 保存裁剪后的图像
                        cv2.imwrite(output_path, crop)
            # 从右上角开始裁剪一列
            x = width - crop_size
            for y in range(0, height, crop_size):
                if y + crop_size <= height and x + crop_size <= width:
                    crop_2 = image[y:y+crop_size, x:width]
                    crop_count_2 += 1
                    output_path_2 = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_up_right_{crop_count_2}.png")
                    cv2.imwrite(output_path_2, crop_2)
            # 从右下角开始裁剪一个
            x = width - crop_size
            y = height - crop_size
            crop_4 = image[y:height, x:width]
            output_path_4 = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_down_right.png")
            cv2.imwrite(output_path_4, crop_4)

        elif width_multiple and not height_multiple:
            print(f'The width of {filename} can be divided by an integer, neither the height')
            # 从左上角开始裁剪
            # 循环裁剪图像，从左上角开始
            for y in range(0, height, crop_size):
                for x in range(0, width, crop_size):
                    if y + crop_size <= height and x + crop_size <= width:
                        # 裁剪图像
                        crop = image[y:y + crop_size, x:x + crop_size]
                        crop_count_1 += 1
                        # 构造输出文件路径
                        output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_up_left_{crop_count_1}.png")
                        # 保存裁剪后的图像
                        cv2.imwrite(output_path, crop)
            # 从左下角开始裁剪一行
            y = height - crop_size
            for x in range(0, width, crop_size):
                if y + crop_size <= height and x + crop_size <= width:
                    crop_3 = image[y:height, x:x+crop_size]
                    crop_count_3 += 1
                    output_path_3 = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_down_left_{crop_count_3}.png")
                    cv2.imwrite(output_path_3, crop_3)
            # 从右下角开始裁剪一个
            x = width - crop_size
            y = height - crop_size
            crop_4 = image[y:height, x:width]
            output_path_4 = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_down_right.png")
            cv2.imwrite(output_path_4, crop_4)

        elif not width_multiple and not height_multiple:
            # 从左上角开始裁剪
            # 循环裁剪图像，从左上角开始
            for y in range(0, height, crop_size):
                for x in range(0, width, crop_size):
                    if y + crop_size <= height and x + crop_size <= width:
                        # 裁剪图像
                        crop = image[y:y + crop_size, x:x + crop_size]
                        crop_count_1 += 1
                        # 构造输出文件路径
                        output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_up_left_{crop_count_1}.png")
                        # 保存裁剪后的图像
                        cv2.imwrite(output_path, crop)
            # 从右上角开始裁剪一列
            x = width - crop_size
            for y in range(0, height, crop_size):
                if y + crop_size <= height and x + crop_size <= width:
                    crop_2 = image[y:y+crop_size, x:width]
                    crop_count_2 += 1
                    output_path_2 = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_up_right_{crop_count_2}.png")
                    cv2.imwrite(output_path_2, crop_2)
            # 从左下角开始裁剪一行
            y = height - crop_size
            for x in range(0, width, crop_size):
                if y + crop_size <= height and x + crop_size <= width:
                    crop_3 = image[y:height, x:x+crop_size]
                    crop_count_3 += 1
                    output_path_3 = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_down_left_{crop_count_3}.png")
                    cv2.imwrite(output_path_3, crop_3)
            # 从右下角开始裁剪一个
            x = width - crop_size
            y = height - crop_size
            crop_4 = image[y:height, x:width]
            output_path_4 = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_down_right.png")
            cv2.imwrite(output_path_4, crop_4)
    print(f'The {filename} has been cropped!')

print("PROCESS IS OVER!")

# ----------------------------make sure the image can be cropped---------------------------------

# less_crop = []

# filenames = [file for file in os.listdir(input_directory) if file.endswith('.png')]
# for filename in filenames:
#     file_path = os.path.join(input_directory, filename)
#     img = cv2.imread(file_path)
#     height, width = img.shape[:2]
#     if height < 512 or width < 512:
#         less_crop.append((filename, height, width))

# print(less_crop)