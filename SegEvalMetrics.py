import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

def calculate_metrics(ground_truth, prediction):
    # Flatten the arrays to compute metrics
    gt_flat = ground_truth.flatten()
    pred_flat = prediction.flatten()

    # Calculate accuracy
    accuracy = accuracy_score(gt_flat, pred_flat)

    # Calculate IoU (Intersection over Union) for binary case
    iou = jaccard_score(gt_flat, pred_flat, pos_label=255)

    # Calculate Precision, Recall, F1-score
    precision = precision_score(gt_flat, pred_flat, pos_label=255)
    recall = recall_score(gt_flat, pred_flat, pos_label=255)
    f1 = f1_score(gt_flat, pred_flat, pos_label=255)

    return accuracy, iou, precision, recall, f1

def evaluate_otsu_segmentation(ground_truth_folder, segmented_folder):
    metrics = {
        'accuracy': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    # 获取标签图像和分割结果的文件列表
    ground_truth_files = [f for f in os.listdir(ground_truth_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for gt_file in ground_truth_files:
        gt_path = os.path.join(ground_truth_folder, gt_file)
        segmented_path = os.path.join(segmented_folder, gt_file)

        # 读取标签图像和分割结果
        ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        segmented = cv2.imread(segmented_path, cv2.IMREAD_GRAYSCALE)
        
        if ground_truth is None or segmented is None:
            print(f"无法读取图像文件: {gt_path} 或 {segmented_path}")
            continue
        
        # Binarize ground truth (assuming non-background pixels are labeled with 255)
        _, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
        _, segmented = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)

        # Calculate metrics
        accuracy, iou, precision, recall, f1 = calculate_metrics(ground_truth, segmented)
        
        metrics['accuracy'].append(accuracy)
        metrics['iou'].append(iou)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)

    # Calculate mean IoU (mIoU)
    miou = np.mean(metrics['iou'])

    # Print average metrics
    print("Evaluation Results:")
    print(f"Accuracy:  {np.mean(metrics['accuracy']):.4f}")
    print(f"mIoU:      {miou:.4f}")
    print(f"Precision: {np.mean(metrics['precision']):.4f}")
    print(f"Recall:    {np.mean(metrics['recall']):.4f}")
    print(f"F1-score:  {np.mean(metrics['f1']):.4f}")


# 示例用法
ground_truth_folder = "/Users/zhangyao2414/research/article/data_processed_up/shale/masks_threshold/Cropped_512"  # 替换为你的标签图像文件夹路径
segmented_folder = "/Users/zhangyao2414/research/article/data_processed_up/shale/imgs/imgs_filtered_min_3/kmeans_seg"        # 替换为你的分割结果文件夹路径

evaluate_otsu_segmentation(ground_truth_folder, segmented_folder)

# import cv2
# import os
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

# def calculate_metrics(ground_truth, prediction):
#     # Flatten the arrays to compute metrics
#     gt_flat = ground_truth.flatten()
#     pred_flat = prediction.flatten()

#     # 现在将0（背景）作为正类
#     accuracy = accuracy_score(gt_flat, pred_flat)
#     iou = jaccard_score(gt_flat, pred_flat, pos_label=0)
#     precision = precision_score(gt_flat, pred_flat, pos_label=0)
#     recall = recall_score(gt_flat, pred_flat, pos_label=0)
#     f1 = f1_score(gt_flat, pred_flat, pos_label=0)

#     return accuracy, iou, precision, recall, f1

# def evaluate_otsu_segmentation(ground_truth_folder, segmented_folder):
#     metrics = {
#         'accuracy': [],
#         'iou': [],
#         'precision': [],
#         'recall': [],
#         'f1': []
#     }

#     ground_truth_files = [f for f in os.listdir(ground_truth_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
#     for gt_file in ground_truth_files:
#         gt_path = os.path.join(ground_truth_folder, gt_file)
#         segmented_path = os.path.join(segmented_folder, gt_file)

#         ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
#         segmented = cv2.imread(segmented_path, cv2.IMREAD_GRAYSCALE)
        
#         if ground_truth is None or segmented is None:
#             print(f"无法读取图像文件: {gt_path} 或 {segmented_path}")
#             continue
        
#         _, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
#         _, segmented = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)

#         accuracy, iou, precision, recall, f1 = calculate_metrics(ground_truth, segmented)
        
#         metrics['accuracy'].append(accuracy)
#         metrics['iou'].append(iou)
#         metrics['precision'].append(precision)
#         metrics['recall'].append(recall)
#         metrics['f1'].append(f1)

#     miou = np.mean(metrics['iou'])

#     print("Evaluation Results (Background as Positive):")
#     print(f"Accuracy:  {np.mean(metrics['accuracy']):.4f}")
#     print(f"mIoU:      {miou:.4f}")
#     print(f"Precision: {np.mean(metrics['precision']):.4f}")
#     print(f"Recall:    {np.mean(metrics['recall']):.4f}")
#     print(f"F1-score:  {np.mean(metrics['f1']):.4f}")

# # 示例用法
# ground_truth_folder = "/Users/zhangyao2414/research/artical/masks"
# segmented_folder = "/Users/zhangyao2414/research/artical/imgs_clahe/imgs_oust"

# evaluate_otsu_segmentation(ground_truth_folder, segmented_folder)