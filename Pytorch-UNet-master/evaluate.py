import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import f1_score, multiclass_dice_coeff, dice_coeff, iou_coeff, mean_iou, frequency_weighted_iou, accuracy, multiclass_iou_coeff, precision_coeff, recall_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score = 0
    mean_iou_score = 0
    freq_weighted_iou_score = 0
    acc_score = 0
    recall_score = 0  # 添加用于召回率的变量
    f1_score_value = 0  # 添加用于F1-score的变量
    precision_score = 0  # 添加用于Precision的变量

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'cuda' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                iou_score += iou_coeff(mask_pred, mask_true, reduce_batch_first=False)
                mean_iou_score += mean_iou(mask_pred, mask_true)
                freq_weighted_iou_score += frequency_weighted_iou(mask_pred, mask_true)
                acc_score += accuracy(mask_pred, mask_true)
                recall_score += recall_coeff(mask_pred, mask_true, reduce_batch_first=False)  # 计算召回率
                f1_score_value += f1_score(mask_pred, mask_true, reduce_batch_first=False)  # 计算F1-score
                precision_score += precision_coeff(mask_pred, mask_true, reduce_batch_first=False)  # 计算Precision

            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                iou_score += multiclass_iou_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                mean_iou_score += (multiclass_iou_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False) + iou_coeff(mask_pred[:, 0, :, :], mask_true[:, 0, :, :], reduce_batch_first=False)) / 2.0
                acc_score += accuracy(mask_pred, mask_true)
                recall_score += recall_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)  # 计算召回率
                f1_score_value += f1_score(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)  # 计算F1-score
                precision_score += precision_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)  # 计算Precision

    net.train()
    return dice_score / max(num_val_batches, 1), iou_score / max(num_val_batches, 1), mean_iou_score / max(num_val_batches, 1), acc_score / max(num_val_batches, 1), recall_score / max(num_val_batches, 1), f1_score_value / max(num_val_batches, 1), precision_score / max(num_val_batches, 1)  # 返回Precision