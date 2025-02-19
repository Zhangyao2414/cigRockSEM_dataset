import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def iou_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of IoU coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim) - inter
    union = torch.where(union == 0, inter, union)

    iou = (inter + epsilon) / (union + epsilon)
    return iou.mean()

def mean_iou(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Mean IoU for all classes
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    input = input.argmax(dim=1)
    target = target.argmax(dim=1)

    intersect = (input * target).sum(dim=(-1, -2))
    union = input.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) - intersect

    iou = intersect / union.clamp(min=epsilon)
    return iou.mean()

def frequency_weighted_iou(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Frequency Weighted IoU for all classes
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    input = input.argmax(dim=1)
    target = target.argmax(dim=1)

    intersect = (input * target).sum(dim=(-1, -2))
    union = input.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) - intersect

    iou = intersect / union.clamp(min=epsilon)
    freq = target.sum(dim=(-1, -2)) / target.numel()
    return (freq * iou).sum()

def accuracy(input: Tensor, target: Tensor):
    # Accuracy for all classes
    assert input.size() == target.size()
    # assert input.dim() == 3

    input = input.argmax(dim=1)
    target = target.argmax(dim=1)

    correct = (input == target).sum().float()
    total = target.numel()
    return correct / total


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def multiclass_iou_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return iou_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def recall_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Recall for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    true_positive = (input * target).sum(dim=sum_dim)
    actual_positive = target.sum(dim=sum_dim)

    recall = (true_positive + epsilon) / (actual_positive + epsilon)
    return recall.mean()

def f1_score(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    precision = precision_coeff(input, target, reduce_batch_first, epsilon)
    recall = recall_coeff(input, target, reduce_batch_first, epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1.mean()

def precision_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Precision for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    true_positive = (input * target).sum(dim=sum_dim)
    predicted_positive = input.sum(dim=sum_dim)

    precision = (true_positive + epsilon) / (predicted_positive + epsilon)
    return precision.mean()

