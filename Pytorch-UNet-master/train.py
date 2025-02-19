import argparse
import logging
import os
import random
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import RandomApply
from tqdm import tqdm
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
import albumentations as A
import matplotlib.pyplot as plt

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
import evl
import numpy as np


dir_img = Path('/home/yaozhang/research/pytorch-deeplab-xception-master/dataset/PV/JPEGImages')
dir_mask = Path('/home/yaozhang/research/pytorch-deeplab-xception-master/dataset/PV/SegmentationClass')
# dir_test_img = Path('/Users/zhangyao2414/research/test/image')
# dir_test_label = Path('/Users/zhangyao2414/research/test/label')
dir_checkpoint = Path('./checkpoints/')
sum_loss_shale = []
batch_loss_shale = []

os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"
# wandb.init(
#     project='U-Net',
#     resume='allow',
#     anonymous='allow',
#     mode='offline',
#     settings=wandb.Settings(temp_dir=os.path.expanduser('~/.wandb_tmp'))
# )

def train_model(
        model,
        device,
        epochs: int = 100,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1.0,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 0. Data Augmentation

    train_transform = evl.RandomTransformCompose([
        evl.RandomHorizontalFlip(),
        evl.RandomVerticalFlip(),
        # evl.RandomRotation(45)
        # evl.RandomCrop(256)
        # evl.Resize((256, 256))
    ])

    # train_transform = A.Compose([
    #     A.Rotate(limit=45),  # 旋转角度限制在-45到45度之间
    #     A.HorizontalFlip(p=0.5),  # 水平翻转概率为0.5
    #     A.RandomCrop(width=512, height=512),  # 随机裁剪为512*512大小的图像
    #     A.RandomBrightnessContrast(p=1),  # 随机改变亮度和对比度
    #     OneOf([
    #         A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    #         A.GridDistortion(p=0.5),
    #         A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
    #     ], p=0.8)
    # ], is_check_shapes=False)



    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale, transform=train_transform)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale, transform=None)


    # 2. Split into train / validation partitions
    n_val = int(len(dataset)*val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='allow')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'cuda' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                batch_loss_shale.append(loss.item())
                # print(epoch_loss)
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, iou_score, miou, acc, recall, F1, precision= evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        logging.info('Validation IoU score: {}'.format(iou_score))
                        logging.info('Validation Mean IoU score: {}'.format(miou))
                        # logging.info('Validation Frequency Weighted IoU score: {}'.format(fwiou))
                        logging.info('Validation Accuracy score: {}'.format(acc))
                        logging.info('Validation Recall score {}:'.format(recall))
                        logging.info('Validation F1 score {}:'.format(F1))
                        logging.info('Validation Precision score {}:'.format(precision))

                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'validation IoU': iou_score,
                                'validation Mean IoU': miou,
                                # 'validation Frequency Weighted IoU': fwiou,
                                'validation Accuracy': acc,
                                'validation Recall': recall,
                                'validation F1 score': F1,
                                'validation precision score': precision,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
        print(epoch_loss)
        sum_loss_shale.append(epoch_loss)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    # parser.add_argument('--device', type=int, default=0)
    # parser.add_argument('--use_gpu', default=True, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()


    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    # device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
    # device = torch.device("mps")
    logging.info(f'Using device {device}')
    # print('xxxxxxxxxxxxxx', args.classes)

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.CudaError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    # 指定要保存的文件路径
    # file_path = 'sum_loss_list.txt'
    # # 将列表中的每个元素写入文件5
    # with open(file_path, 'w') as f:
    #     for number in sum_loss_shale:
    #         f.write("%f\n" % number)
    # print("List saved to", file_path)

    # file = 'batch_loss_list.txt'
    # with open(file, 'w') as files:
    #     for number in batch_loss_shale:
    #         files.write("%f\n" % number)
    # print("List saved to", file)


    torch.save(model.state_dict(), "shale_100_all_2025.pth")
