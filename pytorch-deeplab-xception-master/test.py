import torch
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from modeling.deeplab import *

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

model = DeepLab(num_classes=2,
                        backbone='resnet',
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)
model = nn.DataParallel(model)
state_dict = torch.load('/home/yaozhang/research/run/PV/shale_all_100_2025/model_best.pth.tar')
mask_values = state_dict.pop('mask_values', [0, 1])
model.load_state_dict(state_dict['state_dict'])
model = model.to(device)

file_path = '/home/yaozhang/research/pytorch-deeplab-xception-master/dataset/PV/JPEGImages/B5-2HF-ID-163_018-Mag-20000x-Scale-5um-Depth-4168.4m-crop_up_left_2.png'
img = Image.open(file_path)
img = img.convert('RGB')
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
img = transform(img)
img = img.unsqueeze(0)
model.eval()
img = img.to(device)
pre = model(img)
pre = pre.argmax(dim=1)
print(pre.shape)
pre = pre[0].long().squeeze().cpu().numpy()
print(np.unique(pre))
pre = (pre.astype(np.uint8))
# pred = Image.fromarray(pre)
result = mask_to_image(pre, mask_values)
result.save('/home/yaozhang/research/pytorch-deeplab-xception-master/shale/B5-2HF-ID-163_018-Mag-20000x-Scale-5um-Depth-4168.4m-crop_up_left_2.png')
