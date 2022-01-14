import os
import cv2
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from models.mobilenetV3 import MobileNetV3 as mobilenet

warnings.filterwarnings("ignore")


def pil_loader(imgpath):
    with open(imgpath, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

# python gradcam_comparison.py
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='kvasirV2/demo/', type=str)
parser.add_argument('--num_classes', default=8, type=int)
parser.add_argument('--model_name', default='mobilenet', type=str)
parser.add_argument('--with_LIO', default=False, type=bool)
parser.add_argument('--model_norm_dir', default='weights/intelComp/mobile-net/mobilenet_sgd.pth', type=str)
parser.add_argument('--model_LIO_dir', default='weights/intelComp/mobile-net_w_LIO/np_3/mobilenet_sgd.pth', type=str)
parser.add_argument('--resolution', default=512, type=int)
args = parser.parse_args()

# Params
IMAGE_DIR = args.image_dir
MODEL_LIO_DIR = args.model_LIO_dir
MODEL_NORM_DIR = args.model_norm_dir
MODEL_NAME = args.model_name
WITH_LIO = args.with_LIO
NUM_CLASSES = args.num_classes
RESOLUTION = args.resolution


# print torch and cuda information
print('=========== torch & cuda infos ================')
print('torch version : ' + torch.__version__)
print('available: ' + str(torch.cuda.is_available()))
print('count: ' + str(torch.cuda.device_count()))
torch.backends.cudnn.benchmark = True


# =============================================================================
# GradCAM visualization
# Construct the CAM object once, and then re-use it on many images:

target_category = None

with open('class-index.json') as f:
    data = json.load(f)

key_list = list(data.keys())
value_list = list(data.values())

df_test = pd.read_csv("testing.csv", dtype={'ImagePath': str, 'index': int})

RESOLUTION=512
data_transforms = {
    'preprocess': transforms.Compose([
        transforms.RandomCrop((512,512)),
        transforms.RandomRotation(degrees=15),
        transforms.Resize((RESOLUTION,RESOLUTION)),
        transforms.RandomHorizontalFlip(),
    ]),
    'totensor': transforms.Compose([
        transforms.Resize((RESOLUTION,RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'None': transforms.Compose([
        transforms.CenterCrop((512,512)),
        transforms.Resize((RESOLUTION,RESOLUTION)),
    ]),
}



# preprocessing function
preprocessing = transforms.Compose([
    transforms.CenterCrop((512,512)),
    transforms.Resize((RESOLUTION,RESOLUTION)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

savepath="/home/openvino/sharedata/Pytorch-LIO-lastest/Pytorch-LIO20220102/Pytorch-LIO/kvasirV2/resize/"
files = os.listdir(IMAGE_DIR)
rows = len(files)
plt.figure(figsize=(15, 5*rows))
for fidx, file in enumerate(files):
    bg_image_class = df_test[df_test['ImagePath'].str.match(file)]['index'].values[0]
    image = pil_loader(IMAGE_DIR + file)
    image = data_transforms["None"](image)
    image = data_transforms["totensor"](image)
    print(file)
    print(savepath+file)
    save_image(image,savepath+file)




