import os
import cv2
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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.resnet import resnet_swap_2loss_add as resnet
from models.mobilenetV3 import MobileNetV3 as mobilenet

# prevent showing warnings
warnings.filterwarnings("ignore")

# print torch and cuda information
print('=========== torch & cuda infos ================')
print('torch version : ' + torch.__version__)
print('available: ' + str(torch.cuda.is_available()))
print('count: ' + str(torch.cuda.device_count()))
torch.backends.cudnn.benchmark = True


# python visualization_gradcam.py  --num_classes 8 --model_name mobilenet --with_LIO True --resolution 512 --model_dir weights/intelComp/mobile-net_w_LIO/mobilenet_sgd.pth
# python visualization_gradcam.py  --num_classes 8 --model_name mobilenet --with_LIO False --resolution 512 --model_dir weights/intelComp/mobile-net/mobilenet_sgd.pth
# python visualization_gradcam.py  --num_classes 8 --model_name resnet --with_LIO True --resolution 512 --model_dir weights/intelComp/resnet_w_LIO/resnet_sgd.pth
# python visualization_gradcam.py  --num_classes 8 --model_name resnet --with_LIO False --resolution 512 --model_dir weights/intelComp/resnet/resnet_sgd.pth

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='kvasirV2/test/00003.jpg', type=str)
parser.add_argument('--num_classes', default=9, type=int)
#parser.add_argument('--model_name', default='mobilenet', type=str)
parser.add_argument('--model_name', default='resnet', type=str)
parser.add_argument('--with_LIO', default=True, type=bool)
#parser.add_argument('--model_dir', default='weights/intelComp/mobile-net/mobilenet_sgd.pth', type=str)
parser.add_argument('--model_dir', default='weights/resnet/resnet_sgd.pth', type=str)
parser.add_argument('--resolution', default=512, type=int)
args = parser.parse_args()

# Params
IMG_PATH = args.image_path
MODEL_DIR = args.model_dir
MODEL_NAME = args.model_name
WITH_LIO = args.with_LIO
NUM_CLASSES = args.num_classes
RESOLUTION = args.resolution

# =============================================================================
# call out gpu is possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('{} will be used in the training process !!!'.format(device))


# =============================================================================
# Load image
image = Image.open(IMG_PATH)
preprocessing = transforms.Compose([
    transforms.CenterCrop((512,512)),
    transforms.Resize((RESOLUTION,RESOLUTION)),
])


# =============================================================================
# Load model
if MODEL_NAME == 'mobilenet':
    model = mobilenet(in_dim=960, num_classes=NUM_CLASSES, size=int(7/448 * RESOLUTION), with_LIO=False)
else:
    model = resnet(stage=3, in_dim=2048, num_classes=NUM_CLASSES, size=int(7/448 * RESOLUTION), with_LIO=False)
model.to(device) # model.cuda()

# load weight
try:
    assert os.path.isfile(MODEL_DIR), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(MODEL_DIR)
    model.load_state_dict(checkpoint['model'])
    print('weights loaded!')
except:
    pass


if MODEL_NAME == 'mobilenet':
    target_layer = [list(model.children())[-4][16]]
else:
    target_layer = [list(model.children())[-4][2]]
print('model target layers: ')
print(target_layer)

# =============================================================================
# GradCAM visualization
# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)

# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
target_category = None

input_tensor = preprocessing(image.copy()).unsqueeze(0)
bg_image = (np.array(image) / 255.0).astype('float32')
bg_image = cv2.resize(bg_image, (448, 448), interpolation=cv2.INTER_AREA)
# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(bg_image, grayscale_cam)
plt.imshow(visualization)
title = 'gradcam_result_{}.jpg'.format(MODEL_NAME)
if (WITH_LIO):
    title += "_with_LIO"
plt.axis('off')
plt.savefig('{}.jpg'.format(title))
plt.show()
