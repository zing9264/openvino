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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.mobilenetV3 import MobileNetV3 as mobilenet

warnings.filterwarnings("ignore")

# python gradcam_comparison.py
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='kvasirV2/demo/', type=str)
parser.add_argument('--num_classes', default=8, type=int)
parser.add_argument('--model_name', default='mobilenet', type=str)
parser.add_argument('--with_LIO', default=True, type=bool)
parser.add_argument('--model_norm_dir', default='./weights/intelComp/mobile-net/mobilenet_sgd.pth', type=str)
parser.add_argument('--model_LIO_dir', default='./weights/intelComp/mobile-net_w_LIO/np_3/mobilenet_sgd.pth', type=str)
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
torch.backends.cudnn.benchmark = True

device = "cpu"
print('{} will be used in the training process !!!'.format(device))

# Load model and weight
def load_weight(model, model_dir):
    try:
        assert os.path.isfile(model_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(model_dir,  map_location=device)
        model.load_state_dict(checkpoint['model'])
        print('{} : weights loaded!'.format(model_dir))
    except Exception as e:
        print('{} : weights NOT loaded!'.format(model_dir))
        print(e)

    return model

model_LIO = mobilenet(in_dim=960, num_classes=NUM_CLASSES, size=int(7/448 * RESOLUTION), with_LIO=False)
model_NORM = mobilenet(in_dim=960, num_classes=NUM_CLASSES, size=int(7/448 * RESOLUTION), with_LIO=False)
model_LIO.to(device), model_NORM.to(device)
model_LIO = load_weight(model_LIO, MODEL_LIO_DIR)
model_NORM = load_weight(model_NORM, MODEL_NORM_DIR)

# get target layer
target_layer_LIO = [list(model_LIO.children())[-4][16]]
target_layer_NORM = [list(model_NORM.children())[-4][16]]

# =============================================================================
# GradCAM visualization
# Construct the CAM object once, and then re-use it on many images:
cam_LIO = GradCAM(model=model_LIO, target_layers=target_layer_LIO )
cam_NORM = GradCAM(model=model_NORM, target_layers=target_layer_NORM)
target_category = None

with open('class-index.json') as f:
    data = json.load(f)

key_list = list(data.keys())
value_list = list(data.values())

df_test = pd.read_csv("testing.csv", dtype={'ImagePath': str, 'index': int})

# preprocessing function
preprocessing = transforms.Compose([
    transforms.CenterCrop((512,512)),
    transforms.Resize((RESOLUTION,RESOLUTION)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

files = os.listdir(IMAGE_DIR)
rows = len(files)
plt.figure(figsize=(15, 5*rows))
for fidx, file in enumerate(files):
    bg_image_class = df_test[df_test['ImagePath'].str.match(file)]['index'].values[0]
    image = Image.open(IMAGE_DIR + file)
    input_tensor = preprocessing(image.copy()).unsqueeze(0)

    bg_image = (np.array(image) / 255.0).astype('float32')
    bg_image = cv2.resize(bg_image, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
    bg_image = np.clip(bg_image, 0, 1)

    plt.subplot(rows, 3, fidx * 3 + 1)
    plt.imshow(bg_image)
    plt.title(key_list[value_list.index(bg_image_class)])
    plt.axis('off')

    titles = []
    outputs = model_LIO(input_tensor.to(device))
    _, predicted = outputs.max(1)
    titles.append('LIO: ' + key_list[value_list.index(predicted + 1)])

    outputs = model_NORM(input_tensor.to(device))
    _, predicted = outputs.max(1)
    titles.append('NORM: ' + key_list[value_list.index(predicted + 1)])

    pidx = 1
    for cam, title in zip([cam_LIO, cam_NORM], titles):
        pidx += 1
        plt.subplot(rows, 3, fidx * 3 + pidx)
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(bg_image, grayscale_cam,use_rgb=True)
        plt.title(title)
        plt.imshow(visualization)
        plt.axis('off')

plt.show()
plt.savefig('demo.jpg')





