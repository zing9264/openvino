# -*- coding: utf-8 -*-
import os
import sys
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from multiprocessing.reduction import ForkingPickler
from utils.dataset import collate_fn_test, dataset
from models.resnet import resnet_swap_2loss_add as resnet
from models.mobilenetV3 import MobileNetV3 as mobilenet

# prevent showing warnings
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

# print torch and cuda information
print('=========== torch & cuda infos ================')
print('torch version : ' + torch.__version__)
print('available: ' + str(torch.cuda.is_available()))
print('count: ' + str(torch.cuda.device_count()))
torch.backends.cudnn.benchmark = True

# python inference.py --batch_size 5 --model_name mobilenet --with_CBAM True --resolution 512 --num_classes 8 --save_dir weights/intelComp/mobile-net_CBAM_w_LIO/
# python inference.py --batch_size 5 --model_name mobilenet --num_classes 8 --resolution 512 --save_dir weights/intelComp/mobile-net_w_LIO/
# python inference.py --batch_size 5 --model_name mobilenet --num_classes 8 --resolution 512 --save_dir weights/intelComp/mobile-net/
# python inference.py --batch_size 5 --model_name resnet --num_classes 8 --resolution 512 --save_dir weights/intelComp/resnet_w_LIO/
# python inference.py --batch_size 5 --model_name resnet --num_classes 8 --resolution 512 --save_dir weights/intelComp/resnet/
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_worker', default=16, type=int)
parser.add_argument('--num_classes', default=8, type=int)
parser.add_argument('--model_name', default='mobilenet', type=str)
parser.add_argument('--with_LIO', default=True, type=bool)
parser.add_argument('--with_CBAM', default=False, type=bool)
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--test_dir', default='kvasirV2/test', type=str)
parser.add_argument('--save_dir', default='weights/intelComp/resnet/', type=str)
parser.add_argument('--resolution', default=512, type=int)
args = parser.parse_args()

# Params
STAGE = 3
TEST_DIR = args.test_dir
BATCH_SIZE = args.batch_size
NUM_WORKER = args.num_worker
SAVE_DIR = args.save_dir
MODEL_NAME = args.model_name
MODEL_DIR = '{}{}_{}.pth'.format(args.save_dir, args.model_name, args.optimizer)
NUM_CLASSES = args.num_classes
WITH_LIO = args.with_LIO
WITH_CBAM = args.with_CBAM
RESOLUTION = args.resolution

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

time = datetime.datetime.now()
# Read CSV
df_test = pd.read_csv("testing.csv", dtype={'ImagePath': str, 'index': int})

print('================ Dataset Info =====================')
print('test images:', df_test.shape)
print('num classes:', NUM_CLASSES)

# Data transforms (Preprocessing)
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


# Dataset
data_set = {
    'test': dataset(img_dir=TEST_DIR, anno_pd=df_test, num_positive=0, 
                     preprocess=data_transforms["None"], totensor=data_transforms["totensor"], train=False)   
}

# Dataloader
dataloader = {
    'test': torch.utils.data.DataLoader(data_set['test'], batch_size=BATCH_SIZE, shuffle=False, 
                                         num_workers=NUM_WORKER, collate_fn=collate_fn_test)
}


# =============================================================================
# call out gpu is possible
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"
print('{} will be used in the testing process !!!'.format(device))

if MODEL_NAME == 'mobilenet':
    model = mobilenet(in_dim=960, num_classes=NUM_CLASSES, with_LIO=WITH_LIO, with_CBAM=WITH_CBAM)
else:
    model = resnet(stage=STAGE, in_dim=2048, num_classes=NUM_CLASSES, size=int(7/448 * RESOLUTION), with_LIO=WITH_LIO, with_CBAM=WITH_CBAM)
model.to(device) # model.cuda()

if device == "cuda":
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# load weight
try:
    assert os.path.isdir(SAVE_DIR), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(MODEL_DIR)
    model.load_state_dict(checkpoint['model'])
    # start_epoch = checkpoint['epoch']
    # if start_epoch >= EPOCHS:
        #start_epoch = 0
    print('weights loaded!')
except:
    pass


# =============================================================================
# 关闭pytorch的shared memory功能 (Bus Error)
# Ref: https://github.com/huaweicloud/dls-example/issues/26
for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]


def inference():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader['test']):
            labels = torch.LongTensor(np.array(labels))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
    # validation loss and accuracy
    acc = 1. *correct / total
    print('accuracy: {}'.format(acc))

inference()







