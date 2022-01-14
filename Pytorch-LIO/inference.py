import os
import sys
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd
import time as tt

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
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--test_dir', default='kvasirV2/test', type=str)
parser.add_argument('--save_dir', default='weights/intelComp/resnet/', type=str)
parser.add_argument('--resolution', default=512, type=int)
parser.add_argument('--cuda', default=0, type=int)
args = parser.parse_args()
CUDA = args.cuda

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
# try:
#     torch.cuda.set_device(0)
# except:
#     pass
# #torch.cuda.set_device(CUDA)
#device = torch.device("cuda:{}".format(CUDA) if torch.cuda.is_available() else "cpu")
device="cpu"
print('{} will be used in the testing process !!!'.format(device))

if MODEL_NAME == 'mobilenet':
    model = mobilenet(in_dim=960, num_classes=NUM_CLASSES, size=int(7/448 * RESOLUTION), with_LIO=WITH_LIO)
else:
    model = resnet(stage=STAGE, in_dim=2048, num_classes=NUM_CLASSES, size=int(7/448 * RESOLUTION), with_LIO=WITH_LIO)
model.to(device) # model.cuda()


if device == "cuda":
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# load weight

print(MODEL_DIR)
print(SAVE_DIR)
checkpoint = torch.load(MODEL_DIR,  map_location=device)
model.load_state_dict(checkpoint['model'])
# start_epoch = checkpoint['epoch']
# if start_epoch >= EPOCHS:
    #start_epoch = 0
print('weights loaded!')


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


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def inference():
    model.eval()
    correct, total = 0, 0
    ii=0
    starttime= tt.perf_counter()
    pytorch_total_params = get_n_params(model)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader['test']):
            labels = torch.LongTensor(np.array(labels))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            ii+=1
            if (ii>=20):
                break
    endtime = tt.perf_counter()
    inference_time=endtime-starttime
    print("data counts :{} ".format(ii))
    print("total params :{} ".format(pytorch_total_params))

    print("inference time :{:.4f} s".format(inference_time))
    # validation loss and accuracy
    acc = 1. *correct / total
    print('accuracy: {:.4f}'.format(acc))


def converter():
    source_pth = "/home/openvino/sharedata/Pytorch-LIO/weights/intelComp/mobile-net/mobilenet_sgd.pth"
    target_pth = "/home/openvino/sharedata/Pytorch-LIO-lastest/Pytorch-LIO20220102/Pytorch-LIO/ONNX_Converter/mobilenet_sgd_nolio.onnx"
    # torch.load('filename.pth').to(device)
    batch_size = 1  #批处理大小
    input_shape = (3,RESOLUTION,RESOLUTION)   #输入数据
    input_data_shape = torch.randn(batch_size, *input_shape, device=device)
    torch.onnx.export(model, input_data_shape, target_pth, verbose=True)

inference()
#converter()


    


