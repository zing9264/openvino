#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import argparse
import logging as log
import sys
import cv2
from openvino.inference_engine import IECore, StatusCode
import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.mobilenetV3 import MobileNetV3 as mobilenet
import PIL.Image as Image
from PIL import ImageStat
from torchvision import transforms
import pandas as pd
import numpy as np
import time as tt

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    parser.add_argument('--image_dir', default='kvasirV2/demo/', type=str)
    parser.add_argument('--num_classes', default=8, type=int)
    parser.add_argument('--model_name', default='mobilenet', type=str)
    parser.add_argument('--with_LIO', default=True, type=bool)
    parser.add_argument('--model_norm_dir', default='./weights/intelComp/mobile-net/mobilenet_sgd.pth', type=str)
    parser.add_argument('--model_LIO_dir', default='./weights/intelComp/mobile-net_w_LIO/np_3/mobilenet_sgd.pth', type=str)
    parser.add_argument('--resolution', default=512, type=int)
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str,default="/home/openvino/sharedata/Pytorch-LIO-lastest/Pytorch-LIO20220102/Pytorch-LIO/ONNX_Converter/model/mobilenet_sgd.xml",
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-l', '--extension', type=str, default=None,
                      help='Optional. Required by the CPU Plugin for executing the custom operation on a CPU. '
                      'Absolute path to a shared library with the kernels implementations.')
    args.add_argument('-c', '--config', type=str, default=None,
                      help='Optional. Required by GPU or VPU Plugins for the custom operation kernel. '
                      'Absolute path to operation description file (.xml).')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, MYRIAD, HDDL or HETERO: '
                      'is acceptable. The sample will look for a suitable plugin for device specified. '
                      'Default value is CPU.')
    args.add_argument('--labels', default="labels.csv", type=str, help='Optional. Path to a labels mapping file.')
    args.add_argument('-nt', '--number_top', default=10, type=int, help='Optional. Number of top results.')

    # fmt: on
    return parser.parse_args()

def pil_loader(imgpath):
    with open(imgpath, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def param_to_string(metric) -> str:
    """Convert a list / tuple of parameters returned from IE to a string"""
    if isinstance(metric, (list, tuple)):
        return ', '.join([str(x) for x in metric])
    else:
        return str(metric)

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    args = parse_args()
    # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating Inference Engine')
    ie = IECore()
    if args.extension and args.device == 'CPU':
        log.info(f'Loading the {args.device} extension: {args.extension}')
        ie.add_extension(args.extension, args.device)
    if args.config and args.device in ('GPU', 'MYRIAD', 'HDDL'):
        log.info(f'Loading the {args.device} configuration: {args.config}')
        ie.set_config({'CONFIG_FILE': args.config}, args.device)

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    log.info(f'Reading the network: {args.model}')
    # (.xml and .bin files) or (.onnx file)
    net = ie.read_network(model=args.model)
    if len(net.input_info) != 1:
        log.error('Sample supports only single input topologies')
        return -1
    if len(net.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1
    # ---------------------------Step 3. Configure input & output----------------------------------------------------------
    log.info('Configuring input and output blobs')
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    num_of_classes = max(net.outputs[out_blob].shape)
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
    # ---------------------------Step 4. Prepare input---------------------------------------------------------------------
        # Generate a label list
    if args.labels:
        with open(args.labels, 'r') as f:
            labels = [line.split(',')[0].strip() for line in f]

    RESOLUTION=512
        # preprocessing function
    preprocessing = transforms.Compose([
        transforms.CenterCrop((512,512)),
        transforms.Resize((RESOLUTION,RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

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
    df=pd.read_csv('testing.csv')

    input_data = []
    _, _, h, w = net.input_info[input_blob].input_data.shape

    files = os.listdir(IMAGE_DIR)
    rows = len(files)
    plt.figure(figsize=(15, 5*rows))
    num_of_input=0
    inputpath=[]
    input_tensor=[]
    fidxlist=[]
    for file in files:
        num_of_input+=1
        ipath=IMAGE_DIR + file
        inputpath.append(ipath)
        image = pil_loader(ipath)
        # image = data_transforms["None"](image)
        # image = data_transforms["totensor"](image)
        image = preprocessing(image.copy()).unsqueeze(0)
        input_data.append(image)

    # ---------------------------Step 5. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    print("Loading start")
    exec_net = ie.load_network(network=net, device_name=args.device, num_requests=num_of_input)
    print("Loading fin")

    # ---------------------------Step 6 Do inference----------------------------------------------------------------------
    log.info('Starting inference in asynchronous mode')
    starttime= tt.perf_counter()
    for i in range(num_of_input):
        exec_net.requests[i].async_infer({input_blob: input_data[i]})
    # ---------------------------Step 7. Process output--------------------------------------------------------------------
    # Create a list to control a order of output
    output_queue = list(range(num_of_input))

    while True:
        for i in output_queue:
            # Immediately returns a inference status without blocking or interrupting
            infer_status = exec_net.requests[i].wait(0)
            if infer_status == StatusCode.RESULT_NOT_READY:
                continue
            log.info(f'Infer request {i} returned {infer_status}')
            if infer_status != StatusCode.OK:
                return -2
            # Read infer request results from buffer
            res = exec_net.requests[i].output_blobs[out_blob].buffer
            # Change a shape of a numpy.ndarray with results to get another one with one dimension
            probs = res.reshape(num_of_classes)
            # Get an array of args.number_top class IDs in descending order of probability
            top_n_idexes = np.argsort(probs)[-args.number_top :][::-1]
            header = 'classid probability'
            header = header + ' label' if args.labels else header


            _min=0
            _max=0
            
            for class_id in top_n_idexes:
                if probs[class_id] > _max:
                    _max=probs[class_id]
                if probs[class_id] < _min:
                    _min=probs[class_id]
            k=_max-_min
            _sum=0
            for class_id in top_n_idexes:
                _sum+=((probs[class_id]-_min)/k)+0.01

            log.info(f'Image path: {inputpath[i]}')
            log.info(f'Real Anser Label: {df[df["ImagePath"]==inputpath[i][-9:]]["label"].tolist()[0]}')
            log.info(f'model path: {args.model}')

            log.info(f'Top {args.number_top} results: ')
            log.info(header)
            log.info('-' * len(header))
            for class_id in top_n_idexes:
                probability_indent = ' ' * (len('classid') - len(str(class_id)) + 1)
                label_indent = ' ' * (len('probability') - 8) if args.labels else ''
                label = labels[class_id] if args.labels else ''
                log.info(f'{class_id+1}{probability_indent}{(((probs[class_id]-_min)/k)+0.01)/_sum:.7f}{label_indent}{label}')
            log.info('')
            output_queue.remove(i)
        if len(output_queue) == 0:
            break
    endtime = tt.perf_counter()
    inference_time=endtime-starttime
    log.info(f'inference time : {inference_time:.4f} s')
    log.info(f'Computing Grad-Cam...')

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
    return 0

if __name__ == '__main__':
    sys.exit(main())

