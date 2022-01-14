#!/bin/bash
cd /home/openvino/sharedata/Pytorch-LIO-lastest/Pytorch-LIO20220102/Pytorch-LIO

echo "start random image choice  image num 5"
python3  random_image_choice.py --image_num 5
echo "finish random image choice "
echo "start openvino demo"
python3 demo_gradcam_inference.py 
echo "finish openvino demo"
