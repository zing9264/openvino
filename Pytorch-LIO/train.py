import os
import sys
import math
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as torchdata
import torch.backends.cudnn as cudnn
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from multiprocessing.reduction import ForkingPickler

from utils.scheduler import WarmupMultiStepLR
from utils.rela import calc_rela
from utils.dataset import collate_fn_train, collate_fn_test, dataset
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

# python train.py --num_classes 8 --model_name mobilenet --epochs 20 --resolution 512 --num_positive 5 --batch_size 3 --with_LIO True --save_dir weights/intelComp/mobile-net_w_LIO/
# python train.py --num_classes 8 --model_name mobilenet --epochs 20 --resolution 512 --batch_size 3 --with_LIO False --save_dir weights/intelComp/mobile-net/
# python train.py --num_classes 8 --model_name resnet --epochs 20 --resolution 512 --num_positive 5 --batch_size 3 --with_LIO True --save_dir weights/intelComp/resnet_w_LIO/
# python train.py --num_classes 8 --model_name resnet --epochs 20 --resolution 512 --batch_size 3 --with_LIO False --save_dir weights/intelComp/resnet/
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=360, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_positive', default=3, type=int)
parser.add_argument('--num_worker', default=16, type=int)
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--num_classes', default=8, type=int)
parser.add_argument('--model_name', default='mobilenet', type=str)
parser.add_argument('--with_LIO', default=True, type=bool)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--valid_dir', default='kvasirV2/valid', type=str)
parser.add_argument('--train_dir', default='kvasirV2/train', type=str)
parser.add_argument('--save_dir', default='weights/intelComp/resnet/', type=str)
parser.add_argument('--resolution', default=512, type=int)
parser.add_argument('--cuda', default=0, type=int)
args = parser.parse_args()

# Params
STAGE = 3
TRAIN_DIR = args.train_dir
VALID_DIR = args.valid_dir
NUM_POSITIVE = args.num_positive
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
NUM_WORKER = args.num_worker
MODEL_NAME = args.model_name
SAVE_DIR = args.save_dir
MODEL_DIR = '{}{}_{}.pth'.format(args.save_dir, args.model_name, args.optimizer)
NUM_CLASSES = args.num_classes
WITH_LIO = args.with_LIO
LEARNING_RATE = args.learning_rate
RESOLUTION = args.resolution
CUDA = args.cuda

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

time = datetime.datetime.now()
# Read CSV
df_train = pd.read_csv("training.csv", dtype={'ImagePath': str, 'index': int})
df_valid = pd.read_csv("validation.csv", dtype={'ImagePath': str, 'index': int})

print('================ Dataset Info =====================')
print('train images:', df_train.shape)
print('valid images:', df_valid.shape)
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
    'train': dataset(img_dir=TRAIN_DIR, anno_pd=df_train, num_positive=NUM_POSITIVE, 
                     preprocess=data_transforms["preprocess"], totensor=data_transforms["totensor"], train=True),
    'valid': dataset(img_dir=VALID_DIR, anno_pd=df_valid, num_positive=NUM_POSITIVE, 
                     preprocess=data_transforms["None"], totensor=data_transforms["totensor"], train=False)   
}

# Dataloader
dataloader = {
    'train': torch.utils.data.DataLoader(data_set['train'], batch_size=BATCH_SIZE, shuffle=True, 
                                         num_workers=NUM_WORKER, collate_fn=collate_fn_train),
    'valid': torch.utils.data.DataLoader(data_set['valid'], batch_size=BATCH_SIZE, shuffle=False, 
                                         num_workers=NUM_WORKER, collate_fn=collate_fn_test)
}


# =============================================================================
# call out gpu is possible
torch.cuda.set_device(CUDA)
device = torch.device("cuda:{}".format(CUDA) if torch.cuda.is_available() else "cpu")
print('{} will be used in the training process !!!'.format(device))

if MODEL_NAME == 'mobilenet':
    model = mobilenet(in_dim=960, num_classes=NUM_CLASSES, size=int(7/448 * RESOLUTION), with_LIO=WITH_LIO)
else:
    model = resnet(stage=STAGE, in_dim=2048, num_classes=NUM_CLASSES, size=int(7/448 * RESOLUTION), with_LIO=WITH_LIO)
model.to(device) # model.cuda()

if device == "cuda":
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True


# =============================================================================
# optimizer, loss function, & learning rate scheduler
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
criterion = CrossEntropyLoss()
scheduler = WarmupMultiStepLR(optimizer, warmup_epoch = 2, milestones = [5, 10, 15, 20, 25])

# load weight
best_acc, start_epoch = 0, 0
try:
    assert os.path.isdir(SAVE_DIR), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(MODEL_DIR)
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
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



# =============================================================================
# Training Process
def mask_to_binary(x):
    N, H, W = x.shape
    x = x.view(N, H*W)
    thresholds = torch.mean(x, dim=1, keepdim=True)
    binary_x = (x > thresholds).float()
    return binary_x.view(N, H, W)

def calc_bce_loss(x, y, loss_f):
    x = F.sigmoid(x)
    return loss_f(x, y)
    
train_losses, valid_losses = [], []
train_accs, valid_accs = [], []
loss_weight = { 'mask': 0.1, 'coord': 0.1 }

def train_LIO(epoch):
    global train_losses, train_accs, loss_weight
    model.train(True)
    train_loss, correct, total = 0, 0, 0
    bar_len_total, batch_total = 30, len(dataloader['train'])
    print('current lr: %f' % optimizer.param_groups[0]['lr'])
    print('Epoch {}/{}'.format(epoch + 1, EPOCHS))

    for batch_idx, (imgs, positive_imgs, labels) in enumerate(dataloader['train']):
        labels = torch.LongTensor(np.array(labels))
        imgs, positive_imgs, labels = imgs.to(device), positive_imgs.to(device), labels.to(device)

        N = imgs.size(0)
        positive_labels = labels.view(N, 1).expand((N, NUM_POSITIVE)).contiguous().view(-1)
        optimizer.zero_grad()

        # *********** input main images ***************
        train_output = model(imgs)
        main_cls_feature, main_img_f = train_output[0], train_output[1]
        main_mask, coord_loss_1, main_probs = train_output[2], train_output[3], train_output[4]

        cls_loss_1 = criterion(main_probs, labels)
        coord_loss_1 = torch.mean(coord_loss_1 * mask_to_binary(main_mask))
        loss_1 = cls_loss_1 + coord_loss_1 * loss_weight['coord']

        # *********** input positive images ************
        train_output = model(positive_imgs)
        positive_cls_feature, positive_imgs_f = train_output[0], train_output[1]
        positive_masks, coord_loss_2, positive_probs = train_output[2], train_output[3], train_output[4]

        cls_loss_2 = criterion(positive_probs, positive_labels)
        coord_loss_2 = torch.mean(coord_loss_2 * mask_to_binary(positive_masks))
        loss_2 = cls_loss_2 + coord_loss_2 * loss_weight['coord']

        # *********** rela calculation **************
        n, c, h, w = positive_imgs_f.shape
        positive_imgs_f = positive_imgs_f.view(N, NUM_POSITIVE, c, h, w).transpose(0, 1).contiguous().view(N*NUM_POSITIVE, c, h, w)
        positive_masks = positive_masks.view(N, NUM_POSITIVE, h, w).transpose(0, 1).contiguous().view(N*NUM_POSITIVE, h, w)
        all_img_features = torch.cat((main_img_f, positive_imgs_f), dim=0)
        all_pred_masks = torch.cat((main_mask, positive_masks), dim=0)
        mask_reg_loss = calc_rela(all_img_features, all_pred_masks, NUM_POSITIVE).mean()

        # *********** total loss ****************
        loss = loss_1 + loss_2 + mask_reg_loss * loss_weight['mask']
        train_loss += loss.item()

        # *********** training acc **************
        _, predicted = main_probs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        train_acc = 1. * correct / total
        
        # *********** backpropagation ***************
        loss.backward()
        optimizer.step()

        # showing current progress
        bar_len = math.floor(bar_len_total * (batch_idx + 1) / batch_total)  - 1
        print('{}/{} '.format(batch_idx + 1, batch_total) + 
              '[' + '=' * bar_len + '>' + '.' * (bar_len_total - (bar_len + 1))  + '] ' + 
              '- train loss : {:.3f} '.format(loss.data.item()) +
              '- train acc : {:.3f} '.format(train_acc)
             , end='\r')  
        
    train_losses.append(train_loss / (batch_idx + 1))
    train_accs.append(train_acc)

    model.train(False)

def train(epoch):
    global train_losses, train_accs, loss_weight
    model.train(True)
    train_loss, correct, total = 0, 0, 0
    bar_len_total, batch_total = 30, len(dataloader['train'])
    print('current lr: %f' % optimizer.param_groups[0]['lr'])
    print('Epoch {}/{}'.format(epoch + 1, EPOCHS))

    for batch_idx, (imgs, labels) in enumerate(dataloader['train']):
        labels = torch.LongTensor(np.array(labels))
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        # *********** input main images ***************
        main_probs = model(imgs)
        loss = criterion(main_probs, labels)
        train_loss += loss.item()

        # *********** training acc **************
        _, predicted = main_probs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        train_acc = 1. * correct / total
        
        # *********** backpropagation ***************
        loss.backward()
        optimizer.step()

        # showing current progress
        bar_len = math.floor(bar_len_total * (batch_idx + 1) / batch_total)  - 1
        print('{}/{} '.format(batch_idx + 1, batch_total) + 
              '[' + '=' * bar_len + '>' + '.' * (bar_len_total - (bar_len + 1))  + '] ' + 
              '- train loss : {:.3f} '.format(loss.data.item()) +
              '- train acc : {:.3f} '.format(train_acc)
             , end='\r')  
        
    train_losses.append(train_loss / (batch_idx + 1))
    train_accs.append(train_acc)

    model.train(False)

def valid(epoch):
    global best_acc, train_losses, train_accs, valid_losses, valid_accs
    model.eval()
    valid_loss, correct, total = 0, 0, 0
    bar_len_total, batch_total = 30, len(dataloader['valid'])

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader['valid']):
            labels = torch.LongTensor(np.array(labels))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()
        
    # validation loss and accuracy
    valid_acc = 1. *correct / total

    # print result of this epoch
    train_loss, train_acc = train_losses[-1], train_accs[-1]
    print('{}/{} '.format(batch_total, batch_total) + 
          '[' + '=' * bar_len_total  + '] ' + 
          '- train loss : {:.3f} '.format(train_loss) + 
          '- train acc : {:.3f} '.format(train_acc) + 
          '- valid loss : {:.3f} '.format(valid_loss / (batch_idx + 1)) + 
          '- valid acc : {:.3f} '.format(valid_acc))

    # record the final training loss and acc in this epoch
    valid_losses.append(valid_loss / (batch_idx + 1))
    valid_accs.append(valid_acc)

    # save checkpoint if the result achieved is the best
    if valid_acc > best_acc:
        print('Best accuracy achieved, saving model to ' + MODEL_DIR)
        state = {
            'model': model.state_dict(),
            'acc': valid_acc,
            'epoch': epoch
        }
        torch.save(state, MODEL_DIR)
        best_acc = valid_acc

for epoch in range(start_epoch, EPOCHS):
    scheduler.step(epoch)
    if WITH_LIO:
        train_LIO(epoch)
    else:
        train(epoch)
    valid(epoch)

def Performance_Visualization(history):
    fig, axs = plt.subplots(1, 2,figsize=(16,4))
    
    # LOSS
    axs[0].plot(history['loss'], label='train')
    axs[0].plot(history['val_loss'], label='validation')
    axs[0].set(xlabel='epochs', ylabel='loss')
    axs[0].legend(loc='upper right')

    # ACCURACY
    axs[1].plot(history['acc'], label='train')
    axs[1].plot(history['val_acc'], label='validation')
    axs[1].set(xlabel='epochs', ylabel='metrics')
    axs[1].legend(loc='lower right')

    title = 'training_and_validation_graph_{}'.format(MODEL_NAME)
    if WITH_LIO:
        title += "_with_LIO"
    plt.savefig('{}.jpg'.format(title))
    plt.show()

history = dict()
history['loss'], history['acc'] = train_losses, train_accs
history['val_loss'], history['val_acc'] = valid_losses, valid_accs
Performance_Visualization(history)
    


