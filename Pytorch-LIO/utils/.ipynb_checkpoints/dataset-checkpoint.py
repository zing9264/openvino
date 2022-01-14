from __future__ import division
import os
import pickle
import numpy as np
import PIL.Image as Image
from PIL import ImageStat

import torch
import torch.utils.data as data

class dataset(data.Dataset):
    def __init__(self, img_dir, anno_pd, num_positive, preprocess=None,  totensor=None, train=False):
        self.img_dir = img_dir
        self.paths = anno_pd['ImagePath'].tolist()
        self.labels = anno_pd['index'].tolist()
        self.preprocess = preprocess
        self.anno_pd = anno_pd
        self.totensor = totensor
        self.train = train
        self.num_positive = num_positive

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.paths[item])
        img = self.pil_loader(img_path)
        img = self.preprocess(img)
        img = self.totensor(img)
        label = self.labels[item] - 1

        if self.train:
            positive_images = self.fetch_positive(self.num_positive, label, self.paths[item])
            return img, positive_images, label

        return img, label

    def fetch_positive(self, num, label, path):
        positive_imgs_info = self.anno_pd[(self.anno_pd['index'] == (label + 1)) & (self.anno_pd['ImagePath'] != path)]
        positive_imgs_info = positive_imgs_info.sample(min(num, len(positive_imgs_info))).to_dict('records')
        positive_imgs_path = [os.path.join(self.img_dir, e['ImagePath']) for e in positive_imgs_info]
        positive_imgs = [self.pil_loader(img) for img in positive_imgs_path]
        positive_imgs = [self.preprocess(img) for img in positive_imgs]
        positive_imgs = [self.totensor(img) for img in positive_imgs]
        return positive_imgs

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

def collate_fn_train(batch):
    imgs = []
    positives = []
    labels = []
    for sample in batch:
        imgs.append(sample[0])
        positives.extend(sample[1])
        labels.append(sample[2])
    return torch.stack(imgs, 0), torch.stack(positives, 0), labels

def collate_fn_test(batch):
    imgs = []
    labels = []
    for sample in batch:
        imgs.append(sample[0])
        labels.append(sample[1])
    return torch.stack(imgs, 0), labels
