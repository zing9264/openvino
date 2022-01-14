import numpy as np
from torch import nn
from torchvision import models
import torch.nn.functional as F

from .cbam_module import CBAM
from .scl_module import SCLModule

class resnet_swap_2loss_add(nn.Module):
    def __init__(self, stage, in_dim, num_classes, with_LIO, with_CBAM):
        super(resnet_swap_2loss_add,self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.stage1_img = nn.Sequential(*list(resnet50.children())[:5])
        self.stage2_img = nn.Sequential(*list(resnet50.children())[5:6])
        self.stage3_img = nn.Sequential(*list(resnet50.children())[6:7])
        self.stage4_img = nn.Sequential(*list(resnet50.children())[7])

        self.avgpool = nn.AvgPool2d(2, 2)
        self.stage = stage
        if stage == 3:
            self.size = 7
            self.feature_dim = 2048
            self.structure_dim = 1024
        elif stage == 2:
            self.size = 14
            self.feature_dim = 1024
            self.structure_dim = 512
        elif stage == 1:
            self.size = 28
            self.feature_dim = 512
            self.structure_dim = 256
        else:
            raise NotImplementedError("No such stage")
        self.scl_lrx = SCLModule(self.size, self.feature_dim, self.structure_dim, avg=True)
        self.classifier = nn.Linear(in_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.with_LIO = with_LIO
        self.with_CBAM = with_CBAM
        self.CBAM = CBAM(self.feature_dim)

    def forward(self, x):
        x2 = self.stage1_img(x)
        x3 = self.stage2_img(x2)
        x4 = self.stage3_img(x3)
        x5 = self.stage4_img(x4)  # Classifier Feature

        if self.with_CBAM == True:
            pass
            #x = self.CBAM(x)
        if self.training and self.with_LIO:
            if self.stage == 3:
                mask_feature, mask, coord_loss = self.scl_lrx(x5)
            elif self.stage == 2:
                mask_feature, mask, coord_loss = self.scl_lrx(x4)
            elif self.stage == 1:
                mask_feature, mask, coord_loss = self.scl_lrx(x3)
        
        x = self.avgpool(x5)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)


        if self.training and self.with_LIO:
            return x5, mask_feature, mask, coord_loss, out
            
        return out
