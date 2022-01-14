from torch import nn
from torchvision import models
from .scl_module import SCLModule

class MobileNetV3(nn.Module):
    def __init__(self, in_dim, num_classes, with_LIO):
        super(MobileNetV3, self).__init__()
        base = models.quantization.mobilenet_v3_large(pretrained=True)
        self.mobilenet = nn.Sequential(*list(base.children())[0])
        self.avgpool = nn.AvgPool2d(2, 2)
        self.size = 7
        self.feature_dim = 960
        self.structure_dim = 480
        self.scl_lrx = SCLModule(self.size, self.feature_dim, self.structure_dim, avg=True)
        self.classifier = nn.Linear(in_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.with_LIO = with_LIO

    def forward(self, x):
        x = self.mobilenet(x)
        if self.training and self.with_LIO:
            mask_feature, mask, coord_loss = self.scl_lrx(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        if self.training and self.with_LIO:
            return x, mask_feature, mask, coord_loss, out
            
        return out
