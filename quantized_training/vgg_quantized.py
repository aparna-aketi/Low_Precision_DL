import torch.nn as nn
import torchvision.transforms as transforms
import math
from .quantize import quantize, quantize_grad, QConv2d, QLinear, Qsum
from .quantize import RangeEN as RangeN
__all__ = ['vgg11_quantized']

NUM_BITS        = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD   = 8
BIPRECISION     = True

def qconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes, dataset):
        super(VGG, self).__init__()
        self.features = features
        self.pool = False
        if 'imagenet' in dataset:
            self.pool = True
            self.avgpool = nn.AvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            QLinear(512,512, num_bits=NUM_BITS,
                          num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
            nn.ReLU(True),
            QLinear(512, num_classes, num_bits=NUM_BITS,
                          num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        if self.pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = qconv3x3(in_channels, v)
            layers += [conv2d, RangeN(v, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11_quantized(num_classes=10, dataset='cifar10'):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), num_classes, dataset)