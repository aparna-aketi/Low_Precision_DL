import torch.nn as nn
import torchvision.transforms as transforms
import math
from .quantize import quantize, quantize_grad, QConv2d, QLinear, Qsum
from .quantize import RangeEN as RangeN
__all__ = ['resnet_quantized']

NUM_BITS        = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD   = 8
BIPRECISION     = True


def qconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)

def init_modelq(model):
    for m in model.modules():
        if isinstance(m, QConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, RangeN):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    for m in model.modules():
        if isinstance(m, Bottleneckq):
            nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlockq):
            nn.init.constant_(m.bn2.weight, 0)

    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()


class BasicBlockq(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockq, self).__init__()
        self.conv1 = qconv3x3(inplanes, planes, stride)
        self.bn1 = RangeN(planes, num_bits=NUM_BITS,
                           num_bits_grad=NUM_BITS_GRAD)
        self.conv2 = qconv3x3(planes, planes)
        self.bn2 = RangeN(planes, num_bits=NUM_BITS,
                           num_bits_grad=NUM_BITS_GRAD)
        self.downsample = downsample
        self.acc = Qsum(num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD )
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        #out += residual
        out = self.acc(residual, out)
        return out


class Bottleneckq(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneckq, self).__init__()
        self.conv1 = QConv2d(inplanes, planes, kernel_size=1, bias=False,
                             num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        self.bn1 = RangeN(planes, num_bits=NUM_BITS,
                           num_bits_grad=NUM_BITS_GRAD)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, num_bits=NUM_BITS,
                             num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        self.bn2 = RangeN(planes, num_bits=NUM_BITS,
                           num_bits_grad=NUM_BITS_GRAD)
        self.conv3 = QConv2d(planes, planes * 4, kernel_size=1, bias=False,
                             num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        self.bn3 = RangeN(planes * 4, num_bits=NUM_BITS,
                           num_bits_grad=NUM_BITS_GRAD)
        self.acc = Qsum(num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.acc(residual, out)
        return out


class ResNetq(nn.Module):

    def __init__(self):
        super(ResNetq, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QConv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False,
                        num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
                RangeN(planes * block.expansion, num_bits=NUM_BITS,
                        num_bits_grad=NUM_BITS_GRAD)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class ResNet_imagenetq(ResNetq):

    def __init__(self, num_classes=1000,
                 block=Bottleneckq, layers=[3, 4, 23, 3]):
        super(ResNet_imagenetq, self).__init__()
        self.inplanes = 64
        self.conv1 = QConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                             bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        self.bn1 = RangeN(64, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = QLinear(512 * block.expansion, num_classes, num_bits=NUM_BITS,
                          num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)

        init_modelq(self)
        


class ResNet_cifar10q(ResNetq):

    def __init__(self, num_classes=10,
                 block=BasicBlockq, depth=18):
        super(ResNet_cifar10q, self).__init__()
        self.inplanes = 16
        n = int((depth - 2) / 6)
        self.conv1 = QConv2d(3, 16, kernel_size=3, stride=1, padding=1,
                             bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        self.bn1 = RangeN(16, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = QLinear(64, num_classes, num_bits=NUM_BITS,
                          num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        print('bit precisions:', NUM_BITS,NUM_BITS_WEIGHT,NUM_BITS_GRAD)
        init_modelq(self)



def resnet_quantized(num_classes=10, depth=20, dataset='cifar10'):
    if dataset == 'imagenet':
        if depth == 18:
            return ResNet_imagenetq(num_classes=num_classes,
                                   block=BasicBlockq, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenetq(num_classes=num_classes,
                                   block=BasicBlockq, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenetq(num_classes=num_classes,
                                   block=Bottleneckq, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenetq(num_classes=num_classes,
                                   block=Bottleneckq, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenetq(num_classes=num_classes,
                                   block=Bottleneckq, layers=[3, 8, 36, 3])

    elif dataset == 'cifar10':
        #print('imported quantized model')
        return ResNet_cifar10q(num_classes=num_classes, block=BasicBlockq, depth=depth)
    elif dataset == 'imagenette':
        num_classes = num_classes or 10
        return ResNet_imagenetq(num_classes=num_classes, block=BasicBlockq, layers=[2, 2, 2, 2])
        
   
