import torch.nn as nn
import torchvision.transforms as transforms
import math
from .quantize import quantize, quantize_grad, QConv2d, QLinear, Qsum
from .quantize import RangeEN as RangeN
import torch.nn.functional as F
__all__ = ['mobilenetv2_quantized']
NUM_BITS        = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD   = 8
BIPRECISION     = True


def normalization_q(planes, chunks):
    if planes%(16*8)==0: num_chunks=16
    else: num_chunks=chunks
    return RangeN(planes, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD, chunks=num_chunks)
        #return RangeEN_full(planes)
def init_modelq(model):
    for m in model.modules():
        if isinstance(m, QConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, RangeN):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    model.linear.weight.data.normal_(0, 0.01)
    model.linear.bias.data.zero_()

class Block_mq(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, chunks=16):
        super(Block_mq, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = QConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        self.bn1 =  normalization_q(planes, chunks)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        self.bn2 = normalization_q(planes, chunks)
        self.conv3 = QConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        self.bn3 = normalization_q(out_planes, chunks)
       

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                QConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                normalization_q(out_planes, chunks),)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class mobilenetv2_quantized(nn.Module):
    # cfg = (expansion, out_planes, num_blocks, stride)

    def __init__(self, num_classes=10, dataset='imagenette'):
        super(mobilenetv2_quantized, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.dataset = dataset
        if 'cifar' in dataset:
            self.chunks = 16
            self.conv1 = QConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
            self.cfg = [(1,  16, 1, 1),
                        (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                        (6,  32, 3, 2),
                        (6,  64, 4, 2),
                        (6,  96, 3, 1),
                        (6, 160, 3, 2),
                        (6, 320, 1, 1)]
        else:
            self.chunks = 14
            self.conv1 = QConv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
            self.cfg = [(1,  16, 1, 1),
                        (6,  24, 2, 2),  
                        (6,  32, 3, 2),
                        (6,  64, 4, 2),
                        (6,  96, 3, 1),
                        (6, 160, 3, 2),
                        (6, 320, 1, 1)]
        self.bn1 = normalization_q(32, self.chunks)
        self.layers = self._make_layers(32, self.chunks)
        self.conv2 = QConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        self.bn2 = normalization_q(1280, self.chunks)
        self.linear = QLinear(1280, num_classes, num_bits=NUM_BITS,
                          num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        
        #init_modelq(self)

    def _make_layers(self, in_planes, chunks):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block_mq(in_planes, out_planes, expansion, stride, chunks))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layers(out)
        out = self.bn2(self.conv2(out))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        if 'cifar' in self.dataset:
            out = F.avg_pool2d(out, 4)
        else:
            out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


