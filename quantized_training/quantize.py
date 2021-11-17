import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def _mean(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.mean()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).mean(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).mean(dim=0).view(*output_size)
    else:
        return _mean(p.transpose(0, dim), 0).transpose(0, dim)

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

class UniformQuantize(InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None,
                stochastic=False, inplace=False, enforce_true_zero=False, num_chunks=None, out_half=False):

        num_chunks = input.shape[0] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)
        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)  # C
        if max_value is None:
            max_value  = y.max(-1)[0].mean(-1)  # C
        ctx.inplace    = inplace
        ctx.num_bits   = num_bits
        ctx.min_value  = min_value
        ctx.max_value  = max_value
        ctx.stochastic = stochastic
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        qmin = 0.
        qmax = 2.**num_bits - 1.
        scale = (max_value - min_value) / (qmax - qmin)
        scale = max(scale, 1e-8)
        if enforce_true_zero:
            initial_zero_point = qmin - min_value / scale
            zero_point = 0.
            # make zero exactly represented
            if initial_zero_point < qmin:
                zero_point = qmin
            elif initial_zero_point > qmax:
                zero_point = qmax
            else:
                zero_point = initial_zero_point
            zero_point = int(zero_point)
            output.div_(scale).add_(zero_point)
        else:
            output.add_(-min_value).div_(scale).add_(qmin)

        if ctx.stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
        output.clamp_(qmin, qmax).round_()  # quantize

        if enforce_true_zero:
            output.add_(-zero_point).mul_(scale)  # dequantize
        else:
            output.add_(-qmin).mul_(scale).add_(min_value)  # dequantize
        if out_half and num_bits <= 16:
            output = output.half()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None


class UniformQuantizeGrad(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, stochastic=True, inplace=False):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.min_value is None:
            min_value = float(grad_output.min())
            # min_value = float(grad_output.view(
            # grad_output.size(0), -1).min(-1)[0].mean())
        else:
            min_value = ctx.min_value
        if ctx.max_value is None:
            max_value = float(grad_output.max())
            # max_value = float(grad_output.view(
            # grad_output.size(0), -1).max(-1)[0].mean())
        else:
            max_value = ctx.max_value
        grad_input = UniformQuantize().apply(grad_output, ctx.num_bits,
                                             min_value, max_value, ctx.stochastic, ctx.inplace)
        return grad_input, None, None, None, None, None


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias,
                    stride, padding, dilation, groups)
    out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                    stride, padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = F.linear(input.detach(), weight, bias)
    out2 = F.linear(input, weight.detach(), bias.detach()
                    if bias is not None else None)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


def quantize(x, num_bits=8, min_value=None, max_value=None, num_chunks=None, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, num_chunks, stochastic, inplace)



def quantize_grad(x, num_bits=8, min_value=None, max_value=None, stochastic=True, inplace=False):
    return UniformQuantizeGrad().apply(x, num_bits, min_value, max_value, stochastic, inplace)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, momentum=0.1):
        super(QuantMeasure, self).__init__()
#        self.register_buffer('running_min', torch.zeros(1))
#        self.register_buffer('running_max', torch.zeros(1))
        self.running_min = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.running_max = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.momentum = momentum
        self.num_bits = num_bits

    def forward(self, input):
        if self.training:
            min_value = input.detach().view(
                input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(
                input.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(self.momentum).add_(
                min_value * (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(
                max_value * (1 - self.momentum))
        else:
            min_value = self.running_min
            max_value = self.running_max
        return quantize(input, self.num_bits, min_value=float(min_value), max_value=float(max_value), num_chunks=16)


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None, biprecision=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits)
        self.biprecision = biprecision

    def forward(self, input):
        qinput = self.quantize_input(input)
        qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                           min_value=float(self.weight.min()),
                           max_value=float(self.weight.max()))

        #self.weight.data = qweight.data
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_weight)
            #self.bias.data = qbias.data
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                              self.padding, self.dilation, self.groups)
            if self.num_bits_grad is not None:
                output = quantize_grad(output, num_bits=self.num_bits_grad)
        else:
            output = conv2d_biprec(qinput, qweight, qbias, self.stride,
                                   self.padding, self.dilation, self.groups, num_bits_grad=self.num_bits_grad)

        return output


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None, biprecision=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.quantize_input = QuantMeasure(self.num_bits)

    def forward(self, input):
        qinput = self.quantize_input(input)
        qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                           min_value=float(self.weight.min()),
                           max_value=float(self.weight.max()))
        #self.weight.data = qweight.data
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_weight)
            #self.bias.data = qbias.data
        else:
            qbias = None

        if not self.biprecision or self.num_bits_grad is None:
            output = F.linear(qinput, qweight, qbias)
            if self.num_bits_grad is not None:
                output = quantize_grad(output, num_bits=self.num_bits_grad)
        else:
            output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output


class Qsum(nn.Module):
    # this is normalized RangeBN
    def __init__(self, num_bits=8, num_bits_grad=8):
        super(Qsum, self).__init__()
        self.num_bits = num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input1 = QuantMeasure(self.num_bits)
        self.quantize_input2 = QuantMeasure(self.num_bits)
        
    def forward(self, input1, input2):
        qinput1 = self.quantize_input1(input1)
        qinput2 = self.quantize_input2(input2)
        out = qinput1 + qinput2
        out = quantize_grad(out, num_bits=self.num_bits_grad)
        return out
        
        
class RangeBN(nn.Module):
    # this is normalized RangeBN

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-5, num_bits=8, num_bits_grad=8):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.num_bits = num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits)
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        x = self.quantize_input(x)
        if x.dim() == 2:  # 1d
            x = x.unsqueeze(-1,).unsqueeze(-1)

        if self.training:
            B, C, H, W = x.shape
            y = x.transpose(0, 1).contiguous()  # C x B x H x W
            y = y.view(C, self.num_chunks, B * H * W // self.num_chunks)
            mean_max = y.max(-1)[0].mean(-1)  # C
            mean_min = y.min(-1)[0].mean(-1)  # C
            mean = y.view(C, -1).mean(-1)  # C
            scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
                                        0.5) / ((2 * math.log(y.size(-1))) ** 0.5)

            scale = 1 / ((mean_max - mean_min) * scale_fix + self.eps)

            self.running_mean.detach().mul_(self.momentum).add_(
                mean * (1 - self.momentum))

            self.running_var.detach().mul_(self.momentum).add_(
                scale * (1 - self.momentum))
        else:
            mean = self.running_mean
            scale = self.running_var
        scale = quantize(scale, num_bits=self.num_bits, min_value=float(
            scale.min()), max_value=float(scale.max()))
        out = (x - mean.view(1, mean.size(0), 1, 1)) * \
            scale.view(1, scale.size(0), 1, 1)

        if self.weight is not None:
            qweight = quantize(self.weight, num_bits=self.num_bits,
                               min_value=float(self.weight.min()),
                               max_value=float(self.weight.max()))
            out = out * qweight.view(1, qweight.size(0), 1, 1)

        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits)
            out = out + qbias.view(1, qbias.size(0), 1, 1)
        if self.num_bits_grad is not None:
            out = quantize_grad(out, num_bits=self.num_bits_grad)

        if out.size(3) == 1 and out.size(2) == 1:
            out = out.squeeze(-1).squeeze(-1)
        return out
    

class RangeEN(nn.Module):
    def __init__(self, num_features, chunks=16, groups=8, apply_act=True, eps=1e-5, num_bits=8, num_bits_grad=8):
        super(RangeEN, self).__init__()
        self.num_chunks = chunks
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.groups    = groups
        self.eps       = eps
        self.num_bits = num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits)
        param_shape    = (1, num_features, 1, 1)
        self.weight    = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias      = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        #self.hs        = nn.Hardsigmoid()
        if apply_act:
            self.v     = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        B, C, H, W = x.shape
        assert C % self.groups == 0
        
        #scale.view(1, scale.size(0), 1, 1)
        x = self.quantize_input(x)
        if self.apply_act:
            qv = quantize(self.v, num_bits=self.num_bits,
                               min_value=float(self.v.min()),
                               max_value=float(self.v.max()))
            hs = torch.clamp(((x * qv)/6) + 0.5, min=0, max=1)
            n = x * hs # n = x*sigmoid(x*v)
            y = x.reshape(B, self.groups, self.num_chunks, -1)
            mean_max  = y.max(-1)[0].mean(-1)  # C
            mean_min  = y.min(-1)[0].mean(-1)  # C
            scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) ** 0.5) /((2 * math.log(y.size(-1))) ** 0.5)
            scale     = 1 / ((mean_max - mean_min) * scale_fix + self.eps)
            scale = quantize(scale, num_bits=self.num_bits, min_value=float(scale.min()), max_value=float(scale.max()))
            x = n.reshape(B, self.groups, -1) * scale.view(scale.size(0), scale.size(1), 1)  # x = n/var(x) groupwise instance variance
            x = x.reshape(B, C, H, W)
            
        if self.weight is not None:
            qweight = quantize(self.weight, num_bits=self.num_bits,
                               min_value=float(self.weight.min()),
                               max_value=float(self.weight.max()))
            out = x * qweight

        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits)
            out = out + qbias
        if self.num_bits_grad is not None:
            out = quantize_grad(out, num_bits=self.num_bits_grad)

        return out