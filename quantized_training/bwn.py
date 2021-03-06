
"""
Bounded weight norm
Weight Normalization from https://arxiv.org/abs/1602.07868
taken and adapted from https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/weight_norm.py
"""
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
import torch.nn as nn


def gather_params(self, memo=None, param_func=lambda s: s._parameters.values()):
    if memo is None:
        memo = set()
    for p in param_func(self):
        if p is not None and p not in memo:
            memo.add(p)
            yield p
    for m in self.children():
        for p in gather_params(m, memo, param_func):
            yield p

nn.Module.gather_params = gather_params


def _norm(x, dim, p=2):
    """Computes the norm over all dimensions except dim"""
    if p == float('inf'):  # infinity norm
        func = lambda x, dim: x.abs().max(dim=dim)[0]
    else:
        func = lambda x, dim: torch.norm(x, dim=dim, p=p)
    if dim is None:
        return x.norm(p=p)
    elif dim == 0:
        output_size = (x.size(0),) + (1,) * (x.dim() - 1)
        return func(x.contiguous().view(x.size(0), -1), 1).view(*output_size)
    elif dim == x.dim() - 1:
        output_size = (1,) * (x.dim() - 1) + (x.size(-1),)
        return func(x.contiguous().view(-1, x.size(-1)), 0).view(*output_size)
    else:
        return _norm(x.transpose(0, dim), 0).transpose(0, dim)


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


class BoundedWeighNorm(object):

    def __init__(self, name, dim, p):
        self.name = name
        self.dim = dim
        self.p = p

    def compute_weight(self, module):
        v = getattr(module, self.name + '_v')
        pre_norm = getattr(module, self.name + '_prenorm')
        return v * (pre_norm / _norm(v, self.dim, p=self.p))

    @staticmethod
    def apply(module, name, dim, p):
        fn = BoundedWeighNorm(name, dim, p)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        prenorm = _norm(weight, dim, p=p).mean()
        module.register_buffer(name + '_prenorm', prenorm.detach())
        pre_norm = getattr(module, name + '_prenorm')
        print(pre_norm)
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        def gather_normed_params(self, memo=None, param_func=lambda s: fn.compute_weight(s)):
            return gather_params(self, memo, param_func)
        module.gather_params = gather_normed_params
        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_prenorm']
        del module._parameters[self.name + '_v']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def weight_norm(module, name='weight', dim=0, p=2):
    r"""Applies weight normalization to a parameter in the given module.
    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}
    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by `name` (e.g. "weight") with two parameters: one specifying the magnitude
    (e.g. "weight_g") and one specifying the direction (e.g. "weight_v").
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.
    By default, with `dim=0`, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    `dim=None`.
    See https://arxiv.org/abs/1602.07868
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm
    Returns:
        The original module with the weight norm hook
    Example::
        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        Linear (20 -> 40)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])
    """
    BoundedWeighNorm.apply(module, name, dim, p)
    return module


def remove_weight_norm(module, name='weight'):
    r"""Removes the weight normalization reparameterization from a module.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BoundedWeighNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))