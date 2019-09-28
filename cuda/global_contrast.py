import torch
from torch import nn
from torch.autograd import Function

import global_contrast


class GlobalContrastFunction(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return global_contrast.forward(x)

    @staticmethod
    def backward(ctx, grad):
        x = ctx.saved_variables # no *star here
        dx = global_contrast.backward(grad.contiguous(), *x)
        return dx


class GlobalContrast(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x):
        return GlobalContrastFunction.apply(x)
