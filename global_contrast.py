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
    def backward(ctx, grad, x):
        x = ctx.saved_variables # no *star here
        return global_contrast.backward(grad.contiguous(), x)


class GlobalContrast(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x):
        return GlobalContrastFunction.apply(x)

if __name__ == "__main__":

    B = 1 << 4
    C = 1 << 2
    W = 1 << 8
    H = 1 << 8

    globalContrast = GlobalContrast()
    print(global_contrast)

    x = torch.rand((B, C, W, H))
    y = globalContrast(x)
    
    print(y)
