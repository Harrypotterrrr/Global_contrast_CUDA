# segment_mm

This is an operator written in CUDA for PyTorch.

To compute the global contrast among each pixel. This work is inspired by [Xinyu Zhang](https://github.com/Sakura03) and referred to the paper from Mingming Cheng [Global Contrast Based Salient Region Detection](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6871397).

```python
def forward(
    input,
)
    """
    Params:
    ------
        input: float tensor, shape (B, C, W, H)
    
    Returns:
    ------
        output: float tensor, shape (B, 1, W, H)
    """
def backward(
    grad,
    input
):
    """
    Params:
    ------
        grad: float tensor, shape (B, 1, W, H)
        input: float tensor, shape (B, C, W, H)
    
    Returns:
    ------
        d_input: float tensor, shape (B, C, W, H)
    """
```

**For example**
```python
>>>import torch 
>>>from global_contrast import GlobalContrast
>>>x = torch.rand((16, 4, 128, 128)).cuda()
>>>model = GlobalContrast()
>>>y = model(x)
>>>y.size
tensor([16, 4, 128, 128], device='cuda:0')
```

# Benchmark

**TODO**

**Configurations**

| B | C | W | H |
| --- | --- | --- | ---- |
| 128 | 16 | 512 | 512 |

Run in a GTX Titan xp

**Results**

|  | forward(ms) | backward(ms) |
| --- | --- | --- |
| naive|3.451 |3.942 |
|cuda |**0.628** |**1.473** |
**TODO**

# Reference 

- [segment_mm](https://github.com/zhongyuchen/segment_mm)
- [Pytorch C++ and CUDA extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [Pytorch Tensor Basics](https://pytorch.org/cppdocs/notes/tensor_basics.html)
- [custom_op_benchmark](https://github.com/yzh119/custom_op_benchmark)