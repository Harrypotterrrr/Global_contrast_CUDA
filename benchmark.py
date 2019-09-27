import argparse
import time
import torch
import numpy as np
from cuda.global_contrast import GlobalContrast

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('example', type=str, default='cuda', choices=['naive', 'bmm', 'cuda'])
# parser.add_argument('-b', type=int, default=512)
# parser.add_argument('-c', type=int, default=128)
# parser.add_argument('-s', type=int, default=128)
parser.add_argument('-r', '--runs', type=int, default=1000)
options = parser.parse_args()



if __name__ == "__main__":

    globalContrast = GlobalContrast()

    B = 1 << 7
    C = 1 << 4
    W = 1 << 9
    H = 1 << 9
    B = 20
    C = 16
    W = 336
    H = 336
    x = torch.rand((B, C, W, H), requires_grad=True).cuda()

    c = globalContrast(x)

    forward_time = 0
    backward_time = 0
    for _ in tqdm(range(options.runs)):
        if options.example == 'cuda':
            globalContrast.zero_grad()

        start = time.time()
        c = globalContrast(x)
        elapsed = time.time() - start
        forward_time += elapsed

        grad = torch.sum(c)
        start = time.time()
        grad.backward()
        elapsed = time.time() - start
        backward_time += elapsed


    forward_time *= 1000.0
    backward_time *= 1000.0

    print('forward: %.3f  | backward: %.3f' % (
        forward_time/options.runs, backward_time/options.runs
    ))