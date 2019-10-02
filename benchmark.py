import argparse
import time
import torch

from cuda.global_contrast import GlobalContrast

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('example', type=str, default='cuda', choices=['naive', 'cpp', 'cuda'])
parser.add_argument('-b', type=int, default=20)
parser.add_argument('-c', type=int, default=16)
parser.add_argument('-s', type=int, default=336)
parser.add_argument('-r', '--runs', type=int, default=1000)
options = parser.parse_args()



if __name__ == "__main__":

    B = options.b
    C = options.c
    W = options.s
    H = options.s

    globalContrast = GlobalContrast().cuda()

    # x = torch.rand((B, C, W, H), requires_grad=True).cuda()
    x = list(range(0, W * H)) * B * C
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True).cuda().reshape(B, C, W, H)
    x = torch.nn.Parameter(x) # for gradient check

    c = globalContrast(x)

    forward_time = 0
    backward_time = 0
    for _ in tqdm(range(options.runs)):

        if options.example == 'cuda':
            globalContrast.zero_grad()

        start = time.time()
        c = globalContrast(x)
        elapsed_fw = time.time() - start
        forward_time += elapsed_fw

        grad = torch.sum(c)
        start = time.time()
        grad.backward(retain_graph=True)
        elapsed_bk = time.time() - start
        backward_time += elapsed_bk

    forward_time *= 1000.0
    backward_time *= 1000.0

    print('forward: %.3fms  | backward: %.3fms' % (
        forward_time/options.runs, backward_time/options.runs
    ))