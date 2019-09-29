import argparse
import time
import torch

torch.backends.cudnn.benchmark = False

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

    globalContrast = GlobalContrast().cuda()

    # B = 1 << 4
    # C = 1 << 2
    # W = 1 << 2
    # H = 1 << 2
    B = 20
    C = 16
    W = 336
    H = 336

    x = list(range(0, W * H)) * B * C
    x = torch.tensor(x, dtype=torch.float64, requires_grad=True).cuda().reshape(B,C,W,H)
    # x = torch.rand((B, C, W, H), requires_grad=True).cuda()

    # c = globalContrast(x)

    forward_time = 0
    backward_time = 0
    for _ in tqdm(range(options.runs)):

        # if options.example == 'cuda':
        #     globalContrast.zero_grad()

        start = time.time()
        c = globalContrast(x)
        print(c.shape)
        print(c[0,0,:2,:2])
        elapsed_fw = time.time() - start
        forward_time += elapsed_fw
        # print("**************")
        x = torch.rand((B, C, W, H), requires_grad=True).cuda()
        # print("**************")


        # grad = torch.sum(c)

        # start = time.time()
        # grad.backward()
        # elapsed_bk = time.time() - start
        # backward_time += elapsed_bk
        # print('forward: %.3fs  | backward: %.3fs' % (
        #     forward_time, backward_time
        # ))


    forward_time *= 1000.0
    backward_time *= 1000.0

    print('forward: %.3fms  | backward: %.3fms' % (
        forward_time/options.runs, backward_time/options.runs
    ))