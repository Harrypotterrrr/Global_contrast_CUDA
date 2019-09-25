// #include <torch/extension.h>
#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include <vector>

// hyper parameter 
constexpr long BLOCK_SIZE = 32; 

namespace global_contrast_kernel{

template <typename scalar_t> 
__global__ void forward(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> output
) {
    long col = threadIdx.x + blockIdx.x * blockDim.x;
    long row = threadIdx.y + blockIdx.y * blockDim.y;
    const long B = input.size(0);
    const long C = input.size(1);
    const long W = input.size(2);
    const long H = input.size(3);
    scalar_t dis = 0.0f;
    for (auto i=0 ; i<B ; i++){
        for (auto j=0 ; j<C ; j++){
            for (auto _w=0 ; _w<W ; _w++){
                for (auto _h=0 ; _h<H ; _h++){
                    scalar_t diff = input[i][j][col][row] - input[i][j][_w][_h];
                    dis += diff * diff;
                }
            }
        }
        output[i][1][col][row] = dis;
        dis = 0.0f;
    }

    __syncthreads();
}

template <typename scalar_t> 
__global__ void backward(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> grad,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_input
) {

    long col = threadIdx.x + blockIdx.x * blockDim.x;
    long row = threadIdx.y + blockIdx.y * blockDim.y;
    const long B = input.size(0);
    const long C = input.size(1);
    const long W = input.size(2);
    const long H = input.size(3);

    scalar_t tmp = 0.0f;
    for (auto i=0 ; i<B ; i++){
        for (auto j=0 ; j<C ; j++){
            for (auto _w=0 ; _w<W ; _w++){
                for (auto _h=0 ; _h<H ; _h++){
                    tmp += input[i][j][_w][_h];
                }
            }
            d_input[i][j][col][row] = (input[i][j][col][row] - tmp / W / H ) * 4.0f * grad[i][1][col][row];
            tmp = 0.0f;
        }
    }

    __syncthreads();
}

}

torch::Tensor global_contrast_cuda_forward(
    const torch::Tensor& input
) {

    cudaSetDevice(input.get_device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const long B = input.size(0);
    const long C = input.size(1);
    const long W = input.size(2);
    const long H = input.size(3);

    // allocate output tensor
    auto output = torch::zeros({B, 1, W, H}, input.options());

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x, 
        (H + blockSize.y - 1) / blockSize.y);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "global_contrast_cuda_forward", ([&]{
        global_contrast_kernel::forward <scalar_t><<< gridSize, blockSize, 0, stream >>>(
            input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()
        );
    }));
    
    return output;
}


torch::Tensor global_contrast_cuda_backward(
    const torch::Tensor& grad,
    const torch::Tensor& input
) {

    cudaSetDevice(input.get_device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const long B = input.size(0);
    const long C = input.size(1);
    const long W = input.size(2);
    const long H = input.size(3);

    // allocate output tensor
    auto d_input = torch::zeros({B, C, W, H}, input.options());

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x, 
        (H + blockSize.y - 1) / blockSize.y);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "global_contrast_cuda_backward", ([&]{
        global_contrast_kernel::backward <scalar_t><<< gridSize, blockSize, 0, stream >>>(
            grad.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()
        );
    }));
    
    return d_input;
}