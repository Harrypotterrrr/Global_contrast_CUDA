// #include <torch/extension.h>
#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>


#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

// #include "TimingGPU.cuh"

// hyper parameter 
const long BLOCK_SIZE = 32;

namespace global_contrast_kernel{

template <typename scalar_t> 
__global__ void forward(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> feature,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> output
) {
    long col = threadIdx.x + blockIdx.x * blockDim.x;
    long row = threadIdx.y + blockIdx.y * blockDim.y;
    const long B = feature.size(0);
    const long C = feature.size(1);
    const long W = feature.size(2);
    const long H = feature.size(3);
 
    if(col >= W || row >= H)
        return;
    
    scalar_t dis = 0.0f;
    for (auto i=0 ; i<B ; i++){
        for (auto j=0 ; j<C ; j++){
            for (auto _w=0 ; _w<W ; _w++){
                for (auto _h=0 ; _h<H ; _h++){
                    scalar_t diff = feature[i][j][col][row] - feature[i][j][_w][_h];
                    dis += diff * diff;
                }
            }
        }
        output[i][0][col][row] = dis;
        dis = 0.0f;
    }

    __syncthreads();
}

template <typename scalar_t> 
__global__ void forward_split(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> feature,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sum,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> square,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> output
) {
    long col = threadIdx.x + blockIdx.x * blockDim.x;
    long row = threadIdx.y + blockIdx.y * blockDim.y;
    const long B = feature.size(0);
    const long C = feature.size(1);
    const long W = feature.size(2);
    const long H = feature.size(3);

    if(col >= W || row >= H)
        return;

    scalar_t dis = 0.0f, ft = 0.0f;
    for (auto i=0 ; i<B ; i++){
        for (auto j=0 ; j<C ; j++){
            ft = feature[i][j][col][row];
            dis += ft * ft + (square[i][j] - ft * sum[i][j] * 2.0f) / W / H; 
        }
        output[i][0][col][row] = dis / C;
        dis = 0.0f;
    }

    __syncthreads();
}

template <typename scalar_t>
__global__ void _calcSum_forward(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sum
) {

    long bs = threadIdx.x + blockIdx.x * blockDim.x;
    long ch = threadIdx.y + blockIdx.y * blockDim.y;
    const long B = feature.size(0);
    const long C = feature.size(1);
    const long W = feature.size(2);
    const long H = feature.size(3);

    if(bs >= B || ch >= C)
        return;

    scalar_t sm = 0.0f;
    for(auto i=0 ; i<W ; i++){
        for(auto j=0 ; j<H ; j++){
            sm += feature[bs][ch][i][j];
        }
    }
    sum[bs][ch] = sm;
    
    __syncthreads();

}

template <typename scalar_t>
__global__ void _calcSquare_forward(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> square
) {

    long bs = threadIdx.x + blockIdx.x * blockDim.x;
    long ch = threadIdx.y + blockIdx.y * blockDim.y;
    const long B = feature.size(0);
    const long C = feature.size(1);
    const long W = feature.size(2);
    const long H = feature.size(3);

    if(bs >= B || ch >= C)
        return;

    scalar_t sq = 0.0f, temp = 0.0f;
    for(auto i=0 ; i<W ; i++){
        for(auto j=0 ; j<H ; j++){
            temp = feature[bs][ch][i][j];
            sq += temp * temp;
        }
    }
    square[bs][ch] = sq;

    __syncthreads();

}

template <typename scalar_t>
__global__ void _calcSum_backward(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> feature,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_feature
) {

    long bs = threadIdx.x + blockIdx.x * blockDim.x;
    long ch = threadIdx.y + blockIdx.y * blockDim.y;
    const long B = feature.size(0);
    const long C = feature.size(1);
    const long W = feature.size(2);
    const long H = feature.size(3);

    if(bs >= B || ch >= C)
        return;

    scalar_t tmp = 0.0f;
    for(auto i=0 ; i<W ; i++){
        for(auto j=0 ; j<H ; j++){
            tmp += feature[bs][ch][i][j];
        }
    }
    tmp = (-1) * tmp / W / H;
    for(auto i=0 ; i<W ; i++){
        for(auto j=0 ; j<H ; j++){
            d_feature[bs][ch][i][j] = tmp;
        }
    }
    __syncthreads();

}

template <typename scalar_t>
__global__ void _calcGrad(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> grad,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> feature,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_feature
) {

    long col = threadIdx.x + blockIdx.x * blockDim.x;
    long row = threadIdx.y + blockIdx.y * blockDim.y;
    const long B = feature.size(0);
    const long C = feature.size(1);
    const long W = feature.size(2);
    const long H = feature.size(3);

    if(col >= W || row >= H)
        return;

    for (auto i=0 ; i<B ; i++){
        for (auto j=0 ; j<C ; j++){
            d_feature[i][j][col][row] = (feature[i][j][col][row] + d_feature[i][j][col][row]) * 4.0f * grad[i][0][col][row];
        }
    }
    __syncthreads();

}

template <typename scalar_t> 
__global__ void backward(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> grad,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> feature,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_feature
) {

    long col = threadIdx.x + blockIdx.x * blockDim.x;
    long row = threadIdx.y + blockIdx.y * blockDim.y;
    const long B = feature.size(0);
    const long C = feature.size(1);
    const long W = feature.size(2);
    const long H = feature.size(3);

    if(col >= W || row >= H)
        return;

    scalar_t tmp = 0.0f;
    for (auto i=0 ; i<B ; i++){
        for (auto j=0 ; j<C ; j++){
            for (auto _w=0 ; _w<W ; _w++){
                for (auto _h=0 ; _h<H ; _h++){
                    tmp += feature[i][j][_w][_h];
                }
            }
            d_feature[i][j][col][row] = (feature[i][j][col][row] - tmp / W / H ) * 4.0f * grad[i][0][col][row];
            tmp = 0.0f;
        }
    }
    
    __syncthreads();
}

}

torch::Tensor global_contrast_cuda_forward(
    const torch::Tensor& feature
) {

    cudaSetDevice(feature.get_device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const long B = feature.size(0);
    const long C = feature.size(1);
    const long W = feature.size(2);
    const long H = feature.size(3);

    // allocate output tensor
    auto output = torch::zeros({B, 1, W, H}, feature.options());

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x, 
        (H + blockSize.y - 1) / blockSize.y);

    AT_DISPATCH_FLOATING_TYPES(feature.type(), "global_contrast_cuda_forward", ([&]{
        global_contrast_kernel::forward <scalar_t><<< gridSize, blockSize, 0, stream>>>(
            feature.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()
        );
    }));
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << err <<std::endl;
        
    return output;
}

torch::Tensor global_contrast_cuda_forward_split(
    const torch::Tensor& feature
) {

    cudaSetDevice(feature.get_device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const long B = feature.size(0);
    const long C = feature.size(1);
    const long W = feature.size(2);
    const long H = feature.size(3);

    // allocate output tensor
    auto output = torch::zeros({B, 1, W, H}, feature.options());
    auto sum = torch::zeros({B, C}, feature.options());
    auto square = torch::zeros({B, C}, feature.options());

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x, 
        (H + blockSize.y - 1) / blockSize.y);

    AT_DISPATCH_FLOATING_TYPES(feature.type(), "_calcSum_forward", ([&]{
        global_contrast_kernel::_calcSum_forward <scalar_t><<< gridSize, blockSize, 0, stream>>>(
            feature.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            sum.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    cudaDeviceSynchronize();

    AT_DISPATCH_FLOATING_TYPES(feature.type(), "_calcSquare_forward", ([&]{
        global_contrast_kernel::_calcSquare_forward <scalar_t><<< gridSize, blockSize, 0, stream>>>(
            feature.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            square.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    cudaDeviceSynchronize();

    AT_DISPATCH_FLOATING_TYPES(feature.type(), "forward_split", ([&]{
        global_contrast_kernel::forward_split <scalar_t><<< gridSize, blockSize, 0, stream>>>(
            feature.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            sum.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            square.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()
        );
    }));

    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << err <<std::endl;
    
    return output;
}


torch::Tensor global_contrast_cuda_backward(
    const torch::Tensor& grad,
    const torch::Tensor& feature
) {

    cudaSetDevice(feature.get_device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const long B = feature.size(0);
    const long C = feature.size(1);
    const long W = feature.size(2);
    const long H = feature.size(3);

    // allocate output tensor
    auto d_feature = torch::zeros({B, C, W, H}, feature.options());

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x, 
        (H + blockSize.y - 1) / blockSize.y);

    AT_DISPATCH_FLOATING_TYPES(feature.type(), "global_contrast_cuda_backward", ([&]{
        global_contrast_kernel::backward <scalar_t><<< gridSize, blockSize>>>(
            grad.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            feature.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_feature.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()
        );
    }));
    
    return d_feature;
}


torch::Tensor global_contrast_cuda_backward_split(
    const torch::Tensor& grad,
    const torch::Tensor& feature
) {

    cudaSetDevice(feature.get_device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const long B = feature.size(0);
    const long C = feature.size(1);
    const long W = feature.size(2);
    const long H = feature.size(3);

    // allocate output tensor
    auto d_feature = torch::zeros({B, C, W, H}, feature.options());

    dim3 blockSize(4, 4);
    dim3 gridSize((B + blockSize.x - 1) / blockSize.x, 
        (C + blockSize.y - 1) / blockSize.y);

    AT_DISPATCH_FLOATING_TYPES(feature.type(), "global_contrast_cuda_backward", ([&]{
        global_contrast_kernel::_calcSum_backward <scalar_t> <<< gridSize, blockSize, 0, stream>>>(
            feature.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_feature.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()
        );
    }));


    blockSize = dim3 (BLOCK_SIZE, BLOCK_SIZE);
    gridSize = dim3((W + blockSize.x - 1) / blockSize.x, 
        (H + blockSize.y - 1) / blockSize.y);

    AT_DISPATCH_FLOATING_TYPES(feature.type(), "global_contrast_cuda_backward", ([&]{
        global_contrast_kernel::_calcGrad<scalar_t> <<< gridSize, blockSize, 0, stream>>>(
            grad.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            feature.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_feature.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()
        );
    }));

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << err <<std::endl;

    return d_feature;
}

/*
gettimeofday( &end, NULL );
long  timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
gettimeofday( &start, NULL );
printf("The first stage time is %ld us\n", timeuse);

...

gettimeofday( &end, NULL );
timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
gettimeofday( &start, NULL );
printf("The second stage time is %ld us\n", timeuse);
*/
