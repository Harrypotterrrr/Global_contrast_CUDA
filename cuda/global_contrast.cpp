#include <torch/extension.h>

// cuda 

torch::Tensor global_contrast_cuda_forward(
    const torch::Tensor& input
);


torch::Tensor global_contrast_cuda_backward(
    const torch::Tensor& grad,
    const torch::Tensor& input
);

torch::Tensor global_contrast_cuda_forward_split(
    const torch::Tensor& input
);


torch::Tensor global_contrast_cuda_backward_split(
    const torch::Tensor& grad,
    const torch::Tensor& input
);


// c++

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor global_contrast_forward(
    const torch::Tensor& input
) {
    CHECK_INPUT(input);
    
    return global_contrast_cuda_forward_split(input);
}

torch::Tensor global_contrast_backward(
    const torch::Tensor& grad,
    const torch::Tensor& input
) {
    CHECK_INPUT(grad);
    CHECK_INPUT(input);

    return global_contrast_cuda_backward_split(grad, input);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &global_contrast_forward, "global_contrast forward (CUDA)");
    m.def("backward", &global_contrast_backward, "global_contrast backward (CUDA)");
}
