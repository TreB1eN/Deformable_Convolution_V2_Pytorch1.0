#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

int ModulatedDeformConvForwardLaucher(
    at::Tensor data_im, 
    at::Tensor data_offset,
    at::Tensor data_mask,
    at::Tensor data_col,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w);



int ModulatedDeformConvBackwardLaucher(
    at::Tensor data_im, 
    at::Tensor data_offset,
    at::Tensor data_mask,
    at::Tensor data_col,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    at::Tensor grad_offset, 
    at::Tensor grad_mask,
    at::Tensor grad_im);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int modulated_deform_conv_forward_cuda(
    at::Tensor data_im, 
    at::Tensor data_offset,
    at::Tensor data_mask,
    at::Tensor data_col,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w) {

    CHECK_INPUT(data_im);
    CHECK_INPUT(data_offset);
    CHECK_INPUT(data_mask);
    CHECK_INPUT(data_col);

    ModulatedDeformConvForwardLaucher(data_im, data_offset, 
                                    data_mask, data_col, 
                                    kernel_h, kernel_w, 
                                    pad_h, pad_w, 
                                    stride_h, stride_w, 
                                    dilation_h, dilation_w); 
    return 1;
}

int modulated_deform_conv_backward_cuda(
    at::Tensor data_im, 
    at::Tensor data_offset,
    at::Tensor data_mask,
    at::Tensor data_col,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    at::Tensor grad_offset, 
    at::Tensor grad_mask,
    at::Tensor grad_im) {

    CHECK_INPUT(data_im);
    CHECK_INPUT(data_offset);
    CHECK_INPUT(data_mask);
    CHECK_INPUT(data_col);
    CHECK_INPUT(grad_offset);
    CHECK_INPUT(grad_mask);
    CHECK_INPUT(grad_im);

    ModulatedDeformConvBackwardLaucher(data_im, data_offset,
                                      data_mask,data_col,
                                      kernel_h, kernel_w,
                                      pad_h, pad_w,
                                      stride_h, stride_w,
                                      dilation_h, dilation_w,
                                      grad_offset, 
                                      grad_mask,
                                      grad_im);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &modulated_deform_conv_forward_cuda, "ModulatedDeformConv forward (CUDA)");
  m.def("backward", &modulated_deform_conv_backward_cuda, "ModulatedDeformConv backward (CUDA)");
}
