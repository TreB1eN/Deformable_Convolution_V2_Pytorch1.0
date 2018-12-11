#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

int DeformablePSROIPoolForwardLaucher(
    at::Tensor bottom_data,
    at::Tensor bottom_rois,
    at::Tensor bottom_trans,
    at::Tensor top_data,
    at::Tensor top_count_data,
    const bool no_trans,
    const float spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std);

int DeformablePSROIPoolBackwardAccLaucher(
    at::Tensor bottom_data_diff,
    at::Tensor bottom_trans_diff,
    at::Tensor top_diff,
    at::Tensor bottom_data,
    at::Tensor bottom_rois,
    at::Tensor bottom_trans,
    at::Tensor top_count_data,
    const bool no_trans,
    const float spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int modulated_deform_psroi_forward_cuda(
    at::Tensor bottom_data,
    at::Tensor bottom_rois,
    at::Tensor bottom_trans,
    at::Tensor top_data,
    at::Tensor top_count_data,
    const bool no_trans,
    const float spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std) {

    CHECK_INPUT(bottom_data);
    CHECK_INPUT(bottom_rois);
    CHECK_INPUT(bottom_trans);
    CHECK_INPUT(top_data);
    CHECK_INPUT(top_count_data);

    DeformablePSROIPoolForwardLaucher(bottom_data, bottom_rois, 
                                      bottom_trans,top_data, 
                                      top_count_data, no_trans, 
                                      spatial_scale, output_dim, 
                                      group_size,
                                      pooled_size, part_size, 
                                      sample_per_part, trans_std); 
    return 1;
}

int modulated_deform_psroi_backward_cuda(
    at::Tensor bottom_data_diff,
    at::Tensor bottom_trans_diff,
    at::Tensor top_diff,
    at::Tensor bottom_data,
    at::Tensor bottom_rois,
    at::Tensor bottom_trans,
    at::Tensor top_count_data,
    const bool no_trans,
    const float spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std) {

    CHECK_INPUT(bottom_data_diff);
    CHECK_INPUT(bottom_trans_diff);
    CHECK_INPUT(top_diff);
    CHECK_INPUT(bottom_data);
    CHECK_INPUT(bottom_rois);
    CHECK_INPUT(bottom_trans);
    CHECK_INPUT(top_count_data);    

    DeformablePSROIPoolBackwardAccLaucher(bottom_data_diff, bottom_trans_diff,
                                          top_diff, bottom_data,
                                          bottom_rois, bottom_trans,
                                          top_count_data, no_trans,
                                          spatial_scale, output_dim,
                                          group_size, pooled_size,
                                          part_size, sample_per_part, 
                                          trans_std);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &modulated_deform_psroi_forward_cuda, "ModulatedDeformPSROI forward (CUDA)");
  m.def("backward", &modulated_deform_psroi_backward_cuda, "ModulatedDeformPSROI backward (CUDA)");
}
