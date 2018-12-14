#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__device__ float bilinear_interp(
  const float* data,
  const float x,
  const float y,
  const int width,
  const int height) {
  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  float dist_x = static_cast<float>(x - x1);
  float dist_y = static_cast<float>(y - y1);
  float value11 = data[y1*width + x1];
  float value12 = data[y2*width + x1];
  float value21 = data[y1*width + x2];
  float value22 = data[y2*width + x2];
  float value = (1 - dist_x) * (1 - dist_y) * value11 + (1 - dist_x) * dist_y * value12+ dist_x * (1 - dist_y) * value21 + dist_x * dist_y * value22;
  return value;
  }

  __global__ void DeformablePSROIPoolForwardKernel(
    const int nthreads,
    const float* bottom_data,
    const float spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const float* bottom_rois, const float* bottom_trans,
    const bool no_trans,
    const float trans_std,
    const int sample_per_part,
    const int output_dim,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class,
    float* top_data,
    float* top_count) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      const float* offset_bottom_rois = bottom_rois + n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      float roi_start_w = static_cast<float>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
      float roi_start_h = static_cast<float>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
      float roi_end_w = static_cast<float>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
      float roi_end_h = static_cast<float>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;

      // Force too small ROIs to be 1x1
      float roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      float roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      float bin_size_h = roi_height / static_cast<float>(pooled_height);
      float bin_size_w = roi_width / static_cast<float>(pooled_width);

      float sub_bin_size_h = bin_size_h / static_cast<float>(sample_per_part);
      float sub_bin_size_w = bin_size_w / static_cast<float>(sample_per_part);

      int part_h = floor(static_cast<float>(ph) / pooled_height*part_size);
      int part_w = floor(static_cast<float>(pw) / pooled_width*part_size);
      int class_id = ctop / channels_each_class;
      float trans_x = no_trans ? static_cast<float>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;
      float trans_y = no_trans ? static_cast<float>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2 + 1)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;

      float wstart = static_cast<float>(pw)* bin_size_w
        + roi_start_w;
      wstart += trans_x * roi_width;
      float hstart = static_cast<float>(ph) * bin_size_h
        + roi_start_h;
      hstart += trans_y * roi_height;

      float sum = 0;
      int count = 0;
      int gw = floor(static_cast<float>(pw) * group_size / pooled_width);
      int gh = floor(static_cast<float>(ph)* group_size / pooled_height);
      gw = min(max(gw, 0), group_size - 1);
      gh = min(max(gh, 0), group_size - 1);

      const float* offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
      for (int ih = 0; ih < sample_per_part; ih++) {
        for (int iw = 0; iw < sample_per_part; iw++) {
          float w = wstart + iw*sub_bin_size_w;
          float h = hstart + ih*sub_bin_size_h;
          // bilinear interpolation
          if (w<-0.5 || w>width - 0.5 || h<-0.5 || h>height - 0.5) {
            continue;
          }
          w = min(max(w, 0.), width - 1.);
          h = min(max(h, 0.), height - 1.);
          int c = (ctop*group_size + gh)*group_size + gw;
          float val = bilinear_interp(offset_bottom_data + c*height*width, w, h, width, height);
          sum += val;
          count++;
        }
      }
      top_data[index] = count == 0 ? static_cast<float>(0) : sum / count;
      top_count[index] = count;
    }
  }

  int DeformablePSROIPoolForwardLaucher(
          // bottom_data : [batch, channels, height, width]
      // bottom_rois : [num_rois, 5]
      // bottom_trans : [num_rois, 2, roi_size, roi_size]
      // top_data : [num_rois, output_dim, roi_size, roi_size]
      // top_count : [num_rois, output_dim, roi_size, roi_size]
      // The output is in order (n, ctop, ph, pw)
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
    const int nthreads = top_data.size(0) * top_data.size(1) * top_data.size(2) * top_data.size(3);
    const int channels = bottom_data.size(1);
    const int height = bottom_data.size(2);
    const int width = bottom_data.size(3);
    const int pooled_height = pooled_size;
    const int pooled_width = pooled_size;
    const int num_classes = no_trans ? 1 : bottom_trans.size(1) / 2;
    const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;
    
    DeformablePSROIPoolForwardKernel<<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(
                  nthreads, bottom_data.data<float>(), 
                  spatial_scale, channels, height, 
                  width, pooled_height, pooled_width,
                  bottom_rois.data<float>(), bottom_trans.data<float>(), 
                  no_trans, trans_std, sample_per_part, output_dim,
                  group_size, part_size, num_classes, channels_each_class, 
                  top_data.data<float>(), top_count_data.data<float>());
    
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    return 1;
  }

  __global__ void DeformablePSROIPoolBackwardAccKernel(
    const int nthreads,
    const float* top_diff,
    const float* top_count,
    const int num_rois,
    const float spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    float* bottom_data_diff, float* bottom_trans_diff,
    const float* bottom_data,
    const float* bottom_rois,
    const float* bottom_trans,
    const bool no_trans,
    const float trans_std,
    const int sample_per_part,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class) {

      // bottom_data : [batch, channels, heighy, width]
      // bottom_rois : [num_rois, 5]
      // bottom_trans : [num_rois, 2, roi_size, roi_size]
      // top_data : [num_rois, output_dim, roi_size, roi_size]
      // top_count : [num_rois, output_dim, roi_size, roi_size]
      // The output is in order (n, ctop, ph, pw)

    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      const float* offset_bottom_rois = bottom_rois + n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      float roi_start_w = static_cast<float>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
      float roi_start_h = static_cast<float>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
      float roi_end_w = static_cast<float>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
      float roi_end_h = static_cast<float>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;

      // Force too small ROIs to be 1x1
      float roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      float roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      float bin_size_h = roi_height / static_cast<float>(pooled_height);
      float bin_size_w = roi_width / static_cast<float>(pooled_width);

      float sub_bin_size_h = bin_size_h / static_cast<float>(sample_per_part);
      float sub_bin_size_w = bin_size_w / static_cast<float>(sample_per_part);

      int part_h = floor(static_cast<float>(ph) / pooled_height*part_size);
      int part_w = floor(static_cast<float>(pw) / pooled_width*part_size);
      int class_id = ctop / channels_each_class;
      float trans_x = no_trans ? static_cast<float>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;
      float trans_y = no_trans ? static_cast<float>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2 + 1)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;

      float wstart = static_cast<float>(pw)* bin_size_w + roi_start_w;
      wstart += trans_x * roi_width;
      float hstart = static_cast<float>(ph) * bin_size_h + roi_start_h;
      hstart += trans_y * roi_height;

      if (top_count[index] <= 0) {
        continue;
      }
      float diff_val = top_diff[index] / top_count[index];
      const float* offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width;
      float* offset_bottom_data_diff = bottom_data_diff + roi_batch_ind * channels * height * width;
      int gw = floor(static_cast<float>(pw)* group_size / pooled_width);
      int gh = floor(static_cast<float>(ph)* group_size / pooled_height);
      gw = min(max(gw, 0), group_size - 1);
      gh = min(max(gh, 0), group_size - 1);

      for (int ih = 0; ih < sample_per_part; ih++) {
        for (int iw = 0; iw < sample_per_part; iw++) {
          float w = wstart + iw * sub_bin_size_w;
          float h = hstart + ih * sub_bin_size_h;
          // bilinear interpolation
          if (w<-0.5 || w>width - 0.5 || h<-0.5 || h>height - 0.5) {
            continue;
          }
          w = min(max(w, 0.), width - 1.);
          h = min(max(h, 0.), height - 1.);
          int c = (ctop * group_size + gh) * group_size + gw;
          // backward on feature
          int x0 = floor(w);
          int x1 = ceil(w);
          int y0 = floor(h);
          int y1 = ceil(h);
          float dist_x = w - x0, dist_y = h - y0;
          float q00 = (1 - dist_x)*(1 - dist_y);
          float q01 = (1 - dist_x)*dist_y;
          float q10 = dist_x*(1 - dist_y);
          float q11 = dist_x*dist_y;
          int bottom_index_base = c * height *width;
          atomicAdd(offset_bottom_data_diff + bottom_index_base + y0*width + x0, q00*diff_val);
          atomicAdd(offset_bottom_data_diff + bottom_index_base + y1*width + x0, q01*diff_val);
          atomicAdd(offset_bottom_data_diff + bottom_index_base + y0*width + x1, q10*diff_val);
          atomicAdd(offset_bottom_data_diff + bottom_index_base + y1*width + x1, q11*diff_val);

          if (no_trans) {
            continue;
          }
          float U00 = offset_bottom_data[bottom_index_base + y0*width + x0];
          float U01 = offset_bottom_data[bottom_index_base + y1*width + x0];
          float U10 = offset_bottom_data[bottom_index_base + y0*width + x1];
          float U11 = offset_bottom_data[bottom_index_base + y1*width + x1];
          float diff_x = (U11*dist_y + U10*(1 - dist_y) - U01*dist_y - U00*(1 - dist_y)) * trans_std * diff_val;
          diff_x *= roi_width;
          float diff_y = (U11*dist_x + U01*(1 - dist_x) - U10*dist_x - U00*(1 - dist_x)) * trans_std * diff_val;
          diff_y *= roi_height;

          atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2)
                                           * part_size + part_h)
                                           * part_size + part_w, diff_x);
          atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1)
                                           * part_size + part_h)
                                           * part_size + part_w, diff_y);
        }
      }
    }
  }

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
    const float trans_std) {

    const int nthreads = top_diff.size(0) * top_diff.size(1) * top_diff.size(2) * top_diff.size(3);
    const int num_rois = bottom_rois.size(0);
    const int channels = bottom_data_diff.size(1);
    const int height = bottom_data_diff.size(2);
    const int width = bottom_data_diff.size(3);
    const int pooled_height = pooled_size;
    const int pooled_width = pooled_size;
    const int num_classes = no_trans ? 1 : bottom_trans_diff.size(1) / 2;
    const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

    DeformablePSROIPoolBackwardAccKernel<<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(
                nthreads, top_diff.data<float>(), top_count_data.data<float>(), 
                num_rois, spatial_scale, channels, height, width,
                pooled_height, pooled_width, output_dim, 
                bottom_data_diff.data<float>(), bottom_trans_diff.data<float>(),
                bottom_data.data<float>(), bottom_rois.data<float>(), bottom_trans.data<float>(), 
                no_trans, trans_std, sample_per_part,
                group_size, part_size, num_classes, channels_each_class);

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
      
    return 1;
  }