import math
from torch import nn
from torch.autograd import Function
import torch

import modulated_deform_psroi_cuda

class DeformablePSRoIPooling(nn.Module):
    def __init__(self, spatial_scale, roi_size, sampling_ratio, pooled_dim):
        super(DeformablePSRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        self.sampling_ratio = sampling_ratio
        self.pooled_dim = pooled_dim

    def forward(self, bottom_data, bottom_rois):
        return DeformablePSRoIPoolingFunction.apply(bottom_data, 
                                        bottom_rois, 
                                        self.spatial_scale, 
                                        self.roi_size, 
                                        self.sampling_ratio, 
                                        self.pooled_dim)

class DeformableRoIPoolingFunction(Function):
    @staticmethod
    def forward(ctx, bottom_data, bottom_trans, bottom_rois, 
                    no_trans, spatial_scale, 
                    pooled_size, sample_per_part, trans_std):
        ctx.no_trans = no_trans
        ctx.spatial_scale = spatial_scale
        ctx.pooled_size = pooled_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std
        ctx.feature_size = bottom_data.size()
        num_rois = bottom_rois.size(0)
        output_dim = bottom_data.size(1)
        group_size = 1
        part_size = pooled_size
        top_data = torch.zeros([num_rois, output_dim, pooled_size, pooled_size], dtype=torch.float32).to(bottom_data.device)
        top_count = torch.zeros([num_rois, output_dim, pooled_size, pooled_size], dtype=torch.float32).to(bottom_data.device)
        if bottom_data.is_cuda:
            modulated_deform_psroi_cuda.forward(bottom_data, bottom_rois, 
                                                bottom_trans, top_data, top_count, 
                                                no_trans, spatial_scale, output_dim,
                                                group_size, pooled_size, part_size, 
                                                sample_per_part, trans_std)
            ctx.save_for_backward(bottom_data, bottom_trans, bottom_rois, top_count)
        else:
            raise NotImplementedError

        return top_data

    @staticmethod
    def backward(ctx, top_diff):
        [bottom_data, bottom_trans, bottom_rois, top_count] = ctx.saved_tensors
        bottom_data_diff = None
        bottom_trans_diff = None
        if ctx.needs_input_grad[0]:
            batch_size, channels, height, width = ctx.feature_size
            group_size = 1
            part_size = ctx.pooled_size
            num_rois = bottom_rois.size(0)
            output_dim = bottom_data.size(1)
            bottom_data_diff = torch.zeros([batch_size, channels, height, width], dtype=torch.float32).to(top_diff.device)       
            bottom_trans_diff = torch.zeros([num_rois, 2, ctx.pooled_size, ctx.pooled_size], dtype=torch.float32).to(top_diff.device)
            modulated_deform_psroi_cuda.backward(bottom_data_diff, bottom_trans_diff,
                                                top_diff.contiguous(), 
                                                bottom_data, bottom_rois,
                                                bottom_trans, top_count, ctx.no_trans, 
                                                ctx.spatial_scale, output_dim,
                                                group_size, ctx.pooled_size, part_size,
                                                ctx.sample_per_part, ctx.trans_std)

        return bottom_data_diff, bottom_trans_diff, None, None, None, None, None, None