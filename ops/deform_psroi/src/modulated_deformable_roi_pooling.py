import math
from torch import nn
from torch.autograd import Function
import torch

import modulated_deform_psroi_cuda

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ModulatedDeformablePSRoIPooling(nn.Module):
    def __init__(self, imfeat_dim, deform_fc_dim, spatial_scale, pooled_size, sampling_ratio, trans_std=0.1):
        super(ModulatedDeformablePSRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.sampling_ratio = sampling_ratio
        self.trans_std = trans_std
        self.relu = nn.ReLU()
        self.flatten = Flatten()
        self.linear_deform_1 = nn.Linear(imfeat_dim * pooled_size * pooled_size, deform_fc_dim, bias = True)
        self.linear_deform_2 = nn.Linear(deform_fc_dim, deform_fc_dim, bias = True)
        self.linear_deform_3 = nn.Linear(deform_fc_dim, pooled_size * pooled_size * 3, bias = True)
        self.roi_align = DeformablePSRoIPooling(True, 
                                                self.spatial_scale, 
                                                self.pooled_size, 
                                                self.sampling_ratio, 
                                                self.trans_std)
        self.deform_roi_pool = DeformablePSRoIPooling(False,
                                                        self.spatial_scale, 
                                                        self.pooled_size, 
                                                        self.sampling_ratio, 
                                                        self.trans_std)
        self.reset_parameters()       

    def reset_parameters(self):
        nn.init.normal_(self.linear_deform_1.weight, mean=0, std=0.01)
        nn.init.normal_(self.linear_deform_2.weight, mean=0, std=0.01)
        nn.init.normal_(self.linear_deform_3.weight, mean=0, std=0.001)
        nn.init.constant_(self.linear_deform_1.bias, 0)
        nn.init.constant_(self.linear_deform_2.bias, 0)
        nn.init.constant_(self.linear_deform_3.bias, 0)

    def forward(self, bottom_data, bottom_rois):
        roi_align = self.roi_align(bottom_data, bottom_rois)
        feat_deform = self.flatten(roi_align)
        feat_deform = self.relu(self.linear_deform_1(feat_deform))
        feat_deform = self.relu(self.linear_deform_2(feat_deform))
        feat_deform = self.relu(self.linear_deform_3(feat_deform))

        roi_offset = feat_deform[:,:2 * self.pooled_size * self.pooled_size].view(-1, 2, self.pooled_size, self.pooled_size)
        roi_mask = feat_deform[:,2 * self.pooled_size * self.pooled_size:].view(-1, 1, self.pooled_size, self.pooled_size)
        roi_mask_sigmoid = torch.sigmoid(roi_mask)

        deform_roi_pool = self.deform_roi_pool(bottom_data, bottom_rois, roi_offset)

        return roi_mask_sigmoid * deform_roi_pool

class DeformablePSRoIPooling(nn.Module):
    def __init__(self,  no_trans, spatial_scale, pooled_size, sampling_ratio, trans_std):
        super(DeformablePSRoIPooling, self).__init__()
        self.no_trans = no_trans
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.sampling_ratio = sampling_ratio
        self.trans_std = trans_std
    
    def forward(self, bottom_data, bottom_rois, bottom_trans = None):
        if bottom_trans is None:
            bottom_trans = torch.zeros([bottom_rois.shape[0], 2, self.pooled_size, self.pooled_size], dtype=torch.float).to(bottom_data.device)
        return DeformableRoIPoolingFunction.apply(bottom_data, bottom_trans, bottom_rois, 
                                                    self.no_trans, self.spatial_scale, 
                                                    self.pooled_size, self.sampling_ratio, 
                                                    self.trans_std)

class DeformableRoIPoolingFunction(Function):
    @staticmethod
    def forward(ctx, bottom_data, bottom_trans, bottom_rois, 
                    no_trans, spatial_scale, 
                    pooled_size, sampling_ratio, trans_std):
        ctx.no_trans = no_trans
        ctx.spatial_scale = spatial_scale
        ctx.pooled_size = pooled_size
        ctx.sampling_ratio = sampling_ratio
        ctx.trans_std = trans_std
        ctx.feature_size = bottom_data.size()
        num_rois = bottom_rois.size(0)
        output_dim = bottom_data.size(1)
        group_size = 1
        part_size = pooled_size
        top_data = torch.zeros([num_rois, output_dim, pooled_size, pooled_size], dtype=torch.float32).to(bottom_data.device)
        top_count = torch.zeros([num_rois, output_dim, pooled_size, pooled_size], dtype=torch.float32).to(bottom_data.device)
        if bottom_data.is_cuda:
            modulated_deform_psroi_cuda.forward(bottom_data.contiguous(), bottom_rois.contiguous(), 
                                                bottom_trans.contiguous(), top_data, top_count, 
                                                no_trans, spatial_scale, output_dim,
                                                group_size, pooled_size, part_size, 
                                                sampling_ratio, trans_std)
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
                                                bottom_data.contiguous(), bottom_rois.contiguous(),
                                                bottom_trans.contiguous(), top_count, ctx.no_trans, 
                                                ctx.spatial_scale, output_dim,
                                                group_size, ctx.pooled_size, part_size,
                                                ctx.sampling_ratio, ctx.trans_std)

        return bottom_data_diff, bottom_trans_diff, None, None, None, None, None, None