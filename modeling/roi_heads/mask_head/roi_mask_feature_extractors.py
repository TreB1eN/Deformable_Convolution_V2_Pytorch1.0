# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from modeling import registry
from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from modeling.poolers import Pooler
from ops.deform_psroi.src.modulated_deformable_roi_pooling import ModulatedDeformablePSRoIPooling
from modeling.utils import cat

@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNDeformableFeatureExtractor")
class MaskRCNNDeformableFeatureExtractor(nn.Module):
    """
    Heads of Deformable MASK RCNN for classification
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNDeformableFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.DEFORM_POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        channels_before_Pooling = cfg.MODEL.ROI_MASK_HEAD.CHANNELS_BEFORE_POOLING
        deform_pooling_fc_channels = cfg.MODEL.ROI_MASK_HEAD.DEFORM_POOLING_FC_CHANNELS
        
        self.pooler = ModulatedDeformablePSRoIPooling(channels_before_Pooling, 
                                                    deform_pooling_fc_channels, 
                                                    spatial_scale=scales, 
                                                    pooled_size=resolution, 
                                                    sampling_ratio=sampling_ratio) 
    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, proposals):
        rois = self.convert_to_roi_format(proposals)
        x = self.pooler(x, rois)
        return x

@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.DEFORM_POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.pooler = pooler

        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            # Caffe2 implementation uses MSRAFill, which in fact
            # corresponds to kaiming_normal_ in PyTorch
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x

def make_roi_mask_feature_extractor(cfg):
    if cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR == "ResNet50Conv5ROIFeatureExtractor":
        return ResNet50Conv5ROIFeatureExtractor
    else:
        func = registry.ROI_MASK_FEATURE_EXTRACTORS[cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR]
        return func(cfg)