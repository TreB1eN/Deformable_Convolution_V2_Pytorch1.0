import torch
from torch import nn
from structures.image_list import to_image_list
from modeling.backbone import build_backbone
from modeling.rpn.rpn import build_rpn
from modeling.roi_heads.roi_heads import build_roi_heads
from modeling.feature_mimicking.mimicking import Mimicking_head

class DeformConvRCNN(nn.Module):
    """
    Main class for Deformable Convolution R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(DeformConvRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.mimicking_head = Mimicking_head(cfg, self.backbone, self.roi_heads)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        feature_c4, feature_c5 = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, [feature_c4], targets)
        _, box_mimicking_feature, detections, detector_losses = self.roi_heads(feature_c5, proposals, targets)

        if self.training:
            mimicking_losses = self.mimicking_head(images, detections, box_mimicking_feature)
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(mimicking_losses)
            return losses

        return detections