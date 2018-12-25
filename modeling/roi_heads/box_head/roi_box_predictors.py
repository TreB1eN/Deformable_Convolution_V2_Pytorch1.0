# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn

class DeformFastRCNNPredictor(nn.Module):
    def __init__(self, cfg):
        super(DeformFastRCNNPredictor, self).__init__()

        channels_before_Pooling = cfg.MODEL.ROI_BOX_HEAD.CHANNELS_BEFORE_POOLING
        box_pooled_size = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        box_head_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.linear1 = nn.Linear(channels_before_Pooling * box_pooled_size * box_pooled_size, box_head_dim)
        self.linear2 = nn.Linear(box_head_dim, box_head_dim)
        self.cls_score = nn.Linear(box_head_dim, num_classes)
        self.bbox_pred = nn.Linear(box_head_dim, num_classes * 4)

        nn.init.normal_(self.linear1.weight, mean=0, std=0.01)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.normal_(self.linear2.weight, mean=0, std=0.01)
        nn.init.constant_(self.linear2.bias, 0)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return x, cls_logit, bbox_pred

class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


class FPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


_ROI_BOX_PREDICTOR = {
    "FastRCNNPredictor": FastRCNNPredictor,
    "FPNPredictor": FPNPredictor,
    "DeformFastRCNNPredictor": DeformFastRCNNPredictor
}


def make_roi_box_predictor(cfg):
    func = _ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg)
