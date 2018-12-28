# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR

def make_optimizer_DeConv(cfg, model):
    params = []
    deform_params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if 'mask_conv' in key or 'offset_conv' in key:
            deform_params.append(value)
        else:
            params.append(value)
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    deconv_lr_factor = cfg.SOLVER.DECONV_LR_FACTOR
    params = [{"params": params, "lr": lr},
              {"params": deform_params, "lr": lr * deconv_lr_factor}]
    optimizer = torch.optim.SGD(params, weight_decay=weight_decay, momentum=cfg.SOLVER.MOMENTUM)   
    return optimizer

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
