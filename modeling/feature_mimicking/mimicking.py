import torch
from torch.nn.functional import interpolate
from structures.bounding_box import BoxList
from torch import nn
from torch.nn import functional as F

class Mimicking_head(nn.Module):
    def __init__(self, cfg, backbone, roi_heads):
        super(Mimicking_head, self).__init__()
        self.weight_cos = cfg.FEATURE_MIMICKING.WEIGHT_COSINE
        self.weight_cls = cfg.FEATURE_MIMICKING.WEIGHT_CLS
        self.samples_per_img = cfg.FEATURE_MIMICKING.SAMPLES_PER_IMG
        self.resize = cfg.FEATURE_MIMICKING.RESIZE
        self.backbone = backbone
        self.conv = roi_heads.conv
        self.box = roi_heads.box
        cls_num = roi_heads.box.predictor.cls_score.weight.size(0)
        feature_dim = roi_heads.box.predictor.cls_score.weight.size(1)
        self.mimicking_cls_score = nn.Linear(feature_dim, cls_num).to(self.box.predictor.linear1.weight.device)
        nn.init.normal_(self.mimicking_cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.mimicking_cls_score.bias, 0)
    def forward(self, images, detections, box_mimicking_feature):
        resized_imgs, mimicking_labels, mimicking_ids = mimicking_gen(images, detections, self.samples_per_img, self.resize)
        box_mimicking_feature = box_mimicking_feature[mimicking_ids]
        _, x = self.backbone(resized_imgs)
        x = self.conv(x)
        mimicking_proposals = construct_mm_proposals(x)
        x = self.box.feature_extractor(x, mimicking_proposals)
        x = self.box.predictor.linear1(x.view(x.size(0), -1))
        x = self.box.predictor.linear2(x)
        cls_logits = self.mimicking_cls_score(x)
        loss_mimicking_cls = F.cross_entropy(cls_logits, mimicking_labels)
        loss_mimicking_cos_sim = F.cosine_embedding_loss(box_mimicking_feature, x, torch.ones([len(x)], device=x.device))
        loss_mimicking_cls *= self.weight_cls
        loss_mimicking_cos_sim *= self.weight_cos
        return dict(loss_mimicking_cls=loss_mimicking_cls, loss_mimicking_cos_sim=loss_mimicking_cos_sim)

def samples_2_inputs(img, mimicking_samples, all_ids, resize):
    bbox = mimicking_samples.bbox.to(torch.int32)
    xmin, ymin, xmax, ymax = bbox.split(1, dim=-1)
    resized_imgs = []
    labels = mimicking_samples.get_field('labels')
    new_labels = []
    new_ids = []
    for i in range(len(mimicking_samples)):
        x1,y1,x2,y2 = xmin[i], ymin[i], xmax[i], ymax[i]
        label = labels[i]
        id_ = all_ids[i]
        if (x2 - x1).abs().item() <= 16 or (y2 - y1).abs().item() <= 16:
            continue
        sub_img = img[:, :, y1:y2, x1:x2]
        resized_sub_img = interpolate(sub_img, [resize, resize], mode='bilinear')
        resized_imgs.append(resized_sub_img)
        new_labels.append(label.item())
        new_ids.append(id_)
    return torch.cat(resized_imgs, dim=0), torch.tensor(new_labels, dtype=torch.int64, device=xmin.device), new_ids

def mimicking_gen(images, detections, samples_per_img, resize):
    assert len(images.tensors) == len(detections), 'imgs and detections number mismatch !'
    resized_imgs = []
    labels = []
    ids = []
    for i in range(len(detections)):
        img = images.tensors[i].unsqueeze(0)
        mimicking_samples, all_ids = detections[i].random_sample(samples_per_img)
        resized_imgs_, labels_, ids_ = samples_2_inputs(img, mimicking_samples, all_ids, resize)
        resized_imgs.append(resized_imgs_)
        labels.append(labels_)
        ids.append(ids_)
    return torch.cat(resized_imgs, dim=0), torch.cat(labels), concat_ids(ids)

def construct_mm_proposals(imgs):
    bbox = torch.tensor([[0., 0., imgs.shape[2], imgs.shape[3]]], 
                        dtype=torch.float32, device=imgs.device)
    mimicking_proposals = [BoxList(bbox, [imgs.size(2), imgs.size(3)])] * len(imgs)
    return mimicking_proposals

def concat_ids(mimicking_ids):
    ids = mimicking_ids.copy()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ids[j] = [id_ + len(ids[i]) for id_ in ids[j]]
    new_ids = []
    for id_ in ids:
        new_ids += id_
    return new_ids