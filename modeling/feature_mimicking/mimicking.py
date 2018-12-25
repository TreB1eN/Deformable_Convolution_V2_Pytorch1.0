import torch
from torch.nn.functional import interpolate
from structures.bounding_box import BoxList
from torch import nn

class Mimicking_head(nn.Module):
    def __init__(self, backbone, roi_heads):
        super(Mimicking_head, self).__init__()
        self.backbone = backbone
        self.conv = roi_heads.conv
        self.box = roi_heads.box
        cls_num = roi_heads.box.predictor.cls_score.weight.size(0)
        feature_dim = roi_heads.box.predictor.cls_score.weight.size(1)
        self.mimicking_cls_score = nn.Linear(feature_dim, cls_num)
        nn.init.normal_(self.mimicking_cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.mimicking_cls_score.bias, 0)
    def forward(self, x):
        _, x = self.backbone(x)
        x = self.conv(x)
        mimicking_proposals = construct_mm_proposals(x)
        x = self.box.feature_extractor(x, mimicking_proposals)
        x = self.box.predictor.linear1(x.view(x.size(0), -1))
        x = self.box.predictor.linear2(x)
        cls_logits = self.mimicking_cls_score(x)
        return x, cls_logits

def samples_2_inputs(img, mimicking_samples, resize):
    bbox = mimicking_samples.bbox.to(torch.int32)
    xmin, ymin, xmax, ymax = bbox.split(1, dim=-1)
    resized_imgs = []
    for i in range(len(mimicking_samples)):
        x1,y1,x2,y2 = xmin[i], ymin[i], xmax[i], ymax[i]
        sub_img = img[:, :, y1:y2, x1:x2]
        resized_sub_img = interpolate(sub_img, [resize, resize], mode='bilinear')
        resized_imgs.append(resized_sub_img)
    return torch.cat(resized_imgs, dim=0), mimicking_samples.get_field('labels')

def mimicking_gen(images, detections, samples_per_img, resize):
    assert len(images.tensors) == len(detections), 'imgs and detections number mismatch !'
    resized_imgs = []
    labels = []
    ids = []
    for i in range(len(detections)):
        img = images.tensors[i].unsqueeze(0)
        mimicking_samples, ids_ = detections[i].random_sample(samples_per_img)
        resized_imgs_, labels_ = samples_2_inputs(img, mimicking_samples, resize)
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