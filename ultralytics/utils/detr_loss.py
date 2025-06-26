# detr_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes):
    # Convert [cx, cy, w, h] -> [x_min, y_min, x_max, y_max]
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    # Compute IoU for box1 & box2
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def hungarian_match(outputs, targets):
    # outputs: dict with 'pred_logits' and 'pred_boxes'
    # targets: list of dicts with 'labels' and 'boxes'
    indices = []
    for b in range(len(targets)):
        out_prob = outputs['pred_logits'][b].softmax(-1)  # [num_queries, num_classes+1]
        out_bbox = outputs['pred_boxes'][b]               # [num_queries, 4]

        tgt_ids = targets[b]['labels']                    # [num_targets]
        tgt_bbox = targets[b]['boxes']                    # [num_targets, 4]

        cost_class = -out_prob[:, tgt_ids]                # [num_queries, num_targets]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = 1.0 * cost_class + 5.0 * cost_bbox + 2.0 * cost_giou
        C = C.cpu()

        indices.append(linear_sum_assignment(C))
    return indices


class DETRLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.class_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, outputs, targets):
        # outputs: dict with 'pred_logits', 'pred_boxes'
        # targets: list of dicts with 'labels', 'boxes'
        indices = hungarian_match(outputs, targets)
        bs, num_queries = outputs['pred_logits'].shape[:2]

        loss_cls = 0
        loss_bbox = 0
        loss_giou = 0
        for b, (src_idx, tgt_idx) in enumerate(indices):
            src_logits = outputs['pred_logits'][b][src_idx]         # [num_matched, C+1]
            tgt_labels = targets[b]['labels'][tgt_idx]              # [num_matched]
            loss_cls += self.class_loss(src_logits, tgt_labels)

            src_boxes = outputs['pred_boxes'][b][src_idx]
            tgt_boxes = targets[b]['boxes'][tgt_idx]
            loss_bbox += self.l1_loss(src_boxes, tgt_boxes)
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
            loss_giou += (1 - giou).mean()

        return {
            'loss_cls': loss_cls / bs,
            'loss_bbox': loss_bbox / bs,
            'loss_giou': loss_giou / bs,
        }
