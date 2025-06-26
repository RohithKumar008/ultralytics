# File: detr_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import generalized_box_iou
from scipy.optimize import linear_sum_assignment

class DETRLoss:
    def __init__(self, num_classes, matcher_cost_class=1.0, matcher_cost_bbox=5.0, matcher_cost_giou=2.0):
        self.num_classes = num_classes
        self.cost_class = matcher_cost_class
        self.cost_bbox = matcher_cost_bbox
        self.cost_giou = matcher_cost_giou

    def loss(self, outputs, targets):
        # outputs: logits [B, Q, C+1], boxes [B, Q, 4]
        # targets: list of dicts with 'labels' and 'boxes'
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Step 1: Match predictions to targets using Hungarian matcher
        indices = self.matcher(outputs, targets)

        # Step 2: Classification loss (cross entropy)
        idx = self._get_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        pred_logits = outputs["pred_logits"]
        pred_classes = pred_logits[idx]
        target_classes = torch.full(pred_classes.shape[:1], self.num_classes, dtype=torch.int64, device=pred_classes.device)
        target_classes[:len(target_classes_o)] = target_classes_o
        loss_ce = F.cross_entropy(pred_classes, target_classes)

        # Step 3: Box L1 and GIoU
        pred_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(pred_boxes, target_boxes, reduction='none').sum() / bs
        loss_giou = (1 - torch.diag(generalized_box_iou(pred_boxes, target_boxes))).sum() / bs

        return {
            'loss_ce': loss_ce,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }

    def matcher(self, outputs, targets):
        # outputs: logits [B, Q, C+1], boxes [B, Q, 4]
        out_prob = outputs["pred_logits"].softmax(-1)  # [B, Q, C+1]
        out_bbox = outputs["pred_boxes"]

        indices = []
        for b in range(out_prob.shape[0]):
            tgt_ids = targets[b]['labels']
            tgt_bbox = targets[b]['boxes']

            cost_class = -out_prob[b][:, tgt_ids]  # [Q, T]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(out_bbox[b], tgt_bbox)

            cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            i, j = linear_sum_assignment(cost.cpu())
            indices.append((torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)))
        return indices

    def _get_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return (batch_idx, src_idx)
