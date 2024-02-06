import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import bbox_iou, make_anchors, dist2bbox, bbox2dist, xywh2xyxy
from utils.assigner import TaskAlignedAssigner


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        loss_iou = 1.0 - iou

        # target_scores *= (pds*ciou * max(ciou) / max(pds*ciou_max))
        # loss_iou *= target_scores(weight)
        loss_iou *= bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum

        # dfl loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1),
                                     reduction="none").view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)


class YOLOV7Loss(nn.Module):
    def __init__(self, C=80, net_size=(), strides=(), reg_max=16, device=None):
        super(YOLOV7Loss, self).__init__()
        self.device = device
        self.nc = C
        self.nl = len(strides)
        self.no = C + reg_max * 4
        self.reg_max = reg_max
        self.use_dfl = True

        self.strides = torch.tensor(strides, dtype=torch.float32).to(device)
        self.net_size = torch.tensor([net_size, net_size], dtype=torch.float32).to(device)

        self.box_normalizer = 7.5
        self.cls_normalizer = 0.5
        self.dfl_normalizer = 1.5

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(self.reg_max).float().to(device)  # / 120.0

        self.cls_criterion = nn.BCEWithLogitsLoss(reduction='none').to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # b x 8400 x 64 -> b x 8400 x 4
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(self, preds, targets):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        # anchor_points: grid center point
        # stride_tensor: grid stride(8,16,32)
        anchor_points, stride_tensor = make_anchors(preds, self.strides, 0.5)

        pred_distri, pred_scores = torch.cat([xi.view(preds[0].shape[0], self.no, -1) for xi in preds], 2).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]

        # xywh2xyxy based on origin image size, b x max_num_targets_of_batch x 5
        targets = self.preprocess(targets, batch_size, scale_tensor=self.net_size[[1, 0, 1, 0]])
        # gt_labels: bxnx1, gt_bboxes: bxnx4
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # mask_gt: true idx targets of each batch
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes, pred_bboxes(xyxy) -+ anchor_points to pred_boxes (b, 8400, 4)
        # (pred_distri == offset) to anchor_center_points
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # tal assigner: ciou*scores(which center belong to gt_bboxes) + topk
        # target_bboxes: 2x8400x4
        # target_scores: 2x8400x20
        # fg_mask = mask_pos.sum(-2), 2x8400
        _, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            # pred_bboxes*stride_tensor=pred_bboxes_for_org_image_size
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            # anchor_points*stride_tensor=center_points_for_org_image_size
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = target_scores.sum()

        # cls loss
        # normalize target_scores:
        # target_scores *= (pds*ciou * max(ciou) / max(pds*ciou_max))
        loss[1] = self.cls_criterion(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)

        loss[0] *= self.box_normalizer  # box gain
        loss[1] *= self.cls_normalizer  # cls gain
        loss[2] *= self.dfl_normalizer  # dfl gain

        return loss.sum() * batch_size, loss.detach()


