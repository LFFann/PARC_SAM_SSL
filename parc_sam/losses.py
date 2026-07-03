from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.clamp(0, num_classes - 1).long(), num_classes).permute(0, 3, 1, 2).float()
    if mask is not None:
        mask = mask.float().unsqueeze(1)
        prob = prob * mask
        one_hot = one_hot * mask
    dims = (0, 2, 3)
    inter = (prob * one_hot).sum(dims)
    denom = (prob.square() + one_hot.square()).sum(dims)
    return (1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()


def supervised_segmentation_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    ce = F.cross_entropy(logits, target.long(), ignore_index=ignore_index)
    valid = target != ignore_index
    dice = dice_loss_from_logits(logits, target.clamp(0, num_classes - 1), num_classes, valid)
    return ce + dice


def weighted_pseudo_loss(logits: torch.Tensor, pseudo: torch.Tensor, weight: torch.Tensor, num_classes: int) -> torch.Tensor:
    ce_map = F.cross_entropy(logits, pseudo.long(), reduction="none")
    weight = weight.float().clamp_min(0.0)
    ce = (ce_map * weight).sum() / weight.sum().clamp_min(1.0)
    dice = dice_loss_from_logits(logits, pseudo, num_classes, weight > 0)
    return ce + dice


def proposal_set_loss(logits: torch.Tensor, candidate_set: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)
    mass = (prob * candidate_set.float()).sum(dim=1).clamp_min(1e-6)
    loss = -torch.log(mass)
    weight = weight.float().clamp_min(0.0)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def negative_set_loss(logits: torch.Tensor, negative_set: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if negative_set.sum() == 0:
        return logits.new_tensor(0.0)
    prob = torch.softmax(logits, dim=1)
    penalty = (prob * negative_set.float()).sum(dim=1)
    weight = weight.float().clamp_min(0.0)
    return (penalty * weight).sum() / weight.sum().clamp_min(1.0)


def correlation_consistency_loss(features: torch.Tensor, target_prob: torch.Tensor, sample_pixels: int = 512) -> torch.Tensor:
    b, c, h, w = features.shape
    feat = F.normalize(features.flatten(2).transpose(1, 2), dim=-1)
    prob = target_prob.flatten(2).transpose(1, 2)
    n = feat.shape[1]
    if n > sample_pixels:
        idx = torch.linspace(0, n - 1, steps=sample_pixels, device=features.device).long()
        feat = feat[:, idx]
        prob = prob[:, idx]
    feat_corr = torch.bmm(feat, feat.transpose(1, 2))
    prob_corr = torch.bmm(prob, prob.transpose(1, 2))
    prob_corr = prob_corr / prob_corr.detach().abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return F.smooth_l1_loss(feat_corr, prob_corr.detach())


def masked_cosine_alignment(student_features: torch.Tensor, sam_embeddings: torch.Tensor | None, masks: torch.Tensor) -> torch.Tensor:
    if sam_embeddings is None:
        return student_features.new_tensor(0.0)
    if masks.ndim != 4:
        raise ValueError("masks must be [B,C,H,W]")
    b, classes, h, w = masks.shape
    student = F.normalize(student_features, dim=1)
    sam = F.normalize(F.interpolate(sam_embeddings, size=(h, w), mode="bilinear", align_corners=False), dim=1)
    losses = []
    for class_idx in range(classes):
        m = masks[:, class_idx : class_idx + 1].float()
        denom = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        s_vec = (student * m).sum(dim=(2, 3), keepdim=True) / denom
        t_vec = (sam * m).sum(dim=(2, 3), keepdim=True) / denom
        losses.append(1.0 - (s_vec * t_vec).sum(dim=1).mean())
    return torch.stack(losses).mean()

