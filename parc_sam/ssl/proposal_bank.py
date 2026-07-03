from __future__ import annotations

import torch
import torch.nn.functional as F

from .conformal import ClassConditionalRiskController


class ProposalSetBuilder:
    """Fuse EMA teacher, SAM proposals, and prototype evidence into set-valued targets."""

    def __init__(self, num_classes: int, config: dict):
        self.num_classes = int(num_classes)
        self.config = config

    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(labels.long().clamp(0, self.num_classes - 1), self.num_classes).permute(0, 3, 1, 2).bool()

    def _risk_candidates(self, teacher_prob: torch.Tensor, risk: ClassConditionalRiskController) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(self.config.get("use_risk", True)):
            return risk.prediction_sets(teacher_prob)
        threshold = float(self.config.get("teacher_confidence", 0.6))
        candidate = teacher_prob >= threshold
        empty = candidate.sum(dim=1, keepdim=True) == 0
        if empty.any():
            candidate = candidate.clone()
            candidate.scatter_(1, teacher_prob.argmax(dim=1, keepdim=True), True)
        return candidate, empty.squeeze(1)

    def _cap_candidate_set(self, candidate: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        max_set = int(self.config.get("max_candidate_set_size", 2))
        if max_set <= 0 or max_set >= self.num_classes:
            return candidate
        top_idx = evidence.topk(max_set, dim=1).indices
        top_candidate = torch.zeros_like(candidate)
        top_candidate.scatter_(1, top_idx, True)
        capped = candidate & top_candidate
        empty = capped.sum(dim=1, keepdim=True) == 0
        return torch.where(empty, top_candidate, capped)

    def _foreground_rescue(self, candidate: torch.Tensor, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rescue = torch.zeros_like(candidate[:, 0], dtype=torch.bool)
        if self.num_classes <= 1 or not bool(self.config.get("foreground_guard", True)):
            return candidate, rescue
        min_ratio = float(self.config.get("min_foreground_participation", 0.0))
        if min_ratio <= 0:
            return candidate, rescue
        foreground_present = candidate[:, 1:].any(dim=1)
        current_ratio = float(foreground_present.float().mean().detach().cpu())
        if current_ratio >= min_ratio:
            return candidate, rescue
        fg_scores, fg_local = evidence[:, 1:].max(dim=1)
        flat_scores = fg_scores.flatten()
        k = max(1, int(round(min_ratio * flat_scores.numel())))
        adaptive_threshold = flat_scores.topk(min(k, flat_scores.numel())).values.min()
        fixed_threshold = evidence.new_tensor(float(self.config.get("foreground_rescue_confidence", 0.35)))
        threshold = torch.maximum(adaptive_threshold, fixed_threshold)
        rescue = (~foreground_present) & (fg_scores >= threshold)
        if rescue.any():
            rescued_class = fg_local + 1
            candidate = candidate.clone()
            candidate.scatter_(1, rescued_class.unsqueeze(1), rescue.unsqueeze(1))
        return candidate, rescue

    def _sam_evidence_weight(self, sam_prob: torch.Tensor, sam_out: dict) -> torch.Tensor:
        weight = torch.ones_like(sam_prob)
        weight[:, 0] = float(self.config.get("sam_background_weight", 0.35))
        fg_weight = float(self.config.get("sam_foreground_weight", 0.70))
        if self.num_classes <= 1:
            return weight
        sam_iou = sam_out.get("iou") if isinstance(sam_out, dict) else None
        if isinstance(sam_iou, torch.Tensor) and sam_iou.numel() > 0:
            sam_iou = sam_iou.to(sam_prob.device).detach().float()
            if sam_iou.ndim == 2 and sam_iou.shape[1] >= self.num_classes - 1:
                min_iou = float(self.config.get("sam_iou_min", 0.40))
                power = float(self.config.get("sam_iou_power", 1.0))
                denom = max(1e-6, 1.0 - min_iou)
                quality = ((sam_iou[:, : self.num_classes - 1] - min_iou) / denom).clamp(0.0, 1.0)
                quality = quality.pow(power).view(sam_prob.shape[0], self.num_classes - 1, 1, 1)
                weight[:, 1:] = fg_weight * quality
                return weight
        weight[:, 1:] = fg_weight
        return weight

    def _area_limit_for_class(self, prior: float) -> float:
        floor = float(self.config.get("max_foreground_area_floor", 0.02))
        multiplier = float(self.config.get("max_foreground_area_multiplier", 6.0))
        ceiling = float(self.config.get("max_foreground_area_ceiling", 0.30))
        return max(floor, min(ceiling, prior * multiplier))

    def _apply_area_guard(
        self,
        candidate: torch.Tensor,
        pseudo: torch.Tensor,
        evidence: torch.Tensor,
        risk: ClassConditionalRiskController,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        area_guard = torch.zeros_like(pseudo, dtype=torch.bool)
        area_stats: dict[str, float] = {}
        if self.num_classes <= 1 or not bool(self.config.get("class_area_guard", True)):
            return candidate, pseudo, evidence, area_guard, area_stats

        prior = risk.pixel_prior.to(evidence.device).float()
        _, _, h, w = evidence.shape
        pixels_per_image = h * w
        min_pixels = int(self.config.get("area_guard_min_pixels", 16))
        candidate = candidate.clone()
        for class_idx in range(1, self.num_classes):
            limit_ratio = self._area_limit_for_class(float(prior[class_idx].detach().cpu()))
            limit_pixels = max(min_pixels, int(round(limit_ratio * pixels_per_image)))
            cls_guard = torch.zeros_like(pseudo, dtype=torch.bool)
            for batch_idx in range(pseudo.shape[0]):
                cls_mask = pseudo[batch_idx] == class_idx
                cls_count = int(cls_mask.sum().detach().cpu())
                if cls_count <= limit_pixels:
                    continue
                scores = evidence[batch_idx, class_idx].masked_fill(~cls_mask, -1e6).flatten()
                keep_idx = scores.topk(min(limit_pixels, scores.numel())).indices
                keep = torch.zeros_like(scores, dtype=torch.bool)
                keep[keep_idx] = True
                keep = keep.view(h, w) & cls_mask
                excess = cls_mask & ~keep
                if excess.any():
                    cls_guard[batch_idx] = excess
                    candidate[batch_idx, 0][excess] = True
                    candidate[batch_idx, class_idx][excess] = True
            area_guard = area_guard | cls_guard
            area_stats[f"class_{class_idx}_area_limit"] = float(limit_ratio)
            area_stats[f"class_{class_idx}_area_guard_ratio"] = float(cls_guard.float().mean().detach().cpu())

        if area_guard.any():
            candidate_mass = candidate.float()
            ambiguous = candidate_mass / candidate_mass.sum(dim=1, keepdim=True).clamp_min(1.0)
            evidence = torch.where(area_guard.unsqueeze(1), ambiguous, evidence)
            pseudo = evidence.argmax(dim=1)
        area_stats["area_guard_ratio"] = float(area_guard.float().mean().detach().cpu())
        return candidate, pseudo, evidence, area_guard, area_stats

    def _stats(
        self,
        candidate: torch.Tensor,
        pseudo: torch.Tensor,
        weight: torch.Tensor,
        negative_set: torch.Tensor,
        singleton: torch.Tensor,
        conflict: torch.Tensor,
        low_reliability: torch.Tensor,
        rescue: torch.Tensor,
        area_guard: torch.Tensor,
        sam_valid: bool,
        risk: ClassConditionalRiskController,
        extra: dict[str, float] | None = None,
    ) -> dict[str, float | bool]:
        stats: dict[str, float | bool] = {
            "proposal_singleton_ratio": float(singleton.float().mean().detach().cpu()),
            "proposal_conflict_ratio": float(conflict.float().mean().detach().cpu()),
            "proposal_avg_set_size": float(candidate.float().sum(dim=1).mean().detach().cpu()),
            "proposal_weight_mean": float(weight.mean().detach().cpu()),
            "sam_used": bool(sam_valid),
            "risk_low_ratio": float(low_reliability.float().mean().detach().cpu()),
            "foreground_rescue_ratio": float(rescue.float().mean().detach().cpu()),
            "area_guard_ratio": float(area_guard.float().mean().detach().cpu()),
            "foreground_candidate_ratio": float(candidate[:, 1:].any(dim=1).float().mean().detach().cpu()) if self.num_classes > 1 else 0.0,
            "background_only_ratio": float((~candidate[:, 1:].any(dim=1)).float().mean().detach().cpu()) if self.num_classes > 1 else 0.0,
        }
        if extra:
            stats.update(extra)
        q = risk.q_per_class.detach().cpu()
        prior = risk.pixel_prior.detach().cpu()
        balance = risk.class_balance_weights().detach().cpu()
        for class_idx in range(self.num_classes):
            cls_candidate = candidate[:, class_idx]
            cls_pseudo = pseudo == class_idx
            cls_negative = negative_set[:, class_idx]
            stats[f"class_{class_idx}_candidate_ratio"] = float(cls_candidate.float().mean().detach().cpu())
            stats[f"class_{class_idx}_pseudo_ratio"] = float(cls_pseudo.float().mean().detach().cpu())
            stats[f"class_{class_idx}_singleton_ratio"] = float((cls_candidate & singleton).float().mean().detach().cpu())
            stats[f"class_{class_idx}_negative_ratio"] = float(cls_negative.float().mean().detach().cpu())
            stats[f"class_{class_idx}_weight_mean"] = float(weight[cls_pseudo].mean().detach().cpu()) if cls_pseudo.any() else 0.0
            stats[f"risk_q_class_{class_idx}"] = float(q[class_idx])
            stats[f"class_{class_idx}_prior"] = float(prior[class_idx])
            stats[f"class_{class_idx}_balance_weight"] = float(balance[class_idx])
        return stats

    def build(
        self,
        teacher_prob: torch.Tensor,
        risk: ClassConditionalRiskController,
        sam_out: dict | None = None,
        prototype_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        teacher_prob = teacher_prob.detach()
        device = teacher_prob.device
        candidate, low_reliability = self._risk_candidates(teacher_prob, risk)
        evidence = teacher_prob.clone()
        evidence_sources = torch.ones_like(evidence)
        extra_stats: dict[str, float] = {}

        sam_valid = bool(self.config.get("use_sam", True) and sam_out and sam_out.get("valid", False) and "prob" in sam_out)
        if sam_valid:
            sam_prob = sam_out["prob"].to(device).detach()
            sam_weight = self._sam_evidence_weight(sam_prob, sam_out)
            sam_conf = (sam_prob * sam_weight).max(dim=1, keepdim=True).values
            sam_candidate = (sam_prob >= float(self.config.get("min_sam_confidence", 0.5))) & (sam_weight > 0)
            empty = sam_candidate.sum(dim=1, keepdim=True) == 0
            if empty.any():
                sam_candidate = sam_candidate.clone()
                sam_candidate.scatter_(1, sam_prob.argmax(dim=1, keepdim=True), True)
            intersect = candidate & sam_candidate
            candidate = torch.where(intersect.sum(dim=1, keepdim=True) > 0, intersect, candidate | sam_candidate)
            evidence = evidence + sam_prob * sam_weight
            evidence_sources = evidence_sources + sam_weight
            sam_label = sam_prob.argmax(dim=1)
            extra_stats["sam_weight_mean"] = float(sam_weight.mean().detach().cpu())
        else:
            sam_prob = None
            sam_label = None

        if bool(self.config.get("use_prototype", True)) and prototype_logits is not None:
            proto_prob = torch.softmax(prototype_logits.detach(), dim=1)
            proto_weight = float(self.config.get("prototype_evidence_weight", 1.0))
            evidence = evidence + proto_weight * proto_prob
            evidence_sources = evidence_sources + proto_weight
            proto_candidate = proto_prob >= float(self.config.get("teacher_confidence", 0.6))
            candidate = candidate | proto_candidate

        evidence = evidence / evidence_sources
        pseudo = evidence.argmax(dim=1)
        target_mode = str(self.config.get("target_mode", "set_valued")).lower()
        if target_mode == "hard":
            candidate = self._one_hot(pseudo)
            rescue = torch.zeros_like(pseudo, dtype=torch.bool)
        elif target_mode == "conformal_single":
            candidate = candidate & self._one_hot(pseudo)
            empty = candidate.sum(dim=1, keepdim=True) == 0
            candidate = torch.where(empty, self._one_hot(pseudo), candidate)
            rescue = torch.zeros_like(pseudo, dtype=torch.bool)
        else:
            candidate = self._cap_candidate_set(candidate, evidence)
            candidate, rescue = self._foreground_rescue(candidate, evidence)

        candidate, pseudo, evidence, area_guard, area_stats = self._apply_area_guard(candidate, pseudo, evidence, risk)
        extra_stats.update(area_stats)

        teacher_conf, teacher_label = teacher_prob.max(dim=1)
        evidence_conf = evidence.max(dim=1).values
        agreement = teacher_label == pseudo
        if sam_label is not None:
            agreement = agreement & (sam_label == pseudo)
        conflict = (teacher_label != pseudo) & (teacher_conf >= float(self.config.get("teacher_confidence", 0.6)))
        base_weight = evidence_conf.clone()
        base_weight = base_weight + float(self.config.get("agreement_bonus", 0.2)) * agreement.float()
        base_weight = base_weight - float(self.config.get("conflict_penalty", 0.5)) * conflict.float()
        class_balance = risk.class_balance_weights(device)
        base_weight = base_weight * class_balance[pseudo].clamp(0.25, 4.0)
        if rescue.any():
            base_weight = torch.where(
                rescue,
                base_weight.clamp_min(float(self.config.get("foreground_min_weight", 0.15))),
                base_weight,
            )
        if area_guard.any():
            guarded_weight = float(self.config.get("area_guard_weight", 0.0))
            base_weight = torch.where(area_guard, base_weight.new_full(base_weight.shape, guarded_weight), base_weight)
        base_weight = base_weight.clamp(0.0, 1.0)
        reliable = (base_weight >= 0.05) & ~conflict & ~area_guard
        singleton = candidate.sum(dim=1) == 1

        negative_set = evidence < float(self.config.get("safe_negative_threshold", 0.03))
        negative_set = negative_set & ~candidate
        stats = self._stats(candidate, pseudo, base_weight, negative_set, singleton, conflict, low_reliability, rescue, area_guard, sam_valid, risk, extra_stats)
        return {
            "pseudo": pseudo.detach(),
            "weight": base_weight.detach(),
            "candidate_set": candidate.detach(),
            "negative_set": negative_set.detach(),
            "reliable": reliable.detach(),
            "singleton": singleton.detach(),
            "area_guard": area_guard.detach(),
            "soft_target": evidence.detach(),
            "stats": stats,
        }
