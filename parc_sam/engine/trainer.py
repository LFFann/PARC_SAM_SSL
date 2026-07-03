from __future__ import annotations

import logging
import math
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from parc_sam.config import save_resolved_config
from parc_sam.data.augment import weak_labeled_batch, weak_strong_unlabeled
from parc_sam.data.dataset import SegmentationDataset, resolve_dataset_root
from parc_sam.engine.amp import autocast_context, make_grad_scaler
from parc_sam.engine.checkpoint import save_checkpoint
from parc_sam.engine.evaluator import evaluate
from parc_sam.engine.visualization import TrainingVisualizer
from parc_sam.losses import (
    correlation_consistency_loss,
    masked_cosine_alignment,
    negative_set_loss,
    proposal_set_loss,
    supervised_segmentation_loss,
    uncertainty_paced_consistency_loss,
    weighted_pseudo_loss,
)
from parc_sam.models import EMATeacher, PARCStudent
from parc_sam.sam import SAMProposalEngine
from parc_sam.ssl import ClassConditionalRiskController, ProposalSetBuilder, SemanticPrototypeMemory
from parc_sam.utils import append_jsonl, resolve_device, seed_everything


class PARCSAMTrainer:
    def __init__(self, config: dict):
        self.config = config
        exp_cfg = config["experiment"]
        seed_everything(int(exp_cfg.get("seed", 2026)))
        self.output_dir = Path(exp_cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        save_resolved_config(config, self.output_dir)

        data_cfg = config["data"]
        model_cfg = config["model"]
        train_cfg = config["train"]
        self.device = resolve_device(str(train_cfg.get("device", "cuda")))
        self.num_classes = int(data_cfg["num_classes"])
        self.ignore_index = int(data_cfg.get("ignore_index", 255))
        self.student = PARCStudent(
            in_channels=int(data_cfg.get("in_channels", 3)),
            num_classes=self.num_classes,
            base_channels=int(model_cfg.get("base_channels", 32)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            feature_dim=int(model_cfg.get("feature_dim", 128)),
        ).to(self.device)
        self.teacher = EMATeacher(self.student, decay=float(train_cfg.get("ema_decay", 0.995))).to(self.device)
        self.sam_engine = SAMProposalEngine(config["sam"], self.num_classes, int(data_cfg.get("in_channels", 3))).to(self.device)
        self.sam_projector = nn.Conv2d(int(model_cfg.get("feature_dim", 128)), 256, kernel_size=1).to(self.device)

        self.risk = ClassConditionalRiskController(
            self.num_classes,
            alpha=float(config["risk"].get("alpha", 0.1)),
            min_pixels_per_class=int(config["risk"].get("min_pixels_per_class", 64)),
            min_quantile=float(config["risk"].get("min_quantile", 0.02)),
            max_quantile=float(config["risk"].get("max_quantile", 0.95)),
            max_foreground_quantile=float(config["risk"].get("max_foreground_quantile", 0.85)),
            prior_momentum=float(config["risk"].get("prior_momentum", 0.90)),
            class_balance_power=float(config["risk"].get("class_balance_power", 0.50)),
        )
        self.target_cfg = config.get("target", {})
        self.proposal_builder = ProposalSetBuilder(self.num_classes, {**config["risk"], **config["sam"], **self.target_cfg})
        self.memory = SemanticPrototypeMemory(
            self.num_classes,
            feature_dim=int(model_cfg.get("feature_dim", 128)),
            momentum=float(config["prototype"].get("momentum", 0.95)),
            temperature=float(config["prototype"].get("temperature", 0.07)),
            min_pixels=int(config["prototype"].get("min_pixels", 32)),
        )
        params = list(self.student.parameters()) + list(self.sam_projector.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=float(train_cfg.get("lr", 1e-3)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
        self.amp = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = make_grad_scaler(self.device.type, self.amp)
        self.best_metrics = {"avg_dice": -1.0}
        self.visual_cfg = config.get("visualization", {})
        self.visualizer = (
            TrainingVisualizer(self.output_dir, self.num_classes, self.visual_cfg)
            if bool(self.visual_cfg.get("enabled", False))
            else None
        )
        self._build_data()
        self.logger.info("device=%s real_sam=%s sam_source=%s", self.device, self.sam_engine.real_sam_available(), self.sam_engine.sam_source)

    def _unsup_ramp(self, iteration: int) -> float:
        train_cfg = self.config["train"]
        start = int(train_cfg.get("unsup_start_iterations", 0))
        warmup = int(train_cfg.get("warmup_iterations", 0))
        if iteration <= start:
            return 0.0
        if warmup <= 0:
            return 1.0
        progress = min(1.0, max(0.0, float(iteration - start) / float(warmup)))
        mode = str(train_cfg.get("unsup_ramp", "linear")).lower()
        if mode == "sigmoid":
            return float(math.exp(-5.0 * (1.0 - progress) ** 2))
        return progress

    def _entropy_mean(self, prob: torch.Tensor) -> float:
        classes = max(1, int(prob.shape[1]))
        ent = -(prob.clamp_min(1e-6) * prob.clamp_min(1e-6).log()).sum(dim=1)
        if classes > 1:
            ent = ent / math.log(classes)
        return float(ent.mean().detach().cpu())

    def _build_prompt_probability(
        self,
        teacher_prob: torch.Tensor,
        prototype_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        teacher_prob = teacher_prob.detach().float()
        prompt_prob = teacher_prob
        prototype_available = 0.0
        teacher_weight = 1.0
        prototype_weight = 0.0
        area_constraint_ratio = 0.0
        prompt_foreground_before_constraint = float((prompt_prob.argmax(dim=1) > 0).float().mean().detach().cpu())
        if bool(self.target_cfg.get("use_prototype_prompt", True)) and isinstance(prototype_logits, torch.Tensor):
            proto_logits = prototype_logits.detach().float()
            if proto_logits.shape[-2:] != teacher_prob.shape[-2:]:
                proto_logits = F.interpolate(proto_logits, size=teacher_prob.shape[-2:], mode="bilinear", align_corners=False)
            temperature = max(1e-4, float(self.target_cfg.get("prompt_temperature", 1.0)))
            proto_prob = torch.softmax(proto_logits / temperature, dim=1)
            valid = self.memory.valid.to(teacher_prob.device).view(1, -1, 1, 1).float()
            foreground_valid = bool(valid[:, 1:].any()) if valid.shape[1] > 1 else bool(valid.any())
            if foreground_valid:
                proto_prob = proto_prob * valid
                proto_prob = proto_prob + teacher_prob * (1.0 - valid)
                proto_prob = proto_prob / proto_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
                raw_teacher_weight = max(0.0, float(self.target_cfg.get("prompt_teacher_weight", 0.55)))
                raw_prototype_weight = max(0.0, float(self.target_cfg.get("prompt_prototype_weight", 0.45)))
                denom = max(1e-6, raw_teacher_weight + raw_prototype_weight)
                teacher_weight = raw_teacher_weight / denom
                prototype_weight = raw_prototype_weight / denom
                prompt_prob = teacher_weight * teacher_prob + prototype_weight * proto_prob
                prompt_prob = prompt_prob / prompt_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
                prototype_available = float(valid.max().detach().cpu())
                prompt_foreground_before_constraint = float((prompt_prob.argmax(dim=1) > 0).float().mean().detach().cpu())
                prompt_prob, area_constraint_ratio = self._constrain_prompt_foreground(prompt_prob, teacher_prob)
        prompt_pred = prompt_prob.argmax(dim=1)
        teacher_prompt_kl = (
            prompt_prob.clamp_min(1e-6) * (prompt_prob.clamp_min(1e-6).log() - teacher_prob.clamp_min(1e-6).log())
        ).sum(dim=1)
        stats = {
            "prompt_prototype_available": prototype_available,
            "prompt_teacher_weight": float(teacher_weight),
            "prompt_prototype_weight": float(prototype_weight),
            "prompt_entropy_mean": self._entropy_mean(prompt_prob),
            "prompt_confidence_mean": float(prompt_prob.max(dim=1).values.mean().detach().cpu()),
            "prompt_foreground_ratio": float((prompt_pred > 0).float().mean().detach().cpu()),
            "prompt_foreground_before_constraint": prompt_foreground_before_constraint,
            "prompt_area_constraint_ratio": float(area_constraint_ratio),
            "prompt_teacher_kl": float(teacher_prompt_kl.mean().detach().cpu()),
        }
        return prompt_prob.detach(), stats

    def _constrain_prompt_foreground(self, prompt_prob: torch.Tensor, teacher_prob: torch.Tensor) -> tuple[torch.Tensor, float]:
        if prompt_prob.shape[1] <= 1:
            return prompt_prob, 0.0
        max_ratio = float(self.target_cfg.get("prompt_max_foreground_ratio", 0.0))
        min_confidence = float(self.target_cfg.get("prompt_min_foreground_confidence", 0.0))
        if max_ratio <= 0.0 and min_confidence <= 0.0:
            return prompt_prob, 0.0
        b, _, h, w = prompt_prob.shape
        fg_score = prompt_prob[:, 1:].max(dim=1).values
        prompt_pred = prompt_prob.argmax(dim=1)
        foreground = prompt_pred > 0
        keep = foreground.clone()
        if min_confidence > 0.0:
            keep = keep & (fg_score >= min_confidence)
        if max_ratio > 0.0:
            max_pixels = max(1, int(round(max_ratio * h * w)))
            flat_scores = fg_score.flatten(1)
            for batch_idx in range(b):
                fg_count = int(keep[batch_idx].sum().item())
                if fg_count > max_pixels:
                    kept_scores = flat_scores[batch_idx][keep[batch_idx].flatten()]
                    threshold = kept_scores.topk(max_pixels, largest=True).values.min()
                    keep[batch_idx] = keep[batch_idx] & (fg_score[batch_idx] >= threshold)
        drop = foreground & ~keep
        if not bool(drop.any()):
            return prompt_prob, 0.0
        constrained = torch.where(drop.unsqueeze(1), teacher_prob, prompt_prob)
        constrained = constrained / constrained.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return constrained, float(drop.float().mean().detach().cpu())

    def _setup_logger(self):
        logger = logging.getLogger("PARC_SAM_SSL")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
        file_handler = logging.FileHandler(self.output_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def _build_data(self):
        cfg = self.config["data"]
        root = resolve_dataset_root(cfg["root"], cfg.get("dataset_name"), cfg.get("labeled_subdir", "labeled"))
        self.config["data"]["resolved_root"] = str(root)
        common = {
            "root": root,
            "num_classes": self.num_classes,
            "image_size": int(cfg.get("image_size", 256)),
            "image_dir_name": cfg.get("image_dir_name", "image"),
            "mask_dir_name": cfg.get("mask_dir_name", "mask"),
            "ignore_index": self.ignore_index,
        }
        self.labeled_set = SegmentationDataset(split=cfg.get("labeled_subdir", "labeled"), has_mask=True, **common)
        self.unlabeled_set = SegmentationDataset(split=cfg.get("unlabeled_subdir", "unlabeled"), has_mask=False, **common)
        self.val_set = SegmentationDataset(split=cfg.get("val_subdir", "val"), has_mask=True, **common)
        workers = int(cfg.get("num_workers", 2))
        self.labeled_loader = DataLoader(
            self.labeled_set,
            batch_size=int(self.config["train"].get("batch_size_labeled", 4)),
            shuffle=True,
            num_workers=workers,
            drop_last=True,
            pin_memory=self.device.type == "cuda",
        )
        self.unlabeled_loader = DataLoader(
            self.unlabeled_set,
            batch_size=int(self.config["train"].get("batch_size_unlabeled", 4)),
            shuffle=True,
            num_workers=workers,
            drop_last=True,
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=max(0, workers // 2))
        self.logger.info("dataset_root=%s labeled=%d unlabeled=%d val=%d", root, len(self.labeled_set), len(self.unlabeled_set), len(self.val_set))

    def train(self):
        max_iter = int(self.config["train"].get("max_iterations", 10000))
        log_interval = int(self.config["experiment"].get("log_interval", 10))
        val_interval = int(self.config["experiment"].get("val_interval", 200))
        save_interval = int(self.config["experiment"].get("save_interval", 1000))
        visual_interval = int(self.visual_cfg.get("interval", val_interval if val_interval > 0 else 200))
        labeled_iter = cycle(self.labeled_loader)
        unlabeled_iter = cycle(self.unlabeled_loader)
        for iteration in range(1, max_iter + 1):
            record_visuals = self.visualizer is not None and visual_interval > 0 and (iteration == 1 or iteration % visual_interval == 0)
            metrics = self.train_step(next(labeled_iter), next(unlabeled_iter), iteration, record_visuals=record_visuals)
            diagnostics = metrics.pop("diagnostics", {})
            health = metrics.pop("health", {})
            if iteration == 1 or iteration % log_interval == 0:
                msg = "iter=%d " % iteration + " ".join(f"{k}={v:.5f}" for k, v in metrics.items() if isinstance(v, float))
                self.logger.info(msg)
                append_jsonl(self.output_dir / "metrics.jsonl", {"iteration": iteration, **metrics})
                if diagnostics:
                    append_jsonl(self.output_dir / "diagnostics.jsonl", {"iteration": iteration, **diagnostics})
                if health:
                    append_jsonl(self.output_dir / "health.jsonl", {"iteration": iteration, **health})
            elif health and health.get("severity") != "ok":
                append_jsonl(self.output_dir / "health.jsonl", {"iteration": iteration, **health})
            if val_interval > 0 and iteration % val_interval == 0:
                val_metrics = evaluate(self.student, self.val_loader, self.num_classes, self.device, self.ignore_index, self.output_dir / "predictions" / f"val_{iteration}")
                self.logger.info("validation iter=%d avg_dice=%.6f avg_iou=%.6f", iteration, val_metrics["avg_dice"], val_metrics["avg_iou"])
                append_jsonl(self.output_dir / "validation.jsonl", {"iteration": iteration, **val_metrics})
                if val_metrics["avg_dice"] > self.best_metrics["avg_dice"]:
                    self.best_metrics = val_metrics
                    save_checkpoint(self.output_dir / "checkpoints" / "best.pt", self, iteration, val_metrics)
            if save_interval > 0 and iteration % save_interval == 0:
                save_checkpoint(self.output_dir / "checkpoints" / f"iter_{iteration}.pt", self, iteration)
        save_checkpoint(self.output_dir / "checkpoints" / "final.pt", self, max_iter, self.best_metrics)
        return self.best_metrics

    def train_step(self, labeled_batch: dict, unlabeled_batch: dict, iteration: int, record_visuals: bool = False) -> dict[str, float]:
        images_l = labeled_batch["image"].to(self.device)
        masks_l = labeled_batch["mask"].to(self.device)
        images_u = unlabeled_batch["image"].to(self.device)
        images_l, masks_l = weak_labeled_batch(images_l, masks_l)
        weak_u, strong_u = weak_strong_unlabeled(images_u)

        loss_cfg = self.config["loss"]
        self.student.train()
        with autocast_context(self.device.type, self.amp):
            out_l = self.student(images_l, return_features=True)
            sup_loss = supervised_segmentation_loss(out_l["logits"], masks_l, self.num_classes, self.ignore_index)
            with torch.no_grad():
                teacher_out = self.teacher(weak_u, return_features=True)
                teacher_prob = torch.softmax(teacher_out["logits"], dim=1)
                self.risk.update(torch.softmax(out_l["logits"], dim=1), masks_l)
                self.memory.update(out_l["features"], masks_l)
                prompt_proto_logits = (
                    self.memory.logits(teacher_out["features"])
                    if bool(self.target_cfg.get("use_prototype_prompt", True))
                    else None
                )
                prompt_prob, prompt_stats = self._build_prompt_probability(teacher_prob, prompt_proto_logits)
                if bool(self.target_cfg.get("use_sam", True)):
                    sam_out = self.sam_engine(weak_u, prompt_prob)
                else:
                    sam_out = {"valid": False, "prob": prompt_prob.detach(), "source": "disabled_by_target"}

            out_u = self.student(strong_u, return_features=True)
            student_prob = torch.softmax(out_u["logits"], dim=1)
            proto_logits = self.memory.logits(out_u["features"]) if bool(self.target_cfg.get("use_prototype", True)) else None
            targets = self.proposal_builder.build(teacher_prob, self.risk, sam_out=sam_out, prototype_logits=proto_logits)
            min_weight = float(loss_cfg.get("min_pseudo_weight", 0.05))
            pseudo_weight = torch.where(targets["reliable"], targets["weight"].clamp_min(min_weight), torch.zeros_like(targets["weight"]))
            if bool(loss_cfg.get("pseudo_on_singletons_only", str(self.target_cfg.get("target_mode", "set_valued")).lower() == "set_valued")):
                pseudo_weight = pseudo_weight * targets["singleton"].float()
            pseudo_loss = weighted_pseudo_loss(out_u["logits"], targets["pseudo"], pseudo_weight, self.num_classes)
            set_loss = proposal_set_loss(out_u["logits"], targets["candidate_set"], targets["weight"])
            neg_loss = negative_set_loss(out_u["logits"], targets["negative_set"], 1.0 - targets["weight"])
            if bool(self.target_cfg.get("use_prototype", True)):
                proto_loss = self.memory.prototype_loss(out_u["features"], targets["pseudo"], targets["weight"])
            else:
                proto_loss = out_u["logits"].new_tensor(0.0)
            if bool(self.target_cfg.get("use_correlation", True)):
                corr_loss = correlation_consistency_loss(out_u["features"], targets["soft_target"])
            else:
                corr_loss = out_u["logits"].new_tensor(0.0)
            projected = self.sam_projector(out_u["features"])
            if bool(self.target_cfg.get("use_alignment", True)):
                align_loss = masked_cosine_alignment(projected, sam_out.get("sam_embeddings"), targets["candidate_set"])
            else:
                align_loss = out_u["logits"].new_tensor(0.0)
            unsup_ramp = self._unsup_ramp(iteration)
            uncertainty_loss = uncertainty_paced_consistency_loss(
                out_u["logits"],
                targets["soft_target"],
                targets["candidate_set"],
                ramp=unsup_ramp,
                min_weight=float(loss_cfg.get("min_uncertainty_weight", 0.05)),
                ambiguity_bonus=float(loss_cfg.get("ambiguity_bonus", 0.50)),
            )
            unsup_loss = (
                float(loss_cfg.get("pseudo", 1.0)) * pseudo_loss
                + float(loss_cfg.get("proposal_set", 0.35)) * (set_loss + neg_loss)
                + float(loss_cfg.get("uncertainty_consistency", 0.0)) * uncertainty_loss
                + float(loss_cfg.get("prototype", 0.15)) * proto_loss
                + float(loss_cfg.get("correlation", 0.10)) * corr_loss
                + float(loss_cfg.get("sam_alignment", 0.15)) * align_loss
            )
            loss = (
                float(loss_cfg.get("supervised", 1.0)) * sup_loss
                + unsup_ramp * unsup_loss
            )

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        grad_clip = float(self.config["train"].get("grad_clip", 0.0))
        if grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(list(self.student.parameters()) + list(self.sam_projector.parameters()), grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.teacher.update(self.student)
        with torch.no_grad():
            if bool(self.target_cfg.get("use_prototype", True)) and unsup_ramp > 0.05:
                self.memory.update(out_u["features"], targets["pseudo"], targets["weight"])

            stats = targets["stats"]
            extra_stats = self._training_diagnostics(teacher_prob, student_prob, sam_out, targets, prompt_prob, prompt_stats)
            diagnostics = {**stats, **extra_stats}
            health = self._health_record(diagnostics)
            if record_visuals and self.visualizer is not None:
                self._write_visualizations(
                    iteration,
                    images_l,
                    masks_l,
                    labeled_batch,
                    weak_u,
                    strong_u,
                    unlabeled_batch,
                    out_l,
                    teacher_prob,
                    prompt_prob,
                    student_prob,
                    sam_out,
                    targets,
                    health,
                )
        return {
            "loss": float(loss.detach().cpu()),
            "sup": float(sup_loss.detach().cpu()),
            "unsup": float(unsup_loss.detach().cpu()),
            "unsup_ramp": float(unsup_ramp),
            "pseudo": float(pseudo_loss.detach().cpu()),
            "set": float(set_loss.detach().cpu()),
            "uncertainty": float(uncertainty_loss.detach().cpu()),
            "proto": float(proto_loss.detach().cpu()),
            "corr": float(corr_loss.detach().cpu()),
            "align": float(align_loss.detach().cpu()),
            "proposal_avg_set_size": float(stats["proposal_avg_set_size"]),
            "proposal_conflict_ratio": float(stats["proposal_conflict_ratio"]),
            "foreground_candidate_ratio": float(stats["foreground_candidate_ratio"]),
            "background_only_ratio": float(stats["background_only_ratio"]),
            "foreground_rescue_ratio": float(stats["foreground_rescue_ratio"]),
            "area_guard_ratio": float(stats.get("area_guard_ratio", 0.0)),
            "risk_low_ratio": float(stats["risk_low_ratio"]),
            "singleton_ratio": float(stats["proposal_singleton_ratio"]),
            "teacher_entropy": float(extra_stats["teacher_entropy_mean"]),
            "prompt_entropy": float(extra_stats["prompt_entropy_mean"]),
            "student_entropy": float(extra_stats["student_entropy_mean"]),
            "student_foreground_ratio": float(extra_stats["student_foreground_ratio"]),
            "teacher_foreground_ratio": float(extra_stats["teacher_foreground_ratio"]),
            "prompt_foreground_ratio": float(extra_stats["prompt_foreground_ratio"]),
            "prompt_area_constraint_ratio": float(extra_stats["prompt_area_constraint_ratio"]),
            "pseudo_foreground_ratio": float(extra_stats["pseudo_foreground_ratio"]),
            "student_class_1_ratio": float(extra_stats.get("student_class_1_ratio", 0.0)),
            "student_class_2_ratio": float(extra_stats.get("student_class_2_ratio", 0.0)),
            "pseudo_class_1_ratio": float(stats.get("class_1_pseudo_ratio", 0.0)),
            "pseudo_class_2_ratio": float(stats.get("class_2_pseudo_ratio", 0.0)),
            "reliable_ratio": float(extra_stats["reliable_ratio"]),
            "negative_pixel_ratio": float(extra_stats["negative_pixel_ratio"]),
            "sam_iou_mean": float(extra_stats.get("sam_iou_mean", 0.0)),
            "sam_used": 1.0 if stats["sam_used"] else 0.0,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "diagnostics": diagnostics,
            "health": health,
        }

    def _training_diagnostics(
        self,
        teacher_prob: torch.Tensor,
        student_prob: torch.Tensor,
        sam_out: dict,
        targets: dict,
        prompt_prob: torch.Tensor | None = None,
        prompt_stats: dict[str, float] | None = None,
    ) -> dict[str, float]:
        student_pred = student_prob.argmax(dim=1)
        teacher_pred = teacher_prob.argmax(dim=1)
        if prompt_prob is None:
            prompt_prob = teacher_prob
        prompt_pred = prompt_prob.argmax(dim=1)
        pseudo = targets["pseudo"]
        candidate_size = targets["candidate_set"].float().sum(dim=1)
        sam_prob = sam_out.get("prob") if isinstance(sam_out, dict) else None
        if isinstance(sam_prob, torch.Tensor):
            sam_prob = sam_prob.to(teacher_prob.device)
            sam_pred = sam_prob.argmax(dim=1)
            sam_foreground_ratio = float((sam_pred > 0).float().mean().detach().cpu())
            sam_confidence_mean = float(sam_prob.max(dim=1).values.mean().detach().cpu())
        else:
            sam_foreground_ratio = 0.0
            sam_confidence_mean = 0.0
        sam_iou = sam_out.get("iou") if isinstance(sam_out, dict) else None
        if isinstance(sam_iou, torch.Tensor) and sam_iou.numel() > 0:
            sam_iou = sam_iou.float()
            sam_iou_mean = float(sam_iou.mean().detach().cpu())
            sam_iou_max = float(sam_iou.max().detach().cpu())
        else:
            sam_iou_mean = 0.0
            sam_iou_max = 0.0
        prompt_stats = prompt_stats or {}
        out: dict[str, float] = {
            "teacher_entropy_mean": self._entropy_mean(teacher_prob),
            "prompt_entropy_mean": float(prompt_stats.get("prompt_entropy_mean", self._entropy_mean(prompt_prob))),
            "student_entropy_mean": self._entropy_mean(student_prob),
            "teacher_confidence_mean": float(teacher_prob.max(dim=1).values.mean().detach().cpu()),
            "prompt_confidence_mean": float(prompt_stats.get("prompt_confidence_mean", prompt_prob.max(dim=1).values.mean().detach().cpu())),
            "student_confidence_mean": float(student_prob.max(dim=1).values.mean().detach().cpu()),
            "sam_confidence_mean": sam_confidence_mean,
            "sam_iou_mean": sam_iou_mean,
            "sam_iou_max": sam_iou_max,
            "prompt_prototype_available": float(prompt_stats.get("prompt_prototype_available", 0.0)),
            "prompt_teacher_weight": float(prompt_stats.get("prompt_teacher_weight", 1.0)),
            "prompt_prototype_weight": float(prompt_stats.get("prompt_prototype_weight", 0.0)),
            "prompt_teacher_kl": float(prompt_stats.get("prompt_teacher_kl", 0.0)),
            "prompt_foreground_before_constraint": float(prompt_stats.get("prompt_foreground_before_constraint", 0.0)),
            "prompt_area_constraint_ratio": float(prompt_stats.get("prompt_area_constraint_ratio", 0.0)),
            "teacher_foreground_ratio": float((teacher_pred > 0).float().mean().detach().cpu()),
            "prompt_foreground_ratio": float((prompt_pred > 0).float().mean().detach().cpu()),
            "student_foreground_ratio": float((student_pred > 0).float().mean().detach().cpu()),
            "sam_foreground_ratio": sam_foreground_ratio,
            "pseudo_foreground_ratio": float((pseudo > 0).float().mean().detach().cpu()),
            "reliable_ratio": float(targets["reliable"].float().mean().detach().cpu()),
            "negative_pixel_ratio": float(targets["negative_set"].any(dim=1).float().mean().detach().cpu()),
            "candidate_size_1_ratio": float((candidate_size == 1).float().mean().detach().cpu()),
            "candidate_size_2_ratio": float((candidate_size == 2).float().mean().detach().cpu()),
            "candidate_size_ge3_ratio": float((candidate_size >= 3).float().mean().detach().cpu()),
        }
        for class_idx in range(self.num_classes):
            out[f"student_class_{class_idx}_ratio"] = float((student_pred == class_idx).float().mean().detach().cpu())
            out[f"teacher_class_{class_idx}_ratio"] = float((teacher_pred == class_idx).float().mean().detach().cpu())
            out[f"prompt_class_{class_idx}_ratio"] = float((prompt_pred == class_idx).float().mean().detach().cpu())
        return out

    def _health_record(self, diagnostics: dict[str, float]) -> dict[str, object]:
        flags: list[str] = []
        fg_min = float(self.visual_cfg.get("min_healthy_foreground_ratio", 0.01))
        fg_max = float(self.visual_cfg.get("max_healthy_foreground_ratio", 0.80))
        bg_max = float(self.visual_cfg.get("max_healthy_background_only_ratio", 0.98))
        singleton_max = float(self.visual_cfg.get("max_healthy_singleton_ratio", 0.995))
        entropy_min = float(self.visual_cfg.get("min_student_entropy", 0.02))
        if not bool(diagnostics.get("sam_used", False)):
            flags.append("sam_inactive")
        if float(diagnostics.get("foreground_candidate_ratio", 0.0)) < fg_min:
            flags.append("foreground_candidate_collapse")
        if float(diagnostics.get("student_foreground_ratio", 0.0)) < fg_min:
            flags.append("student_foreground_collapse")
        if float(diagnostics.get("student_foreground_ratio", 0.0)) > fg_max:
            flags.append("student_foreground_overexpansion")
        if float(diagnostics.get("pseudo_foreground_ratio", 0.0)) > fg_max:
            flags.append("pseudo_foreground_overexpansion")
        if float(diagnostics.get("background_only_ratio", 0.0)) > bg_max:
            flags.append("background_only_dominance")
        if float(diagnostics.get("proposal_singleton_ratio", 0.0)) > singleton_max:
            flags.append("set_supervision_degenerate")
        if float(diagnostics.get("student_entropy_mean", 1.0)) < entropy_min:
            flags.append("student_overconfident")
        cap = float(self.config["risk"].get("max_foreground_quantile", self.config["risk"].get("max_quantile", 1.0)))
        for class_idx in range(1, self.num_classes):
            if float(diagnostics.get(f"risk_q_class_{class_idx}", 0.0)) >= cap - 1e-4:
                flags.append(f"risk_q_class_{class_idx}_saturated")
            area_limit = float(diagnostics.get(f"class_{class_idx}_area_limit", 0.0))
            pseudo_ratio = float(diagnostics.get(f"class_{class_idx}_pseudo_ratio", 0.0))
            if area_limit > 0 and pseudo_ratio > area_limit * 2.0:
                flags.append(f"class_{class_idx}_pseudo_exceeds_prior")
        if float(diagnostics.get("area_guard_ratio", 0.0)) > 0.25:
            flags.append("area_guard_active")
        if len(flags) >= 3:
            severity = "critical"
        elif flags:
            severity = "warn"
        else:
            severity = "ok"
        return {
            "severity": severity,
            "flags": flags,
            "foreground_candidate_ratio": float(diagnostics.get("foreground_candidate_ratio", 0.0)),
            "student_foreground_ratio": float(diagnostics.get("student_foreground_ratio", 0.0)),
            "background_only_ratio": float(diagnostics.get("background_only_ratio", 0.0)),
            "proposal_singleton_ratio": float(diagnostics.get("proposal_singleton_ratio", 0.0)),
        }

    def _write_visualizations(
        self,
        iteration: int,
        images_l: torch.Tensor,
        masks_l: torch.Tensor,
        labeled_batch: dict,
        weak_u: torch.Tensor,
        strong_u: torch.Tensor,
        unlabeled_batch: dict,
        out_l: dict,
        teacher_prob: torch.Tensor,
        prompt_prob: torch.Tensor,
        student_prob: torch.Tensor,
        sam_out: dict,
        targets: dict,
        health: dict,
    ) -> None:
        try:
            sam_prob = sam_out.get("prob") if isinstance(sam_out, dict) and isinstance(sam_out.get("prob"), torch.Tensor) else None
            payload = {
                "images_l": images_l.detach().cpu(),
                "masks_l": masks_l.detach().cpu(),
                "pred_l": out_l["logits"].detach().argmax(dim=1).cpu(),
                "labeled_ids": labeled_batch.get("id", []),
                "weak_u": weak_u.detach().cpu(),
                "strong_u": strong_u.detach().cpu(),
                "unlabeled_ids": unlabeled_batch.get("id", []),
                "teacher_prob": teacher_prob.detach().cpu(),
                "prompt_prob": prompt_prob.detach().cpu(),
                "student_prob": student_prob.detach().cpu(),
                "sam_prob": sam_prob.detach().cpu() if sam_prob is not None else None,
                "pseudo": targets["pseudo"].detach().cpu(),
                "weight": targets["weight"].detach().cpu(),
                "candidate_set": targets["candidate_set"].detach().cpu(),
                "negative_set": targets["negative_set"].detach().cpu(),
                "reliable": targets["reliable"].detach().cpu(),
                "singleton": targets["singleton"].detach().cpu(),
                "area_guard": targets.get("area_guard", torch.zeros_like(targets["pseudo"], dtype=torch.bool)).detach().cpu(),
                "soft_target": targets["soft_target"].detach().cpu(),
            }
            self.visualizer.write(iteration, payload, health)
        except Exception as exc:  # pragma: no cover - visualization must not kill training
            self.logger.warning("visualization failed at iter=%d: %s", iteration, exc)
