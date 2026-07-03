from __future__ import annotations

import logging
import math
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
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
                if bool(self.target_cfg.get("use_sam", True)):
                    sam_out = self.sam_engine(weak_u, teacher_prob)
                else:
                    sam_out = {"valid": False, "prob": teacher_prob.detach(), "source": "disabled_by_target"}

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
            loss = (
                float(loss_cfg.get("supervised", 1.0)) * sup_loss
                + float(loss_cfg.get("pseudo", 1.0)) * pseudo_loss
                + float(loss_cfg.get("proposal_set", 0.35)) * (set_loss + neg_loss)
                + float(loss_cfg.get("prototype", 0.15)) * proto_loss
                + float(loss_cfg.get("correlation", 0.10)) * corr_loss
                + float(loss_cfg.get("sam_alignment", 0.15)) * align_loss
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
            if bool(self.target_cfg.get("use_prototype", True)):
                self.memory.update(out_u["features"], targets["pseudo"], targets["weight"])

            stats = targets["stats"]
            extra_stats = self._training_diagnostics(teacher_prob, student_prob, sam_out, targets)
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
                    student_prob,
                    sam_out,
                    targets,
                    health,
                )
        return {
            "loss": float(loss.detach().cpu()),
            "sup": float(sup_loss.detach().cpu()),
            "pseudo": float(pseudo_loss.detach().cpu()),
            "set": float(set_loss.detach().cpu()),
            "proto": float(proto_loss.detach().cpu()),
            "corr": float(corr_loss.detach().cpu()),
            "align": float(align_loss.detach().cpu()),
            "proposal_avg_set_size": float(stats["proposal_avg_set_size"]),
            "proposal_conflict_ratio": float(stats["proposal_conflict_ratio"]),
            "foreground_candidate_ratio": float(stats["foreground_candidate_ratio"]),
            "background_only_ratio": float(stats["background_only_ratio"]),
            "foreground_rescue_ratio": float(stats["foreground_rescue_ratio"]),
            "risk_low_ratio": float(stats["risk_low_ratio"]),
            "singleton_ratio": float(stats["proposal_singleton_ratio"]),
            "teacher_entropy": float(extra_stats["teacher_entropy_mean"]),
            "student_entropy": float(extra_stats["student_entropy_mean"]),
            "student_foreground_ratio": float(extra_stats["student_foreground_ratio"]),
            "teacher_foreground_ratio": float(extra_stats["teacher_foreground_ratio"]),
            "pseudo_foreground_ratio": float(extra_stats["pseudo_foreground_ratio"]),
            "reliable_ratio": float(extra_stats["reliable_ratio"]),
            "negative_pixel_ratio": float(extra_stats["negative_pixel_ratio"]),
            "sam_used": 1.0 if stats["sam_used"] else 0.0,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "diagnostics": diagnostics,
            "health": health,
        }

    def _training_diagnostics(self, teacher_prob: torch.Tensor, student_prob: torch.Tensor, sam_out: dict, targets: dict) -> dict[str, float]:
        def entropy_mean(prob: torch.Tensor) -> float:
            c = max(1, int(prob.shape[1]))
            ent = -(prob.clamp_min(1e-6) * prob.clamp_min(1e-6).log()).sum(dim=1)
            if c > 1:
                ent = ent / math.log(c)
            return float(ent.mean().detach().cpu())

        student_pred = student_prob.argmax(dim=1)
        teacher_pred = teacher_prob.argmax(dim=1)
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
        out: dict[str, float] = {
            "teacher_entropy_mean": entropy_mean(teacher_prob),
            "student_entropy_mean": entropy_mean(student_prob),
            "teacher_confidence_mean": float(teacher_prob.max(dim=1).values.mean().detach().cpu()),
            "student_confidence_mean": float(student_prob.max(dim=1).values.mean().detach().cpu()),
            "sam_confidence_mean": sam_confidence_mean,
            "teacher_foreground_ratio": float((teacher_pred > 0).float().mean().detach().cpu()),
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
        return out

    def _health_record(self, diagnostics: dict[str, float]) -> dict[str, object]:
        flags: list[str] = []
        fg_min = float(self.visual_cfg.get("min_healthy_foreground_ratio", 0.01))
        bg_max = float(self.visual_cfg.get("max_healthy_background_only_ratio", 0.98))
        singleton_max = float(self.visual_cfg.get("max_healthy_singleton_ratio", 0.995))
        entropy_min = float(self.visual_cfg.get("min_student_entropy", 0.02))
        if not bool(diagnostics.get("sam_used", False)):
            flags.append("sam_inactive")
        if float(diagnostics.get("foreground_candidate_ratio", 0.0)) < fg_min:
            flags.append("foreground_candidate_collapse")
        if float(diagnostics.get("student_foreground_ratio", 0.0)) < fg_min:
            flags.append("student_foreground_collapse")
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
                "student_prob": student_prob.detach().cpu(),
                "sam_prob": sam_prob.detach().cpu() if sam_prob is not None else None,
                "pseudo": targets["pseudo"].detach().cpu(),
                "weight": targets["weight"].detach().cpu(),
                "candidate_set": targets["candidate_set"].detach().cpu(),
                "negative_set": targets["negative_set"].detach().cpu(),
                "reliable": targets["reliable"].detach().cpu(),
                "singleton": targets["singleton"].detach().cpu(),
                "soft_target": targets["soft_target"].detach().cpu(),
            }
            self.visualizer.write(iteration, payload, health)
        except Exception as exc:  # pragma: no cover - visualization must not kill training
            self.logger.warning("visualization failed at iter=%d: %s", iteration, exc)
