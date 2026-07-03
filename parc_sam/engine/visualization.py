from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from parc_sam.utils import append_jsonl


PALETTE = np.asarray(
    [
        [20, 20, 24],
        [230, 76, 60],
        [52, 152, 219],
        [46, 204, 113],
        [241, 196, 15],
        [155, 89, 182],
        [230, 126, 34],
        [26, 188, 156],
    ],
    dtype=np.uint8,
)


def _to_numpy_image(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().float().cpu().clamp(0.0, 1.0)
    if arr.ndim == 3:
        arr = arr.permute(1, 2, 0)
    arr_np = arr.numpy()
    if arr_np.shape[-1] == 1:
        arr_np = np.repeat(arr_np, 3, axis=-1)
    return (arr_np[..., :3] * 255.0).astype(np.uint8)


def _colorize_mask(mask: torch.Tensor | np.ndarray, num_classes: int) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().long().numpy()
    else:
        mask_np = mask.astype(np.int64)
    palette = PALETTE
    if num_classes > len(PALETTE):
        rng = np.random.default_rng(2026)
        extra = rng.integers(32, 240, size=(num_classes - len(PALETTE), 3), dtype=np.uint8)
        palette = np.concatenate([PALETTE, extra], axis=0)
    mask_np = np.clip(mask_np, 0, len(palette) - 1)
    return palette[mask_np]


def _heatmap(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().float().cpu().numpy()
    else:
        arr = value.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    low = float(arr.min())
    high = float(arr.max())
    if high > low:
        arr = (arr - low) / (high - low)
    else:
        arr = np.zeros_like(arr)
    r = np.clip(2.0 * arr, 0.0, 1.0)
    g = np.clip(2.0 * (1.0 - np.abs(arr - 0.5)), 0.0, 1.0)
    b = np.clip(2.0 * (1.0 - arr), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _overlay(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return (image.astype(np.float32) * (1.0 - alpha) + mask_rgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)


def _entropy(prob: torch.Tensor) -> torch.Tensor:
    c = max(1, prob.shape[0])
    ent = -(prob.clamp_min(1e-6) * prob.clamp_min(1e-6).log()).sum(dim=0)
    return ent / float(np.log(c)) if c > 1 else ent


def _panel(title: str, image: np.ndarray, size: tuple[int, int]) -> Image.Image:
    pil = Image.fromarray(image).resize(size, Image.BILINEAR)
    canvas = Image.new("RGB", (size[0], size[1] + 22), (255, 255, 255))
    canvas.paste(pil, (0, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title[:48], fill=(0, 0, 0))
    return canvas


def _make_grid(panels: list[tuple[str, np.ndarray]], cols: int = 4, panel_size: tuple[int, int] = (192, 192)) -> Image.Image:
    rows = int(np.ceil(len(panels) / max(1, cols)))
    tile_w, tile_h = panel_size[0], panel_size[1] + 22
    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), (250, 250, 250))
    for idx, (title, image) in enumerate(panels):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        grid.paste(_panel(title, image, panel_size), (x, y))
    return grid


def _safe_name(value: Any) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


class TrainingVisualizer:
    def __init__(self, output_dir: str | Path, num_classes: int, config: dict):
        self.output_dir = Path(output_dir)
        self.num_classes = int(num_classes)
        self.config = config
        self.max_images = int(config.get("max_images", 2))
        self.panel_size = int(config.get("panel_size", 192))

    def write(self, iteration: int, payload: dict[str, Any], health: dict[str, Any]) -> None:
        records = []
        max_images = max(1, self.max_images)
        for sample_idx in range(min(max_images, int(payload["strong_u"].shape[0]))):
            sample_id = _safe_name(payload.get("unlabeled_ids", [f"u{sample_idx}"])[sample_idx])
            records.extend(self._write_unlabeled(iteration, sample_idx, sample_id, payload, health))
        if "images_l" in payload:
            for sample_idx in range(min(max_images, int(payload["images_l"].shape[0]))):
                sample_id = _safe_name(payload.get("labeled_ids", [f"l{sample_idx}"])[sample_idx])
                records.extend(self._write_labeled(iteration, sample_idx, sample_id, payload))
        if records:
            append_jsonl(
                self.output_dir / "visualizations" / "manifest.jsonl",
                {"iteration": int(iteration), "health": health, "records": records},
            )

    def _write_labeled(self, iteration: int, idx: int, sample_id: str, payload: dict[str, Any]) -> list[dict[str, str]]:
        image = _to_numpy_image(payload["images_l"][idx])
        gt = payload["masks_l"][idx].detach().cpu()
        pred = payload["pred_l"][idx].detach().cpu()
        gt_rgb = _colorize_mask(gt, self.num_classes)
        pred_rgb = _colorize_mask(pred, self.num_classes)
        error = (gt != pred).float()
        panels = [
            ("labeled image", image),
            ("ground truth", _overlay(image, gt_rgb)),
            ("student pred", _overlay(image, pred_rgb)),
            ("error map", _heatmap(error)),
        ]
        path = self.output_dir / "visualizations" / "paper" / f"iter_{iteration:06d}_{sample_id}_labeled.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        _make_grid(panels, cols=4, panel_size=(self.panel_size, self.panel_size)).save(path)
        return [{"kind": "paper_labeled", "path": str(path)}]

    def _write_unlabeled(
        self,
        iteration: int,
        idx: int,
        sample_id: str,
        payload: dict[str, Any],
        health: dict[str, Any],
    ) -> list[dict[str, str]]:
        weak = _to_numpy_image(payload["weak_u"][idx])
        strong = _to_numpy_image(payload["strong_u"][idx])
        teacher_prob = payload["teacher_prob"][idx].detach().cpu()
        prompt_prob = payload["prompt_prob"][idx].detach().cpu() if payload.get("prompt_prob") is not None else teacher_prob
        student_prob = payload["student_prob"][idx].detach().cpu()
        sam_prob = payload["sam_prob"][idx].detach().cpu() if payload.get("sam_prob") is not None else teacher_prob
        candidate = payload["candidate_set"][idx].detach().cpu()
        negative = payload["negative_set"][idx].detach().cpu()
        pseudo = payload["pseudo"][idx].detach().cpu()
        weight = payload["weight"][idx].detach().cpu()
        reliable = payload["reliable"][idx].detach().cpu()
        singleton = payload["singleton"][idx].detach().cpu()
        area_guard = payload.get("area_guard")
        area_guard = area_guard[idx].detach().cpu() if isinstance(area_guard, torch.Tensor) else torch.zeros_like(singleton)
        teacher_pred = teacher_prob.argmax(dim=0)
        prompt_pred = prompt_prob.argmax(dim=0)
        student_pred = student_prob.argmax(dim=0)
        sam_pred = sam_prob.argmax(dim=0)
        set_size = candidate.float().sum(dim=0)
        negative_size = negative.float().sum(dim=0)
        paper_panels = [
            ("weak image", weak),
            ("strong image", strong),
            ("teacher", _overlay(weak, _colorize_mask(teacher_pred, self.num_classes))),
            ("prompt prior", _overlay(weak, _colorize_mask(prompt_pred, self.num_classes))),
            ("SAM proposal", _overlay(weak, _colorize_mask(sam_pred, self.num_classes))),
            ("candidate pseudo", _overlay(weak, _colorize_mask(pseudo, self.num_classes))),
            ("student pred", _overlay(strong, _colorize_mask(student_pred, self.num_classes))),
            ("candidate size", _heatmap(set_size)),
            ("pseudo weight", _heatmap(weight)),
        ]
        diag_panels = [
            ("teacher entropy", _heatmap(_entropy(teacher_prob))),
            ("prompt entropy", _heatmap(_entropy(prompt_prob))),
            ("student entropy", _heatmap(_entropy(student_prob))),
            ("SAM confidence", _heatmap(sam_prob.max(dim=0).values)),
            ("soft target confidence", _heatmap(payload["soft_target"][idx].detach().cpu().max(dim=0).values)),
            ("candidate size", _heatmap(set_size)),
            ("negative classes", _heatmap(negative_size)),
            ("reliable pixels", _heatmap(reliable.float())),
            ("singleton pixels", _heatmap(singleton.float())),
            ("area guard", _heatmap(area_guard.float())),
        ]
        records = []
        paper_path = self.output_dir / "visualizations" / "paper" / f"iter_{iteration:06d}_{sample_id}_unlabeled.png"
        diagnostic_path = self.output_dir / "visualizations" / "diagnostic" / f"iter_{iteration:06d}_{sample_id}_maps.png"
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        _make_grid(paper_panels, cols=4, panel_size=(self.panel_size, self.panel_size)).save(paper_path)
        _make_grid(diag_panels, cols=4, panel_size=(self.panel_size, self.panel_size)).save(diagnostic_path)
        records.append({"kind": "paper_unlabeled", "path": str(paper_path)})
        records.append({"kind": "diagnostic_unlabeled", "path": str(diagnostic_path)})
        if health.get("severity", "ok") != "ok":
            failure_panels = paper_panels[:2] + diag_panels[:6]
            failure_path = self.output_dir / "visualizations" / "failure" / f"iter_{iteration:06d}_{sample_id}_{health['severity']}.png"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            grid = _make_grid(failure_panels, cols=4, panel_size=(self.panel_size, self.panel_size))
            draw = ImageDraw.Draw(grid)
            draw.rectangle((0, 0, grid.size[0], 26), fill=(255, 245, 220))
            draw.text((8, 6), "flags: " + ", ".join(health.get("flags", []))[:160], fill=(0, 0, 0))
            grid.save(failure_path)
            records.append({"kind": "failure_unlabeled", "path": str(failure_path)})
        return records
