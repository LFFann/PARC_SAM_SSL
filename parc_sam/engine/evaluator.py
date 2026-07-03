from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from parc_sam.utils import dice_iou_per_class, foreground_mean


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


def _save_mask(path: Path, mask: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def _save_rgb(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(path)


def _to_numpy_image(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().float().cpu().clamp(0.0, 1.0)
    if arr.ndim == 3:
        arr = arr.permute(1, 2, 0)
    arr_np = arr.numpy()
    if arr_np.shape[-1] == 1:
        arr_np = np.repeat(arr_np, 3, axis=-1)
    return (arr_np[..., :3] * 255.0).astype(np.uint8)


def _colorize_mask(mask: torch.Tensor | np.ndarray, num_classes: int, ignore_index: int = 255) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().long().numpy()
    else:
        mask_np = mask.astype(np.int64)
    palette = PALETTE
    if num_classes > len(PALETTE):
        rng = np.random.default_rng(2026)
        extra = rng.integers(32, 240, size=(num_classes - len(PALETTE), 3), dtype=np.uint8)
        palette = np.concatenate([PALETTE, extra], axis=0)
    valid = (mask_np >= 0) & (mask_np < num_classes)
    color = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    clipped = np.clip(mask_np, 0, len(palette) - 1)
    color[valid] = palette[clipped[valid]]
    color[mask_np == ignore_index] = np.asarray([128, 128, 128], dtype=np.uint8)
    return color


def _overlay_foreground(image: np.ndarray, mask: torch.Tensor | np.ndarray, num_classes: int, ignore_index: int = 255, alpha: float = 0.45) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().long().numpy()
    else:
        mask_np = mask.astype(np.int64)
    color = _colorize_mask(mask_np, num_classes, ignore_index)
    foreground = (mask_np > 0) & (mask_np != ignore_index)
    out = image.copy().astype(np.float32)
    out[foreground] = out[foreground] * (1.0 - alpha) + color[foreground].astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


def _save_prediction_artifacts(save_dir: Path, sample_id: str, image: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, num_classes: int, ignore_index: int):
    image_np = _to_numpy_image(image)
    pred_np = pred.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    _save_mask(save_dir / "pred_mask" / f"{sample_id}.png", pred_np)
    _save_mask(save_dir / "gt_mask" / f"{sample_id}.png", mask_np)
    _save_rgb(save_dir / "image" / f"{sample_id}.png", image_np)
    _save_rgb(save_dir / "pred_color" / f"{sample_id}.png", _colorize_mask(pred_np, num_classes, ignore_index))
    _save_rgb(save_dir / "gt_color" / f"{sample_id}.png", _colorize_mask(mask_np, num_classes, ignore_index))
    _save_rgb(save_dir / "pred_overlay" / f"{sample_id}.png", _overlay_foreground(image_np, pred_np, num_classes, ignore_index))
    _save_rgb(save_dir / "gt_overlay" / f"{sample_id}.png", _overlay_foreground(image_np, mask_np, num_classes, ignore_index))


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, num_classes: int, device, ignore_index: int = 255, save_dir: str | Path | None = None):
    model.eval()
    device = torch.device(device)
    rows = []
    all_dice, all_iou = [], []
    save_dir = Path(save_dir) if save_dir else None
    for batch in dataloader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        logits = model(image)
        pred = logits.argmax(dim=1)
        for i in range(pred.shape[0]):
            dice, iou = dice_iou_per_class(pred[i], mask[i], num_classes, ignore_index)
            all_dice.append(dice)
            all_iou.append(iou)
            sample_id = batch.get("id", [f"sample_{len(rows)}"])[i]
            rows.append({"id": sample_id, "avg_dice": foreground_mean(dice), "avg_iou": foreground_mean(iou)})
            if save_dir is not None:
                safe_id = str(sample_id).replace("/", "_").replace("\\", "_")
                _save_prediction_artifacts(save_dir, safe_id, batch["image"][i], pred[i], mask[i], num_classes, ignore_index)
    class_dice = np.nanmean(np.asarray(all_dice, dtype=float), axis=0).tolist()
    class_iou = np.nanmean(np.asarray(all_iou, dtype=float), axis=0).tolist()
    metrics = {
        "class_dice": class_dice,
        "class_iou": class_iou,
        "avg_dice": foreground_mean(class_dice),
        "avg_iou": foreground_mean(class_iou),
    }
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "avg_dice", "avg_iou"])
            writer.writeheader()
            writer.writerows(rows)
    return metrics
