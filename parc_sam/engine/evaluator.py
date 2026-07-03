from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from parc_sam.utils import dice_iou_per_class, foreground_mean


def _save_mask(path: Path, mask: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


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
                _save_mask(save_dir / "pred_mask" / f"{safe_id}.png", pred[i].cpu().numpy())
                _save_mask(save_dir / "gt_mask" / f"{safe_id}.png", mask[i].cpu().numpy())
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

