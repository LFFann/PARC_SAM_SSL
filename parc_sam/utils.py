from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + os.linesep)


def dice_iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    valid = target != ignore_index
    dice, iou = [], []
    for class_idx in range(num_classes):
        p = (pred == class_idx) & valid
        t = (target == class_idx) & valid
        inter = (p & t).sum().item()
        p_sum = p.sum().item()
        t_sum = t.sum().item()
        union = (p | t).sum().item()
        dice.append((2.0 * inter + 1e-6) / (p_sum + t_sum + 1e-6))
        iou.append((inter + 1e-6) / (union + 1e-6))
    return dice, iou


def foreground_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    vals = [v for v in values[1:] if math.isfinite(v)]
    return float(sum(vals) / max(1, len(vals)))


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)

