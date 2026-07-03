from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, trainer, iteration: int, metrics: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": int(iteration),
        "student": trainer.student.state_dict(),
        "ema": trainer.teacher.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "risk": trainer.risk.state_dict(),
        "prototype": trainer.memory.state_dict(),
        "config": trainer.config,
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_student_checkpoint(model, checkpoint_path: str | Path, device):
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("student", payload)
    model.load_state_dict(state, strict=True)
    return payload
