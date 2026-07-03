from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "PARC_SAM_SSL",
        "output_dir": "outputs/PARC_SAM_SSL",
        "seed": 2026,
        "log_interval": 10,
        "val_interval": 200,
        "save_interval": 1000,
    },
    "data": {
        "root": "SampleData",
        "dataset_name": "260513_data_multiclass",
        "num_classes": 3,
        "in_channels": 3,
        "image_size": 256,
        "ignore_index": 255,
        "labeled_subdir": "labeled",
        "unlabeled_subdir": "unlabeled",
        "val_subdir": "val",
        "test_subdir": "test",
        "image_dir_name": "image",
        "mask_dir_name": "mask",
        "num_workers": 2,
    },
    "model": {
        "base_channels": 32,
        "dropout": 0.1,
        "feature_dim": 128,
    },
    "sam": {
        "enabled": True,
        "allow_surrogate_without_checkpoint": False,
        "checkpoint": "sam_vit_b_01ec64.pth",
        "model_type": "vit_b",
        "source": "auto",
        "image_size": 1024,
        "min_prompt_area": 16,
        "min_sam_confidence": 0.50,
        "min_prompt_confidence": 0.55,
        "max_prompt_area_ratio": 0.35,
        "prompt_topk_area_ratio": 0.08,
        "empty_prompt_as_sam": False,
        "sam_background_weight": 0.35,
        "sam_foreground_weight": 0.70,
        "sam_iou_min": 0.40,
        "sam_iou_power": 1.0,
        "detach_sam": True,
    },
    "train": {
        "device": "cuda",
        "max_iterations": 10000,
        "batch_size_labeled": 4,
        "batch_size_unlabeled": 4,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "ema_decay": 0.995,
        "amp": True,
        "grad_clip": 5.0,
        "warmup_iterations": 500,
        "unsup_start_iterations": 0,
        "unsup_ramp": "linear",
    },
    "loss": {
        "supervised": 1.0,
        "pseudo": 1.0,
        "proposal_set": 0.35,
        "uncertainty_consistency": 0.10,
        "prototype": 0.15,
        "correlation": 0.10,
        "sam_alignment": 0.15,
        "min_pseudo_weight": 0.05,
        "min_uncertainty_weight": 0.05,
        "ambiguity_bonus": 0.50,
        "pseudo_on_singletons_only": True,
    },
    "risk": {
        "alpha": 0.10,
        "min_pixels_per_class": 64,
        "min_quantile": 0.02,
        "max_quantile": 0.95,
        "max_foreground_quantile": 0.85,
        "prior_momentum": 0.90,
        "class_balance_power": 0.50,
        "teacher_confidence": 0.60,
        "max_candidate_set_size": 2,
        "agreement_bonus": 0.20,
        "conflict_penalty": 0.50,
        "safe_negative_threshold": 0.03,
    },
    "target": {
        "target_mode": "set_valued",
        "use_risk": True,
        "use_sam": True,
        "use_prototype": True,
        "use_prototype_prompt": True,
        "use_correlation": True,
        "use_alignment": True,
        "prompt_teacher_weight": 0.55,
        "prompt_prototype_weight": 0.45,
        "prompt_temperature": 1.0,
        "prompt_max_foreground_ratio": 0.35,
        "prompt_min_foreground_confidence": 0.40,
        "foreground_guard": True,
        "min_foreground_participation": 0.02,
        "foreground_rescue_confidence": 0.35,
        "foreground_min_weight": 0.15,
        "class_area_guard": True,
        "max_foreground_area_multiplier": 6.0,
        "max_foreground_area_floor": 0.02,
        "max_foreground_area_ceiling": 0.30,
        "area_guard_min_pixels": 16,
        "area_guard_weight": 0.0,
        "prototype_evidence_weight": 0.75,
    },
    "prototype": {
        "momentum": 0.95,
        "temperature": 0.07,
        "min_pixels": 32,
    },
    "visualization": {
        "enabled": True,
        "interval": 200,
        "max_images": 2,
        "panel_size": 192,
        "min_healthy_foreground_ratio": 0.01,
        "max_healthy_foreground_ratio": 0.80,
        "max_healthy_background_only_ratio": 0.98,
        "max_healthy_singleton_ratio": 0.995,
        "min_student_entropy": 0.02,
    },
}


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            if path.suffix.lower() == ".json":
                loaded = json.load(f)
            else:
                if yaml is None:
                    raise ImportError("PyYAML is required for YAML config files")
                loaded = yaml.safe_load(f) or {}
        config = _deep_update(config, loaded)
    if overrides:
        config = _deep_update(config, overrides)
    return config


def save_resolved_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resolved_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return path
