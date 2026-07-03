from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from parc_sam.config import load_config
from parc_sam.data.dataset import SegmentationDataset, resolve_dataset_root
from parc_sam.engine.checkpoint import load_student_checkpoint
from parc_sam.engine.evaluator import evaluate
from parc_sam.models import PARCStudent
from parc_sam.utils import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PARC-SAM-SSL student")
    parser.add_argument("--config", type=str, default="configs/parc_sam_ssl_3class.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["train"]["device"] = args.device
    device = resolve_device(cfg["train"]["device"])
    data_cfg = cfg["data"]
    root = resolve_dataset_root(data_cfg["root"], data_cfg.get("dataset_name"), data_cfg.get("labeled_subdir", "labeled"))
    split_key = f"{args.split}_subdir"
    split_dir = data_cfg.get(split_key, args.split)
    dataset = SegmentationDataset(
        root=root,
        split=split_dir,
        num_classes=int(data_cfg["num_classes"]),
        image_size=int(data_cfg["image_size"]),
        image_dir_name=data_cfg.get("image_dir_name", "image"),
        mask_dir_name=data_cfg.get("mask_dir_name", "mask"),
        has_mask=True,
        ignore_index=int(data_cfg.get("ignore_index", 255)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model_cfg = cfg["model"]
    model = PARCStudent(
        in_channels=int(data_cfg.get("in_channels", 3)),
        num_classes=int(data_cfg["num_classes"]),
        base_channels=int(model_cfg.get("base_channels", 32)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        feature_dim=int(model_cfg.get("feature_dim", 128)),
    ).to(device)
    load_student_checkpoint(model, args.checkpoint, device)
    save_dir = Path(args.save_dir) if args.save_dir else Path(args.checkpoint).parent.parent / f"prediction_{args.split}"
    metrics = evaluate(model, loader, int(data_cfg["num_classes"]), device, int(data_cfg.get("ignore_index", 255)), save_dir)
    print(metrics)


if __name__ == "__main__":
    main()

