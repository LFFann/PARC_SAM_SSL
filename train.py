from __future__ import annotations

import argparse

from parc_sam.config import load_config
from parc_sam.engine import PARCSAMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train PARC-SAM-SSL")
    parser.add_argument("--config", type=str, default="configs/parc_sam_ssl_3class.yaml")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--sam-checkpoint", type=str, default="")
    parser.add_argument("--target-mode", choices=["set_valued", "hard", "conformal_single"], default="")
    parser.add_argument("--disable-sam", action="store_true")
    parser.add_argument("--disable-prototype", action="store_true")
    parser.add_argument("--disable-correlation", action="store_true")
    parser.add_argument("--disable-alignment", action="store_true")
    parser.add_argument("--disable-foreground-guard", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.output_dir:
        overrides.setdefault("experiment", {})["output_dir"] = args.output_dir
    if args.device:
        overrides.setdefault("train", {})["device"] = args.device
    if args.max_iterations > 0:
        overrides.setdefault("train", {})["max_iterations"] = args.max_iterations
    if args.data_root:
        overrides.setdefault("data", {})["root"] = args.data_root
    if args.dataset_name:
        overrides.setdefault("data", {})["dataset_name"] = args.dataset_name
    if args.sam_checkpoint:
        overrides.setdefault("sam", {})["checkpoint"] = args.sam_checkpoint
    if args.target_mode:
        overrides.setdefault("target", {})["target_mode"] = args.target_mode
    if args.disable_sam:
        overrides.setdefault("target", {})["use_sam"] = False
        overrides.setdefault("sam", {})["enabled"] = False
    if args.disable_prototype:
        overrides.setdefault("target", {})["use_prototype"] = False
    if args.disable_correlation:
        overrides.setdefault("target", {})["use_correlation"] = False
    if args.disable_alignment:
        overrides.setdefault("target", {})["use_alignment"] = False
    if args.disable_foreground_guard:
        overrides.setdefault("target", {})["foreground_guard"] = False
    trainer = PARCSAMTrainer(load_config(args.config, overrides))
    trainer.train()


if __name__ == "__main__":
    main()
