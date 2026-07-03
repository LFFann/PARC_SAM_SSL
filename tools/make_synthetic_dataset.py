from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Create a tiny PARC-SAM-SSL smoke dataset")
    parser.add_argument("--root", type=str, default="synthetic_data/smoke")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--labeled", type=int, default=4)
    parser.add_argument("--unlabeled", type=int, default=4)
    parser.add_argument("--val", type=int, default=2)
    parser.add_argument("--test", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _case(size: int, rng: np.random.Generator, with_mask: bool):
    image = Image.new("RGB", (size, size), color=(20, 20, 20))
    mask = Image.new("L", (size, size), color=0)
    draw_img = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)
    cx = int(rng.integers(size // 4, size * 3 // 4))
    cy = int(rng.integers(size // 4, size * 3 // 4))
    r = int(rng.integers(size // 10, size // 5))
    draw_img.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(220, 70, 80))
    draw_mask.ellipse((cx - r, cy - r, cx + r, cy + r), fill=1)
    x0 = int(rng.integers(4, size // 2))
    y0 = int(rng.integers(4, size // 2))
    x1 = min(size - 2, x0 + int(rng.integers(size // 6, size // 3)))
    y1 = min(size - 2, y0 + int(rng.integers(size // 6, size // 3)))
    draw_img.rectangle((x0, y0, x1, y1), fill=(70, 210, 120))
    draw_mask.rectangle((x0, y0, x1, y1), fill=2)
    arr = np.asarray(image).astype(np.int16)
    noise = rng.normal(0, 8, size=arr.shape)
    image = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return image, mask if with_mask else None


def _write_split(root: Path, split: str, n: int, size: int, rng: np.random.Generator, with_mask: bool):
    image_dir = root / split / "image"
    mask_dir = root / split / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    if with_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(n):
        image, mask = _case(size, rng, with_mask)
        name = f"{split}_{idx:03d}.png"
        image.save(image_dir / name)
        if mask is not None:
            mask.save(mask_dir / name)


def main():
    args = parse_args()
    root = Path(args.root)
    rng = np.random.default_rng(args.seed)
    _write_split(root, "labeled", args.labeled, args.image_size, rng, True)
    _write_split(root, "unlabeled", args.unlabeled, args.image_size, rng, False)
    _write_split(root, "val", args.val, args.image_size, rng, True)
    _write_split(root, "test", args.test, args.image_size, rng, True)
    print(f"wrote synthetic dataset to {root}")


if __name__ == "__main__":
    main()

