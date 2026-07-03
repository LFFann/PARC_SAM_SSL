from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def resolve_dataset_root(root: str | Path, dataset_name: str | None, labeled_subdir: str = "labeled") -> Path:
    root = Path(root)
    if (root / labeled_subdir / "image").exists():
        return root
    if dataset_name and (root / dataset_name / labeled_subdir / "image").exists():
        return root / dataset_name
    checked = [root / labeled_subdir / "image"]
    if dataset_name:
        checked.append(root / dataset_name / labeled_subdir / "image")
    raise FileNotFoundError("Dataset root not found. Checked: " + ", ".join(str(p) for p in checked))


def _list_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing image directory: {path}")
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"No images found in {path}")
    return files


def _matching_mask(image_path: Path, image_dir: Path, mask_dir: Path) -> Path:
    rel = image_path.relative_to(image_dir)
    direct = mask_dir / rel
    if direct.exists():
        return direct
    for ext in IMAGE_EXTENSIONS:
        candidate = (mask_dir / rel).with_suffix(ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing mask for image: {image_path}")


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        num_classes: int,
        image_size: int,
        image_dir_name: str = "image",
        mask_dir_name: str = "mask",
        has_mask: bool = True,
        ignore_index: int = 255,
    ):
        self.root = Path(root)
        self.split = split
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.ignore_index = int(ignore_index)
        self.image_dir = self.root / split / image_dir_name
        self.mask_dir = self.root / split / mask_dir_name
        self.records = []
        for image_path in _list_images(self.image_dir):
            self.records.append(
                {
                    "image_path": image_path,
                    "mask_path": _matching_mask(image_path, self.image_dir, self.mask_dir) if has_mask else None,
                    "id": image_path.relative_to(self.image_dir).with_suffix("").as_posix(),
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def _read_array(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            return np.load(path)
        return np.asarray(Image.open(path))

    def _load_image(self, path: Path) -> torch.Tensor:
        arr = self._read_array(path)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = np.clip(arr, 0.0, 1.0)
            image = Image.fromarray((arr * 255).astype(np.uint8))
        else:
            image = Image.fromarray(arr.astype(np.uint8))
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        out = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(out).permute(2, 0, 1).contiguous()

    def _load_mask(self, path: Path) -> torch.Tensor:
        arr = self._read_array(path)
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask = Image.fromarray(arr.astype(np.uint8)).resize((self.image_size, self.image_size), Image.NEAREST)
        out = np.asarray(mask).astype(np.int64)
        if self.num_classes <= 2:
            out = (out > 0).astype(np.int64)
        valid = (out == self.ignore_index) | ((out >= 0) & (out < self.num_classes))
        if not np.all(valid):
            bad = sorted(np.unique(out[~valid]).tolist())
            raise ValueError(f"Mask {path} contains invalid class ids {bad}; expected 0..{self.num_classes - 1}")
        return torch.from_numpy(out).long()

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        item: dict[str, object] = {
            "image": self._load_image(record["image_path"]),
            "id": record["id"],
            "image_path": str(record["image_path"]),
        }
        if record["mask_path"] is not None:
            item["mask"] = self._load_mask(record["mask_path"])
            item["mask_path"] = str(record["mask_path"])
        else:
            item["mask_path"] = ""
        return item

