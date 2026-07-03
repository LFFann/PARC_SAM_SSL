from __future__ import annotations

import importlib
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def _temporary_model_path(root: Path):
    old_path = list(sys.path)
    old_modules = {k: v for k, v in sys.modules.items() if k == "Model" or k.startswith("Model.")}
    for key in list(old_modules):
        sys.modules.pop(key, None)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        for key in [k for k in list(sys.modules) if k == "Model" or k.startswith("Model.")]:
            sys.modules.pop(key, None)
        sys.modules.update(old_modules)
        sys.path[:] = old_path


class SAMProposalEngine(nn.Module):
    """Training-time generalist proposal engine.

    It can run a real SAM model when a checkpoint is supplied. For CPU smoke
    tests, configs may explicitly allow a surrogate proposal path.
    """

    def __init__(self, config: dict, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.config = config
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.enabled = bool(config.get("enabled", True))
        self.allow_surrogate = bool(config.get("allow_surrogate_without_checkpoint", False))
        self.sam = None
        self.sam_source = "disabled"
        self.image_size = int(config.get("image_size", 1024))
        self.min_prompt_area = int(config.get("min_prompt_area", 16))
        if self.enabled:
            self.sam = self._load_sam()
        if self.sam is None and self.enabled and not self.allow_surrogate:
            raise RuntimeError(
                "SAM is enabled but no real SAM model could be loaded. "
                "Set sam.checkpoint correctly or use a smoke config with allow_surrogate_without_checkpoint=true."
            )
        if self.sam is None and self.enabled and self.allow_surrogate:
            self.sam_source = "surrogate"
        if self.sam is not None:
            for param in self.sam.parameters():
                param.requires_grad_(False)
            self.sam.eval()

    def real_sam_available(self) -> bool:
        return self.sam is not None

    def _uses_local_prompt_embeddings(self) -> bool:
        return str(self.sam_source).startswith("local:")

    def _load_sam(self):
        checkpoint = Path(str(self.config.get("checkpoint", "")))
        if not checkpoint.exists():
            return None
        model_type = str(self.config.get("model_type", "vit_b"))
        source = str(self.config.get("source", "auto"))
        if source in ("auto", "segment_anything"):
            try:
                from segment_anything import sam_model_registry

                if model_type in sam_model_registry:
                    self.sam_source = "segment_anything"
                    return sam_model_registry[model_type](checkpoint=str(checkpoint))
            except ImportError:
                if source == "segment_anything":
                    raise
        if source in ("auto", "local"):
            root = self._find_local_model_root()
            if root is not None:
                with _temporary_model_path(root):
                    sam_module = importlib.import_module("Model.sam")
                    registry = sam_module.sam_model_registry
                    args = SimpleNamespace(
                        image_size=self.image_size,
                        in_channels=self.in_channels,
                        num_classes=self.num_classes,
                        point_nums=1,
                        box_nums=1,
                        mod="sam",
                        thd=False,
                        chunk=1,
                    )
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        self.sam_source = f"local:{root}"
                        return registry[model_type](args, checkpoint=str(checkpoint))
        return None

    def _find_local_model_root(self) -> Path | None:
        here = Path(__file__).resolve()
        candidates = [
            here.parents[2],
            here.parents[2] / "vendor",
            Path.cwd(),
            Path.cwd() / "PARC_SAM_SSL",
        ]
        for root in candidates:
            if (root / "Model" / "sam" / "__init__.py").exists():
                return root
        return None

    @torch.no_grad()
    def forward(self, images: torch.Tensor, teacher_prob: torch.Tensor) -> dict:
        if not self.enabled:
            return {"valid": False, "prob": teacher_prob.detach(), "source": "disabled"}
        if self.sam is None:
            return self._surrogate(images, teacher_prob)
        device = next(self.sam.parameters()).device
        images = images.to(device)
        teacher_prob = teacher_prob.to(device)
        b, _, h, w = images.shape
        prompts = self._build_prompts(teacher_prob, self.image_size)
        if prompts["image_index"].numel() == 0:
            return self._surrogate(images, teacher_prob)

        sam_images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1) / 255.0
        std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1) / 255.0
        sam_images = (sam_images - mean) / std
        embeddings = self.sam.image_encoder(sam_images)
        prompt_embeddings = embeddings.index_select(0, prompts["image_index"])
        prompt_points, prompt_boxes, prompt_masks = self._format_prompts_for_sam(prompts)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=prompt_points,
            boxes=prompt_boxes,
            masks=prompt_masks,
        )
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=prompt_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        fg_logits_flat = F.interpolate(low_res_masks[:, :1], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        fg_prob = torch.zeros((b, self.num_classes - 1, h, w), device=device)
        fg_iou = torch.zeros((b, self.num_classes - 1), device=device)
        for row, image_idx, class_idx in zip(fg_logits_flat, prompts["image_index"], prompts["class_ids"]):
            fg_prob[image_idx, class_idx - 1] = torch.sigmoid(row)
        for row, image_idx, class_idx in zip(iou_predictions[:, 0], prompts["image_index"], prompts["class_ids"]):
            fg_iou[image_idx, class_idx - 1] = torch.sigmoid(row)
        bg = (1.0 - fg_prob.max(dim=1, keepdim=True).values).clamp(1e-5, 1.0)
        prob = torch.cat([bg, fg_prob], dim=1)
        prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return {
            "valid": True,
            "prob": prob.detach(),
            "iou": fg_iou.detach(),
            "sam_embeddings": embeddings.detach(),
            "source": self.sam_source,
            "prompt_count": int(prompts["image_index"].numel()),
        }

    def _surrogate(self, images: torch.Tensor, teacher_prob: torch.Tensor) -> dict:
        smooth = F.avg_pool2d(teacher_prob.detach(), kernel_size=5, stride=1, padding=2)
        sharpened = torch.softmax(torch.log(smooth.clamp_min(1e-6)) / 0.75, dim=1)
        return {
            "valid": True,
            "prob": sharpened,
            "iou": sharpened.flatten(2).amax(dim=2),
            "sam_embeddings": None,
            "source": "surrogate",
            "prompt_count": 0,
        }

    def _format_prompts_for_sam(self, prompts: dict[str, torch.Tensor]):
        if not self._uses_local_prompt_embeddings():
            return (prompts["point_coords"], prompts["point_labels"]), prompts["boxes"], prompts["mask_inputs"]

        encoder = self.sam.prompt_encoder
        point_coords = prompts["point_coords"].float() / float(max(1, self.image_size))
        box_coords = prompts["boxes"].float().reshape(-1, 2, 2) / float(max(1, self.image_size))
        point_embeddings = encoder.pe_layer.forward_with_coords(point_coords, (1, 1))
        point_labels = prompts["point_labels"].long()
        point_embeddings = point_embeddings.clone()
        point_embeddings[point_labels == 0] += encoder.point_embeddings[0].weight
        point_embeddings[point_labels == 1] += encoder.point_embeddings[1].weight

        box_embeddings = encoder.pe_layer.forward_with_coords(box_coords, (1, 1)).clone()
        box_embeddings[:, 0, :] += encoder.point_embeddings[2].weight
        box_embeddings[:, 1, :] += encoder.point_embeddings[3].weight
        return (point_embeddings, point_labels), box_embeddings, prompts["mask_inputs"]

    def _build_prompts(self, prob: torch.Tensor, sam_size: int) -> dict[str, torch.Tensor]:
        b, classes, h, w = prob.shape
        device = prob.device
        image_indices = []
        class_ids = []
        boxes = []
        points = []
        labels = []
        masks = []
        for bi in range(b):
            union_fg = prob[bi, 1:].max(dim=0).values if classes > 1 else prob[bi, 0]
            neg_mask = union_fg < 0.10
            if neg_mask.any():
                neg_y, neg_x = torch.where(neg_mask)
                neg_point = torch.stack([neg_x.float().mean(), neg_y.float().mean()])
            else:
                neg_point = torch.tensor([0.0, 0.0], device=device)
            for class_idx in range(1, classes):
                cls = prob[bi, class_idx]
                hard = cls > max(float(self.config.get("min_sam_confidence", 0.5)), float(cls.mean()))
                if int(hard.sum()) >= self.min_prompt_area:
                    yy, xx = torch.where(hard)
                    x0 = xx.min().float()
                    x1 = xx.max().float()
                    y0 = yy.min().float()
                    y1 = yy.max().float()
                    weights = cls[hard]
                    pos_x = (xx.float() * weights).sum() / weights.sum().clamp_min(1e-6)
                    pos_y = (yy.float() * weights).sum() / weights.sum().clamp_min(1e-6)
                else:
                    flat_idx = cls.flatten().argmax()
                    pos_y = (flat_idx // w).float()
                    pos_x = (flat_idx % w).float()
                    pad = max(h, w) * 0.15
                    x0 = (pos_x - pad).clamp(0, w - 1)
                    x1 = (pos_x + pad).clamp(0, w - 1)
                    y0 = (pos_y - pad).clamp(0, h - 1)
                    y1 = (pos_y + pad).clamp(0, h - 1)
                scale_x = sam_size / max(1, w)
                scale_y = sam_size / max(1, h)
                box = torch.stack([x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y])
                pos = torch.stack([pos_x * scale_x, pos_y * scale_y])
                neg = torch.stack([neg_point[0] * scale_x, neg_point[1] * scale_y])
                image_indices.append(torch.tensor(bi, device=device))
                class_ids.append(torch.tensor(class_idx, device=device))
                boxes.append(box)
                points.append(torch.stack([pos, neg], dim=0))
                labels.append(torch.tensor([1, 0], device=device, dtype=torch.long))
                masks.append(cls.unsqueeze(0))
        if not image_indices:
            empty_long = torch.empty(0, device=device, dtype=torch.long)
            return {
                "image_index": empty_long,
                "class_ids": empty_long,
                "boxes": torch.empty(0, 4, device=device),
                "point_coords": torch.empty(0, 2, 2, device=device),
                "point_labels": torch.empty(0, 2, device=device, dtype=torch.long),
                "mask_inputs": torch.empty(0, 1, 256, 256, device=device),
            }
        mask_inputs = F.interpolate(torch.stack(masks, dim=0), size=(256, 256), mode="bilinear", align_corners=False)
        return {
            "image_index": torch.stack(image_indices).long(),
            "class_ids": torch.stack(class_ids).long(),
            "boxes": torch.stack(boxes, dim=0).float(),
            "point_coords": torch.stack(points, dim=0).float(),
            "point_labels": torch.stack(labels, dim=0).long(),
            "mask_inputs": mask_inputs.float(),
        }
