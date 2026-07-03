from __future__ import annotations

import torch
import torch.nn.functional as F


def random_flip(image: torch.Tensor, mask: torch.Tensor | None = None):
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-1,))
        if mask is not None:
            mask = torch.flip(mask, dims=(-1,))
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-2,))
        if mask is not None:
            mask = torch.flip(mask, dims=(-2,))
    return image, mask


def color_jitter(image: torch.Tensor, strength: float = 0.20) -> torch.Tensor:
    scale = 1.0 + (torch.rand((image.shape[0], 1, 1), device=image.device) * 2.0 - 1.0) * strength
    bias = (torch.rand((image.shape[0], 1, 1), device=image.device) * 2.0 - 1.0) * strength
    return (image * scale + bias).clamp(0.0, 1.0)


def cutout(image: torch.Tensor, ratio: float = 0.25) -> torch.Tensor:
    _, h, w = image.shape
    cut_h = max(1, int(h * ratio * torch.rand(()).item()))
    cut_w = max(1, int(w * ratio * torch.rand(()).item()))
    y0 = int(torch.randint(0, max(1, h - cut_h + 1), ()).item())
    x0 = int(torch.randint(0, max(1, w - cut_w + 1), ()).item())
    out = image.clone()
    out[:, y0 : y0 + cut_h, x0 : x0 + cut_w] = 0.0
    return out


def weak_labeled_batch(images: torch.Tensor, masks: torch.Tensor):
    out_images, out_masks = [], []
    for image, mask in zip(images, masks):
        image, mask = random_flip(image, mask)
        out_images.append(image)
        out_masks.append(mask)
    return torch.stack(out_images, dim=0), torch.stack(out_masks, dim=0)


def weak_strong_unlabeled(images: torch.Tensor):
    weak, strong = [], []
    for image in images:
        w_img, _ = random_flip(image, None)
        s_img = color_jitter(w_img, 0.35)
        if torch.rand(()) < 0.7:
            s_img = cutout(s_img, 0.35)
        if torch.rand(()) < 0.5:
            noise = torch.randn_like(s_img) * 0.04
            s_img = (s_img + noise).clamp(0.0, 1.0)
        weak.append(w_img)
        strong.append(s_img)
    return torch.stack(weak, dim=0), torch.stack(strong, dim=0)


def downsample_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask.float().unsqueeze(1), size=size, mode="nearest").squeeze(1).long()

