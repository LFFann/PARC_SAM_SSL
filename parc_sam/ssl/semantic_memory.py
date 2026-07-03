from __future__ import annotations

import torch
import torch.nn.functional as F


class SemanticPrototypeMemory:
    def __init__(self, num_classes: int, feature_dim: int, momentum: float = 0.95, temperature: float = 0.07, min_pixels: int = 32):
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.min_pixels = int(min_pixels)
        self.prototypes = torch.zeros(self.num_classes, self.feature_dim)
        self.valid = torch.zeros(self.num_classes, dtype=torch.bool)

    @torch.no_grad()
    def update(self, features: torch.Tensor, masks: torch.Tensor, weights: torch.Tensor | None = None):
        feat = F.normalize(features.detach(), dim=1)
        if weights is None:
            weights = torch.ones_like(masks, dtype=feat.dtype)
        for class_idx in range(self.num_classes):
            mask = (masks == class_idx).float() * weights.float()
            if mask.sum() < self.min_pixels:
                continue
            vec = (feat * mask.unsqueeze(1)).sum(dim=(0, 2, 3)) / mask.sum().clamp_min(1.0)
            vec = F.normalize(vec, dim=0).cpu()
            if self.valid[class_idx]:
                self.prototypes[class_idx] = F.normalize(self.prototypes[class_idx] * self.momentum + vec * (1.0 - self.momentum), dim=0)
            else:
                self.prototypes[class_idx] = vec
                self.valid[class_idx] = True

    def logits(self, features: torch.Tensor) -> torch.Tensor | None:
        if not self.valid.any():
            return None
        protos = F.normalize(self.prototypes.to(features.device), dim=1)
        feat = F.normalize(features, dim=1)
        return torch.einsum("bchw,kc->bkhw", feat, protos) / self.temperature

    def prototype_loss(self, features: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        logits = self.logits(features)
        if logits is None:
            return features.new_tensor(0.0)
        loss_map = F.cross_entropy(logits, target.long(), reduction="none")
        weight = weight.float().clamp_min(0.0)
        return (loss_map * weight).sum() / weight.sum().clamp_min(1.0)

    def state_dict(self) -> dict:
        return {"prototypes": self.prototypes, "valid": self.valid}

    def load_state_dict(self, state: dict):
        self.prototypes = state["prototypes"].cpu()
        self.valid = state["valid"].cpu().bool()

