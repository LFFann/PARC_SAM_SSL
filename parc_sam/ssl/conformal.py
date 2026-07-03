from __future__ import annotations

import torch


class ClassConditionalRiskController:
    """Class-wise conformal-style prediction sets updated from labeled batches."""

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.1,
        min_pixels_per_class: int = 64,
        min_quantile: float = 0.02,
        max_quantile: float = 0.95,
        max_foreground_quantile: float = 0.85,
        prior_momentum: float = 0.90,
        class_balance_power: float = 0.50,
    ):
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.min_pixels_per_class = int(min_pixels_per_class)
        self.min_quantile = float(min_quantile)
        self.max_quantile = float(max_quantile)
        self.max_foreground_quantile = float(max_foreground_quantile)
        self.prior_momentum = float(prior_momentum)
        self.class_balance_power = float(class_balance_power)
        self.q_per_class = torch.full((self.num_classes,), 0.45)
        self.global_q = torch.tensor(0.45)
        self.pixel_prior = torch.full((self.num_classes,), 1.0 / max(1, self.num_classes))
        self.fitted = False

    @torch.no_grad()
    def update(self, probs: torch.Tensor, masks: torch.Tensor):
        probs = probs.detach().float().cpu()
        masks = masks.detach().cpu()
        scores_by_class = []
        q = []
        counts = []
        for class_idx in range(self.num_classes):
            pix = masks == class_idx
            counts.append(pix.sum().float())
            scores = 1.0 - probs[:, class_idx][pix]
            if scores.numel() > 0:
                scores_by_class.append(scores)
            if scores.numel() >= self.min_pixels_per_class:
                q.append(torch.quantile(scores, min(0.999, 1.0 - self.alpha)))
            else:
                q.append(None)
        if scores_by_class:
            self.global_q = torch.quantile(torch.cat(scores_by_class), min(0.999, 1.0 - self.alpha))
        q_tensor = torch.stack([value if value is not None else self.global_q for value in q])
        q_tensor = q_tensor.clamp(min=self.min_quantile, max=self.max_quantile)
        if self.num_classes > 1:
            fg_cap = torch.full_like(q_tensor[1:], min(self.max_quantile, self.max_foreground_quantile))
            q_tensor[1:] = torch.minimum(q_tensor[1:], fg_cap)
        self.q_per_class = q_tensor
        count_tensor = torch.stack(counts)
        if count_tensor.sum() > 0:
            batch_prior = count_tensor / count_tensor.sum().clamp_min(1.0)
            self.pixel_prior = self.prior_momentum * self.pixel_prior + (1.0 - self.prior_momentum) * batch_prior
            self.pixel_prior = self.pixel_prior / self.pixel_prior.sum().clamp_min(1e-6)
        self.fitted = True

    def prediction_sets(self, probs: torch.Tensor):
        q = self.q_per_class.to(probs.device).view(1, -1, 1, 1)
        candidate = (1.0 - probs) <= q
        empty = candidate.sum(dim=1, keepdim=True) == 0
        if empty.any():
            candidate = candidate.clone()
            candidate.scatter_(1, probs.argmax(dim=1, keepdim=True), True)
        return candidate, empty.squeeze(1)

    def class_balance_weights(self, device: torch.device | None = None) -> torch.Tensor:
        prior = self.pixel_prior.float()
        inv = (prior.mean() / prior.clamp_min(1e-6)).pow(self.class_balance_power)
        inv = inv / inv.mean().clamp_min(1e-6)
        inv = inv.clamp(0.25, 4.0)
        return inv.to(device) if device is not None else inv

    def state_dict(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "alpha": self.alpha,
            "min_pixels_per_class": self.min_pixels_per_class,
            "min_quantile": self.min_quantile,
            "max_quantile": self.max_quantile,
            "max_foreground_quantile": self.max_foreground_quantile,
            "prior_momentum": self.prior_momentum,
            "class_balance_power": self.class_balance_power,
            "q_per_class": self.q_per_class.tolist(),
            "global_q": float(self.global_q),
            "pixel_prior": self.pixel_prior.tolist(),
            "fitted": self.fitted,
        }

    def load_state_dict(self, state: dict):
        self.num_classes = int(state["num_classes"])
        self.alpha = float(state["alpha"])
        self.min_pixels_per_class = int(state["min_pixels_per_class"])
        self.min_quantile = float(state.get("min_quantile", 0.02))
        self.max_quantile = float(state.get("max_quantile", 0.95))
        self.max_foreground_quantile = float(state.get("max_foreground_quantile", 0.85))
        self.prior_momentum = float(state.get("prior_momentum", 0.90))
        self.class_balance_power = float(state.get("class_balance_power", 0.50))
        self.q_per_class = torch.tensor(state["q_per_class"]).float()
        self.global_q = torch.tensor(state["global_q"]).float()
        self.pixel_prior = torch.tensor(state.get("pixel_prior", [1.0 / self.num_classes] * self.num_classes)).float()
        self.fitted = bool(state.get("fitted", True))
