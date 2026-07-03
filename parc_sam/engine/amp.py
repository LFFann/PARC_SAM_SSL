from __future__ import annotations

from contextlib import nullcontext

try:
    from torch.amp import GradScaler as AmpGradScaler
    from torch.amp import autocast as amp_autocast
except ImportError:  # pragma: no cover
    AmpGradScaler = None
    amp_autocast = None

try:
    from torch.cuda.amp import GradScaler as CudaGradScaler
    from torch.cuda.amp import autocast as cuda_autocast
except ImportError:  # pragma: no cover
    CudaGradScaler = None
    cuda_autocast = None


class NoOpGradScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None


def make_grad_scaler(device_type: str, enabled: bool):
    enabled = bool(enabled) and device_type == "cuda"
    if AmpGradScaler is not None:
        try:
            return AmpGradScaler(device_type, enabled=enabled)
        except TypeError:
            return AmpGradScaler(enabled=enabled)
    if CudaGradScaler is not None:
        return CudaGradScaler(enabled=enabled)
    return NoOpGradScaler()


def autocast_context(device_type: str, enabled: bool):
    enabled = bool(enabled) and device_type == "cuda"
    if not enabled:
        return nullcontext()
    if amp_autocast is not None:
        try:
            return amp_autocast(device_type=device_type, enabled=True)
        except TypeError:
            return amp_autocast(enabled=True)
    if cuda_autocast is not None:
        return cuda_autocast(enabled=True)
    return nullcontext()

