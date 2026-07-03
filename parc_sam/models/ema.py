from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMATeacher(nn.Module):
    def __init__(self, student: nn.Module, decay: float = 0.995):
        super().__init__()
        self.model = copy.deepcopy(student)
        self.decay = float(decay)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, student: nn.Module):
        for teacher_param, student_param in zip(self.model.parameters(), student.parameters()):
            teacher_param.data.mul_(self.decay).add_(student_param.data, alpha=1.0 - self.decay)
        for teacher_buf, student_buf in zip(self.model.buffers(), student.buffers()):
            teacher_buf.copy_(student_buf)

    def forward(self, *args, **kwargs):
        self.model.eval()
        return self.model(*args, **kwargs)

