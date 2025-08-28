# src/model.py
from __future__ import annotations
import torch
import torch.nn as nn

class MLP(nn.Module):
    """MLP simple para clasificación."""
    def __init__(self, input_dim: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] → logits [B, C]."""
        return self.net(x)
