"""
Prediction head: maps transformer output to raw stat predictions.
"""
import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_targets: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_targets),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_model) -> (batch, n_targets)
        return self.net(x)
