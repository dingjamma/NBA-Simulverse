"""
Embedding modules: player, era, and positional.
"""
import torch
import torch.nn as nn


class PlayerEmbedding(nn.Module):
    def __init__(self, max_players: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(max_players, d_model)
        nn.init.normal_(self.embed.weight, mean=0, std=0.02)

    def forward(self, player_id: torch.Tensor) -> torch.Tensor:
        # player_id: (batch,) -> (batch, d_model)
        return self.embed(player_id)


class EraEmbedding(nn.Module):
    def __init__(self, n_eras: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(n_eras, d_model)
        nn.init.normal_(self.embed.weight, mean=0, std=0.02)

    def forward(self, era_id: torch.Tensor) -> torch.Tensor:
        # era_id: (batch,) -> (batch, d_model)
        return self.embed(era_id)


class TemporalPositionalEncoding(nn.Module):
    """Learnable positional embeddings for the 20-game sequence."""
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Embedding(seq_len, d_model)
        nn.init.normal_(self.pos_embed.weight, mean=0, std=0.02)
        self.register_buffer("positions", torch.arange(seq_len))

    def forward(self, batch_size: int) -> torch.Tensor:
        # Returns (1, seq_len, d_model) broadcast-ready
        return self.pos_embed(self.positions).unsqueeze(0).expand(batch_size, -1, -1)
