"""
NBA-GPT: transformer encoder model for predicting next-game stats.
Self-supervised: given last 20 games, predict game 21.
"""
import torch
import torch.nn as nn

from nba_gpt.config import ModelConfig, MODEL_CONFIG
from nba_gpt.model.embeddings import PlayerEmbedding, EraEmbedding, TemporalPositionalEncoding
from nba_gpt.model.heads import PredictionHead


class InputProjection(nn.Module):
    def __init__(self, n_input_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_input_features, d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> (batch, seq_len, d_model)
        return self.proj(x)


class ContextProjection(nn.Module):
    """Projects explicit next-game conditions into d_model space."""
    def __init__(self, n_context_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_context_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        # ctx: (batch, n_context_features) -> (batch, d_model)
        return self.proj(ctx)


class NBAGPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig = MODEL_CONFIG):
        super().__init__()
        self.cfg = cfg

        self.input_proj = InputProjection(cfg.n_input_features, cfg.d_model)
        self.player_emb = PlayerEmbedding(cfg.max_players, cfg.d_model)
        self.era_emb = EraEmbedding(cfg.n_eras, cfg.d_model)
        self.pos_enc = TemporalPositionalEncoding(cfg.sequence_length, cfg.d_model)
        self.context_proj = ContextProjection(cfg.n_context_features, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        self.head = PredictionHead(cfg.d_model, cfg.d_ff, cfg.n_targets, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        player_id: torch.Tensor,    # (batch,)
        era_id: torch.Tensor,       # (batch,)
        input_seq: torch.Tensor,    # (batch, seq_len, n_features)
        next_game_ctx: torch.Tensor,  # (batch, n_context_features)
    ) -> torch.Tensor:              # (batch, n_targets)
        batch_size = input_seq.size(0)

        # Project raw features to d_model
        x = self.input_proj(input_seq)  # (batch, seq_len, d_model)

        # Add context embeddings (broadcast over sequence)
        player_ctx = self.player_emb(player_id).unsqueeze(1)  # (batch, 1, d_model)
        era_ctx = self.era_emb(era_id).unsqueeze(1)           # (batch, 1, d_model)
        pos_ctx = self.pos_enc(batch_size)                     # (batch, seq_len, d_model)

        x = x + player_ctx + era_ctx + pos_ctx
        x = self.dropout(x)

        # Bidirectional attention over all 20 observed games
        x = self.encoder(x)  # (batch, seq_len, d_model)

        # Fuse sequence representation with explicit next-game context
        seq_repr = x[:, -1, :]                          # (batch, d_model)
        ctx_repr = self.context_proj(next_game_ctx)     # (batch, d_model)
        fused = seq_repr + ctx_repr                     # additive fusion

        return self.head(fused)  # (batch, n_targets)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
