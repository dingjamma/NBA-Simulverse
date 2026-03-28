"""
CareerArcModel: transformer encoder for predicting next-season stats.
Given a player's last 5 seasons, predict season N+1 stats and auxiliary
binary outcomes (breakout, decline, injury risk).
"""
import torch
import torch.nn as nn

from career_arc.config import CareerModelConfig, CAREER_MODEL_CONFIG


class SeasonInputProjection(nn.Module):
    """
    Projects per-season stat + context + embeddings into d_model space.

    Input dimension = n_stat_features + n_context_features + team_embed_dim + era_embed_dim
    """
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        return self.proj(x)


class SeasonPositionalEncoding(nn.Module):
    """Learnable positional embeddings for the 5-season sequence."""
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Embedding(seq_len, d_model)
        nn.init.normal_(self.pos_embed.weight, mean=0, std=0.02)
        self.register_buffer("positions", torch.arange(seq_len))

    def forward(self, batch_size: int) -> torch.Tensor:
        # Returns (1, seq_len, d_model) broadcast-ready
        return self.pos_embed(self.positions).unsqueeze(0).expand(batch_size, -1, -1)


class AuxHead(nn.Module):
    """Binary classification head (sigmoid output)."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.xavier_uniform_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_model) -> (batch,)
        return torch.sigmoid(self.net(x)).squeeze(-1)


class CareerArcModel(nn.Module):
    def __init__(self, cfg: CareerModelConfig = CAREER_MODEL_CONFIG):
        super().__init__()
        self.cfg = cfg

        # Categorical embeddings
        self.team_embed = nn.Embedding(cfg.n_teams, cfg.team_embed_dim)
        self.era_embed = nn.Embedding(cfg.n_eras, cfg.era_embed_dim)
        nn.init.normal_(self.team_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.era_embed.weight, mean=0, std=0.02)

        # Input projection: concat stats + context + team_emb + era_emb -> d_model
        input_dim = cfg.n_stat_features + cfg.n_context_features + cfg.team_embed_dim + cfg.era_embed_dim
        self.input_proj = SeasonInputProjection(input_dim, cfg.d_model)

        self.pos_enc = SeasonPositionalEncoding(cfg.seq_len, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # Regression head: predict all 10 next-season stats
        self.stat_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_targets),
        )
        for m in self.stat_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Auxiliary binary classification heads
        self.breakout_head = AuxHead(cfg.d_model, cfg.dropout)
        self.decline_head = AuxHead(cfg.d_model, cfg.dropout)
        self.injury_risk_head = AuxHead(cfg.d_model, cfg.dropout)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        stat_seq: torch.Tensor,    # (batch, seq_len, n_stat_features)
        ctx_seq: torch.Tensor,     # (batch, seq_len, n_context_features)
        team_ids: torch.Tensor,    # (batch, seq_len)
        era_ids: torch.Tensor,     # (batch, seq_len)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            stat_pred:     (batch, n_targets) — next season stats
            breakout_prob: (batch,) — probability of >20% pts improvement
            decline_prob:  (batch,) — probability of >15% pts decline
            injury_prob:   (batch,) — probability of games_played < 50
        """
        batch_size = stat_seq.size(0)

        # Embed categorical ids
        team_emb = self.team_embed(team_ids)   # (batch, seq_len, team_embed_dim)
        era_emb  = self.era_embed(era_ids)     # (batch, seq_len, era_embed_dim)

        # Concatenate all per-timestep features
        x = torch.cat([stat_seq, ctx_seq, team_emb, era_emb], dim=-1)  # (batch, seq_len, input_dim)

        # Project to d_model and add positional encoding
        x = self.input_proj(x)                  # (batch, seq_len, d_model)
        x = x + self.pos_enc(batch_size)        # (batch, seq_len, d_model)
        x = self.dropout(x)

        # Transformer encoder: bidirectional attention over all 5 seasons
        x = self.encoder(x)                     # (batch, seq_len, d_model)

        # Take last token as the sequence representation for prediction
        last = x[:, -1, :]                      # (batch, d_model)

        stat_pred = self.stat_head(last)                   # (batch, n_targets)
        breakout_prob = self.breakout_head(last)           # (batch,)
        decline_prob = self.decline_head(last)             # (batch,)
        injury_prob = self.injury_risk_head(last)          # (batch,)

        return stat_pred, breakout_prob, decline_prob, injury_prob

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
