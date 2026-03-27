import torch
import pytest
from nba_gpt.config import MODEL_CONFIG, N_TARGETS
from nba_gpt.model.transformer import NBAGPTModel
from nba_gpt.model.embeddings import PlayerEmbedding, EraEmbedding, TemporalPositionalEncoding
from nba_gpt.model.heads import PredictionHead


def _make_batch(batch_size: int = 4):
    cfg = MODEL_CONFIG
    return {
        "player_id": torch.randint(0, 100, (batch_size,)),
        "era_id": torch.randint(0, cfg.n_eras, (batch_size,)),
        "input_seq": torch.randn(batch_size, cfg.sequence_length, cfg.n_input_features),
    }


def test_model_output_shape():
    model = NBAGPTModel(MODEL_CONFIG)
    model.eval()
    batch = _make_batch(4)
    with torch.no_grad():
        out = model(batch["player_id"], batch["era_id"], batch["input_seq"])
    assert out.shape == (4, N_TARGETS)


def test_model_output_shape_batch1():
    model = NBAGPTModel(MODEL_CONFIG)
    model.eval()
    batch = _make_batch(1)
    with torch.no_grad():
        out = model(batch["player_id"], batch["era_id"], batch["input_seq"])
    assert out.shape == (1, N_TARGETS)


def test_model_gradients_flow():
    model = NBAGPTModel(MODEL_CONFIG)
    model.train()
    batch = _make_batch(8)
    target = torch.randn(8, N_TARGETS)
    out = model(batch["player_id"], batch["era_id"], batch["input_seq"])
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_model_parameter_count():
    model = NBAGPTModel(MODEL_CONFIG)
    n_params = model.count_parameters()
    # Should be in the 1M-10M range
    assert 500_000 < n_params < 20_000_000, f"Unexpected param count: {n_params}"


def test_player_embedding_shape():
    emb = PlayerEmbedding(max_players=100, d_model=64)
    ids = torch.tensor([0, 1, 5])
    out = emb(ids)
    assert out.shape == (3, 64)


def test_era_embedding_shape():
    emb = EraEmbedding(n_eras=6, d_model=64)
    ids = torch.tensor([0, 3, 5])
    out = emb(ids)
    assert out.shape == (3, 64)


def test_positional_encoding_shape():
    pos = TemporalPositionalEncoding(seq_len=20, d_model=64)
    out = pos(batch_size=8)
    assert out.shape == (8, 20, 64)


def test_prediction_head_shape():
    head = PredictionHead(d_model=128, d_ff=512, n_targets=6)
    x = torch.randn(16, 128)
    out = head(x)
    assert out.shape == (16, 6)


def test_model_no_nan_outputs():
    model = NBAGPTModel(MODEL_CONFIG)
    model.eval()
    batch = _make_batch(16)
    with torch.no_grad():
        out = model(batch["player_id"], batch["era_id"], batch["input_seq"])
    assert not torch.isnan(out).any(), "Model output contains NaN"
