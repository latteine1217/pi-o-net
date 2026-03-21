# tests/test_pit_ldc.py
"""Unit tests for PiT Cross-Attention Operator — model components."""
import pytest
import torch

# 以下 import 在實作前會 ImportError，這是預期的
from pi_onet.pit_ldc import (
    rff_encode,
    SensorEncoder,
    QueryEncoder,
    CrossAttentionOperator,
    create_pit_model,
)

RFF_F = 16
D_MODEL = 32
NHEAD = 2
N_S = 10  # num sensors
N_Q = 8   # num query points


def _small_model() -> CrossAttentionOperator:
    return CrossAttentionOperator(
        rff_features=RFF_F, rff_sigma=1.0, d_model=D_MODEL,
        nhead=NHEAD, dim_feedforward=64, num_encoder_layers=1, attn_dropout=0.0,
    )


def test_rff_shape():
    """γ(z, B) output shape = [N, 2*rff_features]."""
    B = torch.randn(2, RFF_F)
    z = torch.randn(N_Q, 2)
    out = rff_encode(z, B)
    assert out.shape == (N_Q, 2 * RFF_F)


def test_sensor_encoder_output_shape():
    """SensorEncoder output = [N_s+1, d_model] (re_token prepended)."""
    B = torch.randn(2, RFF_F)
    enc = SensorEncoder(
        rff_features=RFF_F, d_model=D_MODEL, nhead=NHEAD,
        dim_feedforward=64, num_encoder_layers=1, attn_dropout=0.0,
    )
    sensors = torch.randn(N_S, 5)
    re_norm = torch.tensor([[0.5]])
    out = enc(sensors, re_norm, B)
    assert out.shape == (N_S + 1, D_MODEL)


def test_query_encoder_output_shape():
    """QueryEncoder output = [N_q, d_model]."""
    B = torch.randn(2, RFF_F)
    enc = QueryEncoder(rff_features=RFF_F, d_model=D_MODEL)
    xy = torch.randn(N_Q, 2)
    c = torch.randint(0, 3, (N_Q,))
    out = enc(xy, c, B)
    assert out.shape == (N_Q, D_MODEL)


def test_b_not_trainable():
    """B buffer must NOT appear in model.parameters()."""
    net = _small_model()
    param_names = {name for name, _ in net.named_parameters()}
    assert "B" not in param_names


def test_component_scale_trainable():
    """component_scale and component_bias must appear in model.parameters()."""
    net = _small_model()
    param_names = {name for name, _ in net.named_parameters()}
    assert "component_scale" in param_names
    assert "component_bias" in param_names


def test_forward_shape():
    """CrossAttentionOperator.forward output = [N_q, 1]."""
    net = _small_model()
    sensors = torch.randn(N_S, 5)
    xy = torch.randn(N_Q, 2)
    c = torch.randint(0, 3, (N_Q,))
    out = net(sensors, 0.0, xy, c)
    assert out.shape == (N_Q, 1)


def test_forward_backward():
    """loss.backward() succeeds and at least one grad is not None."""
    net = _small_model()
    sensors = torch.randn(N_S, 5)
    xy = torch.randn(N_Q, 2)
    c = torch.randint(0, 3, (N_Q,))
    out = net(sensors, 0.0, xy, c)
    out.sum().backward()
    assert net.output_head.weight.grad is not None


def test_create_pit_model():
    """create_pit_model factory creates model with correct d_model."""
    cfg = {
        "rff_features": RFF_F, "rff_sigma": 1.0, "d_model": D_MODEL,
        "nhead": NHEAD, "dim_feedforward": 64,
        "num_encoder_layers": 1, "attn_dropout": 0.0,
    }
    net = create_pit_model(cfg)
    assert net.cross_attn.embed_dim == D_MODEL


def test_component_scale_differentiates():
    """Different component_scale values produce different outputs for u/v/p."""
    net = _small_model()
    sensors = torch.randn(N_S, 5)
    xy = torch.randn(4, 2)
    with torch.no_grad():
        net.component_scale.data = torch.tensor([1.0, 2.0, 3.0])
        net.component_bias.data = torch.tensor([0.0, 1.0, 2.0])
    out_u = net(sensors, 0.0, xy, torch.zeros(4, dtype=torch.long))
    out_v = net(sensors, 0.0, xy, torch.ones(4, dtype=torch.long))
    out_p = net(sensors, 0.0, xy, torch.full((4,), 2, dtype=torch.long))
    assert not torch.allclose(out_u, out_v)
    assert not torch.allclose(out_v, out_p)
