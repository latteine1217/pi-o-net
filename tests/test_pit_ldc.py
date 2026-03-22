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


# ── Task 2 tests ─────────────────────────────────────────────────────────────

def test_physics_loss_zero_for_linear():
    """u=y, v=0, p=0 是 steady NS 的精確解 — 殘差應接近 0。"""
    from pi_onet.pit_ldc import steady_ns_residuals

    def u_fn(xy):
        return xy[:, 1:2]                           # u = y

    def v_fn(xy):
        return torch.zeros_like(xy[:, 0:1])         # v = 0

    def p_fn(xy):
        return torch.zeros_like(xy[:, 0:1])         # p = 0

    xy = torch.rand(20, 2, requires_grad=True)
    ns_x, ns_y, cont = steady_ns_residuals(u_fn, v_fn, p_fn, xy, re=100.0)
    # For u=y, v=0, p=0: NS_x = y·0 + 0·1 + 0 - (1/Re)·(0+0) = 0
    #                    NS_y = 0·0 + y·0 + 0 - (1/Re)·(0+0) = 0
    #                    cont = 0 + 0 = 0
    assert ns_x.abs().max().item() < 1e-5
    assert ns_y.abs().max().item() < 1e-5
    assert cont.abs().max().item() < 1e-5


def test_bc_loss_interface():
    """compute_bc_loss 接受 model_fn(xy, c=int)，回傳非 NaN 非負值。"""
    from pi_onet.pit_ldc import compute_bc_loss

    def zero_fn(xy, c):
        return torch.zeros(xy.shape[0], 1)

    device = torch.device("cpu")
    loss = compute_bc_loss(model_fn=zero_fn, n_per_wall=5, device=device)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_pit_model_fn_bc_gauge_compat():
    """make_pit_model_fn 的 closure 能與 compute_bc_loss / compute_gauge_loss 介面相容。"""
    from pi_onet.pit_ldc import make_pit_model_fn
    from pi_onet.pit_ldc import compute_bc_loss, compute_gauge_loss

    net = _small_model()
    sensors = torch.randn(N_S, 5)
    device = torch.device("cpu")
    model_fn = make_pit_model_fn(net, sensors, 0.0, device)

    # compute_bc_loss 與 compute_gauge_loss 以 keyword c=0/1/2 呼叫 model_fn
    bc_loss = compute_bc_loss(model_fn=model_fn, n_per_wall=5, device=device)
    gauge_loss = compute_gauge_loss(model_fn=model_fn, device=device)
    assert not torch.isnan(bc_loss)
    assert not torch.isnan(gauge_loss)


# ── Task 3 tests ─────────────────────────────────────────────────────────────

def test_smoke_train(tmp_path):
    """3 steps of training produce a checkpoint and non-NaN losses."""
    import subprocess
    import sys

    config_content = f"""
[train]
data_files = [
  "/Users/latteine/Documents/coding/pi-o-net/data/ldc/cavity_Re3000_256_Uniform.mat",
  "/Users/latteine/Documents/coding/pi-o-net/data/ldc/cavity_Re4000_256_Uniform.mat",
  "/Users/latteine/Documents/coding/pi-o-net/data/ldc/cavity_Re5000_256_Uniform.mat",
]
num_interior_sensors = 10
num_boundary_sensors = 8
num_query_points = 16
num_physics_points = 8
num_bc_points = 8
d_model = 16
nhead = 2
num_encoder_layers = 1
dim_feedforward = 32
rff_features = 8
rff_sigma = 1.0
iterations = 3
checkpoint_period = 2
seed = 0
device = "cpu"
artifacts_dir = "{tmp_path}/artifacts"
"""
    cfg_path = tmp_path / "smoke_pit.toml"
    cfg_path.write_text(config_content)

    result = subprocess.run(
        [sys.executable, "-m", "pi_onet.pit_ldc", "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=300,
        cwd="/Users/latteine/Documents/coding/pi-o-net",
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "nan" not in result.stdout.lower(), f"NaN in output:\n{result.stdout}"
    ckpts = list((tmp_path / "artifacts" / "checkpoints").glob("*.pt"))
    assert len(ckpts) > 0, "No checkpoint saved"
