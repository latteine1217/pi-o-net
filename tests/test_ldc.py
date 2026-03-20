# tests/test_ldc.py
"""Tests for LDC dataset loading and sensor sampling."""
from __future__ import annotations
import numpy as np
import pytest
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def make_fake_mat(tmp_path: Path, re: int) -> Path:
    """Create a minimal fake .mat file for testing (5x5 grid)."""
    import scipy.io
    import numpy as np
    n = 5
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    mat = {
        "X_ref": x.astype(np.float64),
        "Y_ref": y.astype(np.float64),
        "U_ref": np.ones((n, n), dtype=np.float64) * (re / 1000.0),
        "V_ref": np.zeros((n, n), dtype=np.float64),
        "P_ref": np.zeros((n, n), dtype=np.float64),
    }
    path = tmp_path / f"cavity_Re{re}_5_Uniform.mat"
    scipy.io.savemat(str(path), mat)
    return path


# ── Task 1 tests ─────────────────────────────────────────────────────────────

def test_load_ldc_mat_keys(tmp_path):
    from pi_onet.ldc_dataset import load_ldc_mat
    path = make_fake_mat(tmp_path, 3000)
    result = load_ldc_mat(path)
    assert set(result.keys()) == {"x", "y", "u", "v", "p"}


def test_load_ldc_mat_shape(tmp_path):
    from pi_onet.ldc_dataset import load_ldc_mat
    path = make_fake_mat(tmp_path, 3000)
    result = load_ldc_mat(path)
    assert result["x"].shape == (25,)  # 5x5 flattened
    assert result["u"].shape == (25,)


def test_sample_boundary_indices_count():
    from pi_onet.ldc_dataset import sample_boundary_indices
    # 5 per wall = 20 total
    indices = sample_boundary_indices(grid_size=257, n_per_wall=5)
    assert len(indices) == 20
    assert len(np.unique(indices)) == 20  # no duplicates


def test_sample_interior_indices_count():
    from pi_onet.ldc_dataset import sample_interior_indices
    rng = np.random.default_rng(42)
    indices = sample_interior_indices(rng=rng, grid_size=257, n=80)
    assert len(indices) == 80
    # All indices must be in interior (not boundary row/col)
    rows = indices // 257
    cols = indices % 257
    assert np.all(rows >= 1) and np.all(rows <= 255)
    assert np.all(cols >= 1) and np.all(cols <= 255)


def test_ldc_dataset_branch_shape(tmp_path):
    from pi_onet.ldc_dataset import LDCDataset
    paths = [make_fake_mat(tmp_path, re) for re in [3000, 4000, 5000]]
    # Use a tiny 5x5 grid; interior pool = 3x3=9, boundary = 4*n_per_wall
    ds = LDCDataset(
        mat_paths=paths,
        num_interior_sensors=3,
        num_boundary_sensors=4,  # 1 per wall
        train_ratio=0.8,
        seed=42,
    )
    # branch: [3, 1 + 3*3 + 4*3] = [3, 1 + 9 + 12] = [3, 22]
    assert ds.branch_all.shape == (3, 22)


def test_ldc_dataset_re_values(tmp_path):
    from pi_onet.ldc_dataset import LDCDataset
    paths = [make_fake_mat(tmp_path, re) for re in [3000, 4000, 5000]]
    ds = LDCDataset(
        mat_paths=paths, num_interior_sensors=3, num_boundary_sensors=4,
        train_ratio=0.8, seed=42,
    )
    np.testing.assert_allclose(ds.re_values, [3000.0, 4000.0, 5000.0])


def test_ldc_dataset_sample_trunk_train(tmp_path):
    from pi_onet.ldc_dataset import LDCDataset
    paths = [make_fake_mat(tmp_path, re) for re in [3000, 4000, 5000]]
    ds = LDCDataset(
        mat_paths=paths, num_interior_sensors=3, num_boundary_sensors=4,
        train_ratio=0.8, seed=42,
    )
    rng = np.random.default_rng(0)
    trunk_pts, ref_vals = ds.sample_train_trunk(rng=rng, n_per_re=6)
    # trunk_pts: [6*3, 3], ref_vals: [3, 6*3]
    assert trunk_pts.shape == (18, 3)
    assert ref_vals.shape == (3, 18)
    # c values must be 0, 1, or 2
    assert set(trunk_pts[:, 2].astype(int).tolist()).issubset({0, 1, 2})


def test_ldc_dataset_val_trunk_shapes(tmp_path):
    from pi_onet.ldc_dataset import LDCDataset
    paths = [make_fake_mat(tmp_path, re) for re in [3000, 4000, 5000]]
    ds = LDCDataset(
        mat_paths=paths, num_interior_sensors=3, num_boundary_sensors=4,
        train_ratio=0.8, seed=42,
    )
    trunk_pts, ref_vals = ds.sample_val_trunk_all()
    # 5x5=25 total points, 80% train → 20 train, 5 val
    # val trunk: 5 pts * 3 components = 15
    assert trunk_pts.shape[1] == 3
    assert ref_vals.shape[0] == 3  # 3 Re cases
    assert trunk_pts.shape[0] == ref_vals.shape[1]
    assert set(trunk_pts[:, 2].astype(int).tolist()).issubset({0, 1, 2})


# ── Task 2 tests ─────────────────────────────────────────────────────────────

def test_ldc_fourier_trunk_output_shape():
    from pi_onet.ldc_train import LDCFourierTrunkNet
    from pi_onet.train import SimpleMLP
    from deepxde import config as dde_config
    import torch
    num_features, sigma, latent_width = 32, 5.0, 64
    core = SimpleMLP(
        layer_sizes=[2 * num_features + 8, 64, latent_width],
        activation="tanh",
        kernel_initializer="Glorot normal",
    )
    trunk = LDCFourierTrunkNet(num_features=num_features, sigma=sigma, core_net=core)
    x = torch.randn(20, 3)  # (x, y, c)
    x[:, 2] = torch.randint(0, 3, (20,)).float()
    out = trunk(x)
    assert out.shape == (20, latent_width)


def test_ldc_fourier_trunk_b_shape():
    from pi_onet.ldc_train import LDCFourierTrunkNet
    from pi_onet.train import SimpleMLP
    num_features = 32
    core = SimpleMLP([2*num_features+8, 64], "tanh", "Glorot normal")
    trunk = LDCFourierTrunkNet(num_features=num_features, sigma=5.0, core_net=core)
    assert trunk.B.shape == (2, num_features)  # NOT (3, num_features)


def test_ldc_fourier_trunk_b_not_trainable():
    from pi_onet.ldc_train import LDCFourierTrunkNet
    from pi_onet.train import SimpleMLP
    core = SimpleMLP([72, 64], "tanh", "Glorot normal")
    trunk = LDCFourierTrunkNet(num_features=32, sigma=5.0, core_net=core)
    param_names = [n for n, _ in trunk.named_parameters()]
    assert "B" not in param_names


def test_ldc_fourier_trunk_rejects_zero_features():
    from pi_onet.ldc_train import LDCFourierTrunkNet
    from pi_onet.train import SimpleMLP
    with pytest.raises(ValueError):
        LDCFourierTrunkNet(num_features=0, sigma=5.0, core_net=SimpleMLP([8, 4], "tanh", "Glorot normal"))


def test_ldc_deeponet_forward_shape():
    from pi_onet.ldc_train import LDCDeepONet, LDCFourierTrunkNet
    from pi_onet.train import ResNetBranchNet, SimpleMLP
    import torch
    latent = 64
    branch = ResNetBranchNet(flat_dim=301, hidden_dims=[128, 128], latent_width=latent)
    core = SimpleMLP([2*32+8, 128, latent], "tanh", "Glorot normal")
    trunk = LDCFourierTrunkNet(num_features=32, sigma=5.0, core_net=core)
    net = LDCDeepONet(branch_net=branch, trunk_net=trunk)
    b_in = torch.randn(3, 301)    # 3 Re cases
    t_in = torch.randn(60, 3)     # 20 pts × 3 components
    t_in[:, 2] = torch.tensor([0,1,2]*20).float()
    out = net(b_in, t_in)
    assert out.shape == (3, 60)


def test_ldc_deeponet_backward():
    from pi_onet.ldc_train import LDCDeepONet, LDCFourierTrunkNet
    from pi_onet.train import SimpleMLP
    import torch
    latent = 32
    branch = SimpleMLP([301, 64, latent], "tanh", "Glorot normal")
    core = SimpleMLP([2*16+8, 64, latent], "tanh", "Glorot normal")
    trunk = LDCFourierTrunkNet(num_features=16, sigma=5.0, core_net=core)
    net = LDCDeepONet(branch_net=branch, trunk_net=trunk)
    b_in = torch.randn(3, 301)
    t_in = torch.randn(15, 3)
    t_in[:, 2] = torch.tensor([0,1,2]*5).float()
    loss = net(b_in, t_in).mean()
    loss.backward()
    # Check at least one parameter has a gradient
    assert any(p.grad is not None for p in net.parameters())


def test_create_ldc_model_resnet():
    from pi_onet.ldc_train import create_ldc_model
    net = create_ldc_model(
        branch_dim=301,
        branch_hidden_dims=[128, 128],
        trunk_hidden_dims=[64, 64],
        latent_width=64,
        trunk_rff_features=32,
        trunk_rff_sigma=5.0,
        use_resnet_branch=True,
    )
    import torch
    b = torch.randn(3, 301)
    t = torch.randn(30, 3)
    t[:, 2] = torch.tensor([0,1,2]*10).float()
    assert net(b, t).shape == (3, 30)


def test_create_ldc_model_mlp():
    from pi_onet.ldc_train import create_ldc_model
    net = create_ldc_model(
        branch_dim=301,
        branch_hidden_dims=[64],
        trunk_hidden_dims=[64],
        latent_width=32,
        trunk_rff_features=16,
        trunk_rff_sigma=5.0,
        use_resnet_branch=False,
    )
    import torch
    b = torch.randn(3, 301)
    t = torch.randn(9, 3)
    t[:, 2] = torch.tensor([0,1,2]*3).float()
    assert net(b, t).shape == (3, 9)


# ── Task 3 tests ─────────────────────────────────────────────────────────────

def test_steady_ns_residual_zero_for_linear_flow():
    """A linear flow u=y, v=0, p=0 satisfies steady NS exactly at Re=inf (zero viscous term too)."""
    from pi_onet.ldc_train import steady_ns_residuals
    import torch

    n = 20
    xy = torch.rand(n, 2, requires_grad=True)

    def u_fn(xy): return xy[:, 1:2]          # u = y
    def v_fn(xy): return torch.zeros(n, 1)
    def p_fn(xy): return torch.zeros(n, 1)

    ns_x, ns_y, cont = steady_ns_residuals(u_fn, v_fn, p_fn, xy, re=1e8)
    assert ns_x.abs().mean() < 1e-4
    assert ns_y.abs().mean() < 1e-4
    assert cont.abs().mean() < 1e-4


def test_steady_ns_residual_nonzero_for_bad_flow():
    from pi_onet.ldc_train import steady_ns_residuals
    import torch

    n = 10
    xy = torch.rand(n, 2, requires_grad=True)

    # Random flow has nonzero residuals
    def u_fn(xy): return xy[:, 0:1] ** 2
    def v_fn(xy): return xy[:, 1:2] ** 2
    def p_fn(xy): return xy[:, 0:1] * xy[:, 1:2]

    ns_x, ns_y, cont = steady_ns_residuals(u_fn, v_fn, p_fn, xy, re=100.0)
    assert ns_x.abs().mean() > 1e-6 or cont.abs().mean() > 1e-6


def test_bc_loss_zero_for_exact_bcs():
    from pi_onet.ldc_train import compute_bc_loss
    import torch

    # Exact BCs: u=1 at top, u=0 elsewhere, v=0 everywhere
    def model_fn(xy, c):
        # c=0: u, c=1: v, c=2: p
        if c == 0:
            return torch.where(xy[:, 1:2] >= 0.999, torch.ones(len(xy), 1), torch.zeros(len(xy), 1))
        return torch.zeros(len(xy), 1)

    loss = compute_bc_loss(model_fn=model_fn, n_per_wall=5, device=torch.device("cpu"))
    assert loss.item() < 1e-6


def test_gauge_loss_zero_at_origin():
    from pi_onet.ldc_train import compute_gauge_loss
    import torch

    def model_fn(xy, c):
        return torch.zeros(len(xy), 1)  # p=0 everywhere

    loss = compute_gauge_loss(model_fn=model_fn, device=torch.device("cpu"))
    assert loss.item() < 1e-8


# ── Task 4 tests ─────────────────────────────────────────────────────────────

def test_training_smoke(tmp_path):
    """What: Training loop runs 3 steps without error and produces a checkpoint."""
    import subprocess, sys
    # Create 3 fake .mat files
    for re in [3000, 4000, 5000]:
        make_fake_mat(tmp_path, re)

    config_text = f"""
[train]
data_files = [
  "{tmp_path}/cavity_Re3000_5_Uniform.mat",
  "{tmp_path}/cavity_Re4000_5_Uniform.mat",
  "{tmp_path}/cavity_Re5000_5_Uniform.mat",
]
num_interior_sensors = 3
num_boundary_sensors = 4
num_trunk_points = 6
num_physics_points = 4
num_bc_points = 4
branch_hidden_dims = [16, 16]
trunk_hidden_dims = [16, 16]
trunk_rff_features = 8
trunk_rff_sigma = 5.0
latent_width = 16
use_resnet_branch = false
data_loss_weight = 1.0
physics_loss_weight = 0.01
physics_continuity_weight = 1.0
bc_loss_weight = 1.0
gauge_loss_weight = 1.0
iterations = 3
batch_size = 3
optimizer = "adamw"
learning_rate = 0.001
weight_decay = 0.0001
lr_schedule = "none"
min_learning_rate = 1e-6
lr_step_size = 1000
lr_step_gamma = 0.5
checkpoint_period = 2
seed = 42
device = "cpu"
artifacts_dir = "{tmp_path}/artifacts"
"""
    config_path = tmp_path / "test_config.toml"
    config_path.write_text(config_text)

    result = subprocess.run(
        [sys.executable, "-m", "pi_onet.ldc_train", "--config", str(config_path)],
        capture_output=True, text=True, timeout=60,
        cwd="/Users/latteine/Documents/coding/pi-o-net",
    )
    assert result.returncode == 0, f"Training failed:\n{result.stderr}"
    # Checkpoint should exist
    checkpoints = list(Path(f"{tmp_path}/artifacts/checkpoints").glob("*.pt"))
    assert len(checkpoints) > 0
