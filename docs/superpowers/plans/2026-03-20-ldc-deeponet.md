# LDC Multi-Re PI-DeepONet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PI-DeepONet for steady-state multi-Re Lid-Driven Cavity flow with two new files (`ldc_dataset.py`, `ldc_train.py`), a config, and full tests.

**Architecture:** `LDCDeepONet` holds `branch_net` (ResNetBranchNet or SimpleMLP) and `LDCFourierTrunkNet` (RFF on (x,y) + Embedding for c); forward is `branch @ trunk.T + bias`. Training loop is pure PyTorch (no DeepXDE training infra) since LDC is steady-state with custom physics autodiff.

**Tech Stack:** PyTorch, scipy.io (for .mat), deepxde (for dde_config.real only), tomllib, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/pi_onet/ldc_dataset.py` | Create | Load .mat, sample sensors, build branch/trunk/ref tensors |
| `src/pi_onet/ldc_train.py` | Create | LDCFourierTrunkNet, LDCDeepONet, create_ldc_model, physics loss, training loop, CLI |
| `tests/test_ldc.py` | Create | Unit and integration tests |
| `configs/ldc_re3000_5000.toml` | Create | Training config for Re 3000/4000/5000 |

**Imports from `train.py`** (read-only, no modification):
```python
from pi_onet.train import ResNetBranchNet, SimpleMLP, configure_torch_runtime
```

---

## Context for Implementers

**Data format**: Each `.mat` file has `X_ref`, `Y_ref`, `U_ref`, `V_ref`, `P_ref` of shape `(257, 257)`. Domain is `[0,1]²`. After `.ravel()`, flat index = `row * 257 + col`.

**Grid boundary indices**:
- Interior pool: rows `[1:256]` × cols `[1:256]` → flat index `= row*257 + col`, pool size = 255×255 = 65,025
- Top wall (row=256): flat indices `256*257 + col`, col ∈ [0,256]
- Bottom wall (row=0): flat indices `0*257 + col`
- Left wall (col=0): flat indices `row*257 + 0`
- Right wall (col=256): flat indices `row*257 + 256`

**Branch vector layout** (dim=301):
- `[0]`: `Re_norm = (Re - 4000) / 816.5`
- `[1:241]`: interior sensors u×80, v×80, p×80
- `[241:301]`: boundary sensors u×20, v×20, p×20

**DeepONet forward**: `output[i, j] = branch[i] · trunk[j] + bias` where `i` = Re case, `j` = trunk point.

**Physics autodiff pattern** (from existing `train.py`):
```python
# requires_grad on xy, not on c column
xy = trunk_phys[:, :2]   # [N, 2], requires_grad=True
def grad(y, x): return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
```

**dde_config.real(torch)**: Returns `torch.float32` by default; used for all Linear layers and buffers for dtype consistency.

---

## Task 1: `ldc_dataset.py` — Data Loading and Sensor Sampling

**Files:**
- Create: `src/pi_onet/ldc_dataset.py`
- Test: `tests/test_ldc.py` (section 1)

- [ ] **Step 1: Write failing tests for data loading**

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
cd /Users/latteine/Documents/coding/pi-o-net
uv run pytest tests/test_ldc.py -k "Task 1 or test_load or test_sample or test_ldc_dataset" -v 2>&1 | head -40
```
Expected: FAIL with `ModuleNotFoundError: No module named 'pi_onet.ldc_dataset'`

- [ ] **Step 3: Implement `ldc_dataset.py`**

```python
# src/pi_onet/ldc_dataset.py
"""LDC multi-Re dataset loader.

What: Load cavity flow .mat files, sample sensor locations, build branch/trunk tensors.
Why: Steady-state LDC has no temporal dimension; structure is simpler than Kolmogorov pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.io

RE_MEAN: float = 4000.0
RE_STD: float = 816.5  # population std of {3000, 4000, 5000}; update if Re set changes


def load_ldc_mat(path: Path) -> dict[str, np.ndarray]:
    """What: Load a single LDC .mat file; return flattened 1-D field arrays."""
    data = scipy.io.loadmat(str(path))
    return {
        "x": data["X_ref"].ravel().astype(np.float64),
        "y": data["Y_ref"].ravel().astype(np.float64),
        "u": data["U_ref"].ravel().astype(np.float64),
        "v": data["V_ref"].ravel().astype(np.float64),
        "p": data["P_ref"].ravel().astype(np.float64),
    }


def sample_interior_indices(
    rng: np.random.Generator, grid_size: int, n: int
) -> np.ndarray:
    """What: Sample n flat indices from interior rows/cols [1:grid_size-1].

    Why: Exclude boundary rows/columns so interior sensors do not overlap with
         boundary sensors that have explicit BC meaning.
    """
    inner = grid_size - 2  # e.g. 255 for grid_size=257
    rows = np.repeat(np.arange(1, grid_size - 1), inner)  # [inner*inner]
    cols = np.tile(np.arange(1, grid_size - 1), inner)
    pool = rows * grid_size + cols  # flat indices
    chosen = rng.choice(pool, size=n, replace=False)
    return chosen


def sample_boundary_indices(grid_size: int, n_per_wall: int) -> np.ndarray:
    """What: Return n_per_wall uniformly spaced flat indices per wall (4 walls).

    Why: Boundary sensors give the branch information about how well BCs are satisfied,
         complementing interior sensors that observe the flow structure.
    """
    pts = np.linspace(0, grid_size - 1, n_per_wall, dtype=int)
    top    = (grid_size - 1) * grid_size + pts   # row = grid_size-1
    bottom = 0 * grid_size + pts                  # row = 0
    left   = pts * grid_size + 0                  # col = 0
    right  = pts * grid_size + (grid_size - 1)    # col = grid_size-1
    return np.concatenate([top, bottom, left, right])


@dataclass
class LDCDataset:
    """What: Holds all Re cases and exposes branch/trunk/ref tensors for training.

    Why: Encapsulates sensor sampling and train/val split so ldc_train.py stays clean.

    Attributes:
        branch_all:   [num_re, branch_dim]  — one branch vector per Re case
        re_values:    [num_re]              — raw Re values (not normalised)
        train_indices: [num_re, n_train]    — flat grid indices in train pool per Re
        val_indices:   [num_re, n_val]      — flat grid indices in val pool per Re
        grids:         list of dicts        — raw x/y/u/v/p arrays per Re case
        sensor_interior: [n_interior]       — flat sensor indices (shared)
        sensor_boundary: [n_boundary*4]     — flat boundary sensor indices (shared)
    """

    branch_all: np.ndarray
    re_values: np.ndarray
    train_indices: list[np.ndarray]
    val_indices: list[np.ndarray]
    grids: list[dict[str, np.ndarray]]
    sensor_interior: np.ndarray
    sensor_boundary: np.ndarray

    def __init__(
        self,
        mat_paths: Sequence[Path | str],
        num_interior_sensors: int,
        num_boundary_sensors: int,
        train_ratio: float,
        seed: int,
        re_list: Sequence[float] | None = None,
    ) -> None:
        """What: Load all .mat files, sample sensors, build branch vectors, split data."""
        rng = np.random.default_rng(seed)
        grids = [load_ldc_mat(Path(p)) for p in mat_paths]
        n_pts = len(grids[0]["x"])
        grid_size = int(round(n_pts ** 0.5))

        # Sensor locations (shared across all Re)
        self.sensor_interior = sample_interior_indices(rng, grid_size, num_interior_sensors)
        n_per_wall = num_boundary_sensors // 4
        self.sensor_boundary = sample_boundary_indices(grid_size, n_per_wall)
        self.grids = grids

        # Re values
        if re_list is not None:
            re_arr = np.array(re_list, dtype=np.float64)
        else:
            # Infer Re from filenames: cavity_Re<N>_*.mat
            re_arr = np.array([
                float(Path(p).stem.split("_Re")[1].split("_")[0])
                for p in mat_paths
            ], dtype=np.float64)
        self.re_values = re_arr

        # Build branch vectors: [num_re, 1 + n_int*3 + n_bnd*3]
        branch_list = []
        for i, grid in enumerate(grids):
            re_norm = (re_arr[i] - RE_MEAN) / RE_STD
            u_int = grid["u"][self.sensor_interior]
            v_int = grid["v"][self.sensor_interior]
            p_int = grid["p"][self.sensor_interior]
            u_bnd = grid["u"][self.sensor_boundary]
            v_bnd = grid["v"][self.sensor_boundary]
            p_bnd = grid["p"][self.sensor_boundary]
            vec = np.concatenate([[re_norm], u_int, v_int, p_int, u_bnd, v_bnd, p_bnd])
            branch_list.append(vec)
        self.branch_all = np.stack(branch_list, axis=0).astype(np.float32)

        # Train / val split per Re (same random split for all Re)
        all_idx = np.arange(n_pts)
        rng.shuffle(all_idx)
        n_train = int(len(all_idx) * train_ratio)
        train_idx = all_idx[:n_train]
        val_idx = all_idx[n_train:]
        self.train_indices = [train_idx for _ in grids]
        self.val_indices = [val_idx for _ in grids]

    def sample_train_trunk(
        self, rng: np.random.Generator, n_per_re: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """What: Sample n_per_re grid points per Re from train pool; replicate for c=0,1,2.

        Returns:
            trunk_pts: [n_per_re * 3, 3]  — (x, y, c) rows, c cycles 0/1/2
            ref_vals:  [num_re, n_per_re * 3]  — reference u/v/p at each point per Re
        """
        # Sample the same spatial indices for all Re (simplifies trunk batching)
        idx = rng.choice(self.train_indices[0], size=n_per_re, replace=False)
        grid0 = self.grids[0]
        x_pts = grid0["x"][idx].astype(np.float32)
        y_pts = grid0["y"][idx].astype(np.float32)

        # Replicate each point 3 times for c = 0, 1, 2
        x_rep = np.repeat(x_pts, 3)
        y_rep = np.repeat(y_pts, 3)
        c_rep = np.tile([0, 1, 2], n_per_re).astype(np.float32)
        trunk_pts = np.stack([x_rep, y_rep, c_rep], axis=1)  # [n_per_re*3, 3]

        # Reference values per Re
        ref_list = []
        for i, grid in enumerate(self.grids):
            u_vals = grid["u"][idx].astype(np.float32)
            v_vals = grid["v"][idx].astype(np.float32)
            p_vals = grid["p"][idx].astype(np.float32)
            # Interleave: u0,v0,p0, u1,v1,p1, ...
            ref = np.empty(n_per_re * 3, dtype=np.float32)
            ref[0::3] = u_vals
            ref[1::3] = v_vals
            ref[2::3] = p_vals
            ref_list.append(ref)
        ref_vals = np.stack(ref_list, axis=0)  # [num_re, n_per_re*3]

        return trunk_pts, ref_vals

    def get_val_trunk(self) -> tuple[np.ndarray, np.ndarray]:
        """What: Return full val set trunk and ref values for checkpoint evaluation."""
        return self.sample_val_trunk_all()

    def sample_val_trunk_all(self) -> tuple[np.ndarray, np.ndarray]:
        """What: Build trunk/ref tensors from the full val pool of Re case 0."""
        idx = self.val_indices[0]
        grid0 = self.grids[0]
        x_pts = grid0["x"][idx].astype(np.float32)
        y_pts = grid0["y"][idx].astype(np.float32)
        x_rep = np.repeat(x_pts, 3)
        y_rep = np.repeat(y_pts, 3)
        c_rep = np.tile([0, 1, 2], len(idx)).astype(np.float32)
        trunk_pts = np.stack([x_rep, y_rep, c_rep], axis=1)

        ref_list = []
        for grid in self.grids:
            u_vals = grid["u"][idx].astype(np.float32)
            v_vals = grid["v"][idx].astype(np.float32)
            p_vals = grid["p"][idx].astype(np.float32)
            ref = np.empty(len(idx) * 3, dtype=np.float32)
            ref[0::3] = u_vals
            ref[1::3] = v_vals
            ref[2::3] = p_vals
            ref_list.append(ref)
        return trunk_pts, np.stack(ref_list, axis=0)
```

- [ ] **Step 4: Run dataset tests — expect PASS**

```bash
uv run pytest tests/test_ldc.py -k "test_load or test_sample or test_ldc_dataset" -v
```
Expected: 7/7 PASS

- [ ] **Step 5: Commit**

```bash
git add src/pi_onet/ldc_dataset.py tests/test_ldc.py
git commit -m "feat: add ldc_dataset.py with sensor sampling and branch/trunk builders"
```

---

## Task 2: `LDCFourierTrunkNet` and `LDCDeepONet` Model

**Files:**
- Create: `src/pi_onet/ldc_train.py` (skeleton + model classes)
- Test: `tests/test_ldc.py` (section 2)

- [ ] **Step 1: Write failing model tests**

Add to `tests/test_ldc.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
uv run pytest tests/test_ldc.py -k "trunk or deeponet or create_ldc" -v 2>&1 | head -30
```
Expected: FAIL with `ModuleNotFoundError: No module named 'pi_onet.ldc_train'`

- [ ] **Step 3: Implement model classes in `ldc_train.py`**

```python
# src/pi_onet/ldc_train.py
"""Train PI-DeepONet on steady-state multi-Re Lid-Driven Cavity flow.

What:
    以 PI-DeepONet 訓練多 Re LDC 穩態問題（無時間維度）。
    Branch: Re_norm + 感測器讀值; Trunk: (x,y,c) with RFF + Embedding。
Why:
    作為 Kolmogorov 暫態 pipeline 的互補驗證基準，時間無關穩態問題。
"""
from __future__ import annotations

import json
import os
import time
import tomllib
from pathlib import Path
from typing import Any

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from deepxde import config as dde_config

from pi_onet.train import ResNetBranchNet, SimpleMLP, configure_torch_runtime


# ── Default config ────────────────────────────────────────────────────────────

DEFAULT_LDC_ARGS: dict[str, Any] = {
    "data_files": None,
    "num_interior_sensors": 80,
    "num_boundary_sensors": 20,
    "num_trunk_points": 2048,
    "num_physics_points": 1024,
    "num_bc_points": 100,
    "branch_hidden_dims": [256, 256],
    "trunk_hidden_dims": [256, 256],
    "trunk_rff_features": 128,
    "trunk_rff_sigma": 5.0,
    "latent_width": 128,
    "use_resnet_branch": True,
    "data_loss_weight": 1.0,
    "physics_loss_weight": 0.1,
    "physics_continuity_weight": 1.0,
    "bc_loss_weight": 1.0,
    "gauge_loss_weight": 1.0,
    "iterations": 10000,
    "batch_size": 3,
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_schedule": "step",
    "lr_step_size": 5000,
    "lr_step_gamma": 0.5,
    "min_learning_rate": 1e-6,
    "checkpoint_period": 2000,
    "seed": 42,
    "device": "auto",
    "artifacts_dir": "../artifacts/ldc-resnet-rff",
}


# ── Model components ──────────────────────────────────────────────────────────

class LDCFourierTrunkNet(torch.nn.Module):
    """What: RFF trunk encoding (x,y) + Embedding for component index c.

    Why: Steady LDC has only 2 spatial dimensions (no time).
         B shape is [2, num_features] — NOT [3, num_features] like FourierFeatureTrunkNet.
         Component index c treated separately via Embedding to avoid scale mismatch.
    """

    def __init__(self, num_features: int, sigma: float, core_net: torch.nn.Module) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features 必須為正整數。")
        if sigma <= 0.0:
            raise ValueError("sigma 必須為正數。")
        self.num_features = int(num_features)
        self.core_net = core_net

        self.component_embedding = torch.nn.Embedding(3, 8)
        torch.nn.init.normal_(self.component_embedding.weight, mean=0.0, std=0.1)

        # B shape [2, num_features] — 2D spatial only
        B = torch.randn(2, self.num_features, dtype=dde_config.real(torch)) * float(sigma)
        self.register_buffer("B", B)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """What: Encode (x,y) via RFF, embed c, concatenate, pass to core_net."""
        xy = inputs[:, :2]                             # [N, 2]
        c_idx = inputs[:, 2].long()                    # [N]
        proj = 2.0 * np.pi * (xy @ self.B.to(dtype=inputs.dtype))  # [N, num_features]
        rff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)  # [N, 2*num_features]
        c_emb = self.component_embedding(c_idx).to(dtype=inputs.dtype)  # [N, 8]
        encoded = torch.cat([rff, c_emb], dim=1)       # [N, 2*num_features + 8]
        return self.core_net(encoded)


class LDCDeepONet(torch.nn.Module):
    """What: DeepONet for steady LDC; output[i,j] = branch[i] · trunk[j] + bias.

    Why: Exposes trunk_net directly for physics autodiff without re-routing through
         branch computation. Bias is a learnable scalar following standard DeepONet.
    """

    def __init__(self, branch_net: torch.nn.Module, trunk_net: torch.nn.Module) -> None:
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.bias = torch.nn.Parameter(
            torch.zeros(1, dtype=dde_config.real(torch))
        )

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        """What: Forward pass; branch_input [batch_Re, 301], trunk_input [N_pts, 3]."""
        b = self.branch_net(branch_input)   # [batch_Re, latent_width]
        t = self.trunk_net(trunk_input)     # [N_pts, latent_width]
        return b @ t.T + self.bias          # [batch_Re, N_pts]


def create_ldc_model(
    branch_dim: int,
    branch_hidden_dims: list[int],
    trunk_hidden_dims: list[int],
    latent_width: int,
    trunk_rff_features: int,
    trunk_rff_sigma: float,
    use_resnet_branch: bool = True,
) -> LDCDeepONet:
    """What: Build LDCDeepONet from config parameters.

    Why: Decouples model construction from training loop; enables testing without I/O.
    """
    if use_resnet_branch:
        branch_net = ResNetBranchNet(
            flat_dim=branch_dim,
            hidden_dims=branch_hidden_dims,
            latent_width=latent_width,
        )
    else:
        branch_net = SimpleMLP(
            layer_sizes=[branch_dim, *branch_hidden_dims, latent_width],
            activation="tanh",
            kernel_initializer="Glorot normal",
        )

    core_net = SimpleMLP(
        layer_sizes=[2 * trunk_rff_features + 8, *trunk_hidden_dims, latent_width],
        activation="tanh",
        kernel_initializer="Glorot normal",
    )
    trunk_net = LDCFourierTrunkNet(
        num_features=trunk_rff_features,
        sigma=trunk_rff_sigma,
        core_net=core_net,
    )
    return LDCDeepONet(branch_net=branch_net, trunk_net=trunk_net)
```

- [ ] **Step 4: Run model tests — expect PASS**

```bash
uv run pytest tests/test_ldc.py -k "trunk or deeponet or create_ldc" -v
```
Expected: 8/8 PASS

- [ ] **Step 5: Commit**

```bash
git add src/pi_onet/ldc_train.py tests/test_ldc.py
git commit -m "feat: add LDCFourierTrunkNet, LDCDeepONet, create_ldc_model"
```

---

## Task 3: Physics Loss Functions

**Files:**
- Modify: `src/pi_onet/ldc_train.py` (add physics loss functions)
- Test: `tests/test_ldc.py` (section 3)

- [ ] **Step 1: Write failing physics loss tests**

Add to `tests/test_ldc.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError or AttributeError**

```bash
uv run pytest tests/test_ldc.py -k "residual or bc_loss or gauge" -v 2>&1 | head -20
```
Expected: FAIL with `ImportError` for `steady_ns_residuals`

- [ ] **Step 3: Implement physics loss functions in `ldc_train.py`**

Add after `create_ldc_model()`:

```python
# ── Physics loss ──────────────────────────────────────────────────────────────

def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """What: First-order partial derivative dy/dx via autograd."""
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def steady_ns_residuals(
    u_fn,
    v_fn,
    p_fn,
    xy: torch.Tensor,
    re: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What: Compute steady NS and continuity residuals at collocation points.

    Why: Steady incompressible NS — no time derivative; Re enters as viscous coefficient.

    Args:
        u_fn, v_fn, p_fn: callables (xy) -> [N, 1], xy has requires_grad=True
        xy: [N, 2] collocation points with requires_grad=True
        re: Reynolds number scalar

    Returns:
        ns_x:  [N, 1]  momentum x residual
        ns_y:  [N, 1]  momentum y residual
        cont:  [N, 1]  continuity residual
    """
    u = u_fn(xy)   # [N, 1]
    v = v_fn(xy)   # [N, 1]
    p = p_fn(xy)   # [N, 1]

    u_xy = _grad(u, xy)            # [N, 2]: [du/dx, du/dy]
    v_xy = _grad(v, xy)            # [N, 2]: [dv/dx, dv/dy]
    p_xy = _grad(p, xy)            # [N, 2]: [dp/dx, dp/dy]

    du_dx, du_dy = u_xy[:, 0:1], u_xy[:, 1:2]
    dv_dx, dv_dy = v_xy[:, 0:1], v_xy[:, 1:2]
    dp_dx, dp_dy = p_xy[:, 0:1], p_xy[:, 1:2]

    # Second derivatives
    du_dx2 = _grad(du_dx, xy)[:, 0:1]   # d²u/dx²
    du_dy2 = _grad(du_dy, xy)[:, 1:2]   # d²u/dy²
    dv_dx2 = _grad(dv_dx, xy)[:, 0:1]   # d²v/dx²
    dv_dy2 = _grad(dv_dy, xy)[:, 1:2]   # d²v/dy²

    nu = 1.0 / float(re)
    ns_x = u * du_dx + v * du_dy + dp_dx - nu * (du_dx2 + du_dy2)
    ns_y = u * dv_dx + v * dv_dy + dp_dy - nu * (dv_dx2 + dv_dy2)
    cont = du_dx + dv_dy

    return ns_x, ns_y, cont


def compute_bc_loss(
    model_fn,
    n_per_wall: int,
    device: torch.device,
) -> torch.Tensor:
    """What: Compute Dirichlet BC loss for u and v on all 4 walls.

    Why: LDC BCs: top (y=1) u=1 v=0; others u=0 v=0. Pressure excluded — no wall pressure BC.
    n_per_wall: number of points per wall (total num_bc_points // 4).
    """
    t = torch.linspace(0.0, 1.0, n_per_wall, device=device, dtype=dde_config.real(torch))
    ones = torch.ones(n_per_wall, device=device, dtype=dde_config.real(torch))
    zeros = torch.zeros(n_per_wall, device=device, dtype=dde_config.real(torch))

    total_loss = torch.zeros(1, device=device, dtype=dde_config.real(torch))
    walls = [
        # (xy_tensor, u_target, v_target)
        (torch.stack([t, ones], dim=1), ones, zeros),       # top: u=1, v=0
        (torch.stack([t, zeros], dim=1), zeros, zeros),     # bottom: u=0, v=0
        (torch.stack([zeros, t], dim=1), zeros, zeros),     # left: u=0, v=0
        (torch.stack([ones, t], dim=1), zeros, zeros),      # right: u=0, v=0
    ]
    for xy_wall, u_target, v_target in walls:
        u_pred = model_fn(xy_wall, c=0).squeeze(1)
        v_pred = model_fn(xy_wall, c=1).squeeze(1)
        total_loss = total_loss + torch.mean((u_pred - u_target) ** 2)
        total_loss = total_loss + torch.mean((v_pred - v_target) ** 2)
    return total_loss


def compute_gauge_loss(model_fn, device: torch.device) -> torch.Tensor:
    """What: Penalise p at bottom-left corner (x=0,y=0) to be 0 (pressure gauge fix).

    Why: Steady NS has pressure determined only up to additive constant; gauge pins it.
    """
    corner = torch.zeros(1, 2, device=device, dtype=dde_config.real(torch))
    p_corner = model_fn(corner, c=2).squeeze()
    return p_corner ** 2
```

- [ ] **Step 4: Run physics tests — expect PASS**

```bash
uv run pytest tests/test_ldc.py -k "residual or bc_loss or gauge" -v
```
Expected: 4/4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/pi_onet/ldc_train.py tests/test_ldc.py
git commit -m "feat: add steady NS residuals, BC loss, and pressure gauge loss"
```

---

## Task 4: Training Loop and CLI

**Files:**
- Modify: `src/pi_onet/ldc_train.py` (add training loop, config loader, main)
- Test: `tests/test_ldc.py` (smoke test)

- [ ] **Step 1: Write smoke test**

Add to `tests/test_ldc.py`:

```python
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
```

- [ ] **Step 2: Run smoke test — expect FAIL**

```bash
uv run pytest tests/test_ldc.py::test_training_smoke -v 2>&1 | head -20
```
Expected: FAIL (training loop not yet implemented)

- [ ] **Step 3: Implement training loop and CLI in `ldc_train.py`**

Add after the physics loss functions:

```python
# ── Utilities ─────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    """What: Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_json(path: Path, data: dict) -> None:
    """What: Write a dict as formatted JSON."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_ldc_config(config_path: Path | None) -> dict[str, Any]:
    """What: Load and validate TOML config against DEFAULT_LDC_ARGS."""
    if config_path is None:
        return {}
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config_data = payload.get("train", payload)
    normalized = dict(config_data)
    unknown = sorted(set(normalized) - set(DEFAULT_LDC_ARGS))
    if unknown:
        raise ValueError(f"LDC config 含有不支援的欄位: {unknown}")
    # Resolve data_files and artifacts_dir relative to config location
    if "data_files" in normalized:
        normalized["data_files"] = [
            str((config_path.parent / Path(p)).resolve())
            for p in normalized["data_files"]
        ]
    if "artifacts_dir" in normalized:
        normalized["artifacts_dir"] = str(
            (config_path.parent / Path(normalized["artifacts_dir"])).resolve()
        )
    return normalized


def make_model_fn(net: LDCDeepONet, branch_tensor: torch.Tensor, device: torch.device):
    """What: Return a closure (xy, c) -> [N,1] querying trunk for one Re case.

    Why: Physics/BC/gauge losses need to query the model for a single Re case
         at arbitrary (x,y) points with a specific component c.
         branch_tensor: [1, 301] (one Re case, already on device).
    """
    def fn(xy: torch.Tensor, c: int) -> torch.Tensor:
        n = len(xy)
        c_col = torch.full((n, 1), float(c), device=device, dtype=xy.dtype)
        trunk_input = torch.cat([xy, c_col], dim=1)   # [N, 3]
        out = net(branch_tensor, trunk_input)          # [1, N]
        return out.T                                   # [N, 1]
    return fn


# ── Training loop ─────────────────────────────────────────────────────────────

def train_ldc(args: dict[str, Any]) -> None:
    """What: Full LDC training loop with data loss, physics loss, BC loss, gauge loss.

    Why: Pure PyTorch loop (no DeepXDE training infra) — steady-state LDC has
         no temporal structure compatible with DeepXDE's DataSet abstraction.
    """
    from pi_onet.ldc_dataset import LDCDataset

    device = configure_torch_runtime(args["device"])
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    rng = np.random.default_rng(args["seed"])

    artifacts_dir = Path(args["artifacts_dir"])
    checkpoints_dir = artifacts_dir / "checkpoints"
    best_dir = artifacts_dir / "best_validation"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = LDCDataset(
        mat_paths=args["data_files"],
        num_interior_sensors=args["num_interior_sensors"],
        num_boundary_sensors=args["num_boundary_sensors"],
        train_ratio=0.8,
        seed=args["seed"],
    )
    branch_all = torch.tensor(dataset.branch_all, device=device)  # [num_re, 301]
    re_values = dataset.re_values.tolist()
    num_re = len(re_values)

    # Model
    branch_dim = dataset.branch_all.shape[1]
    net = create_ldc_model(
        branch_dim=branch_dim,
        branch_hidden_dims=args["branch_hidden_dims"],
        trunk_hidden_dims=args["trunk_hidden_dims"],
        latent_width=args["latent_width"],
        trunk_rff_features=args["trunk_rff_features"],
        trunk_rff_sigma=args["trunk_rff_sigma"],
        use_resnet_branch=args["use_resnet_branch"],
    ).to(device)

    print("=== Configuration ===")
    print(json.dumps({k: v for k, v in args.items() if k != "data_files"}, indent=2, ensure_ascii=False))
    print(f"trainable_parameters: {count_parameters(net)}")

    # Optimizer
    if args["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args["learning_rate"])

    # LR scheduler
    scheduler = None
    if args["lr_schedule"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args["lr_step_size"],
            gamma=args["lr_step_gamma"],
        )

    # Training
    best_val_metric = float("inf")
    best_checkpoint_path: str | None = None
    n_bc_per_wall = max(1, args["num_bc_points"] // 4)

    print("=== Training ===")
    print(f"{'Step':<8} {'L_data':>12} {'L_phys':>12} {'L_bc':>10} {'L_total':>12}")

    for step in range(1, args["iterations"] + 1):
        net.train()
        optimizer.zero_grad()

        # ── Data loss ──
        trunk_np, ref_np = dataset.sample_train_trunk(rng=rng, n_per_re=args["num_trunk_points"])
        trunk_t = torch.tensor(trunk_np, device=device)
        ref_t = torch.tensor(ref_np, device=device)
        pred = net(branch_all, trunk_t)          # [num_re, num_trunk_points*3]
        l_data = torch.mean((pred - ref_t) ** 2)

        # ── Physics loss (shared collocation pts across Re) ──
        xy_phys_np = np.random.uniform(0.0, 1.0, (args["num_physics_points"], 2)).astype(np.float32)
        xy_phys = torch.tensor(xy_phys_np, device=device, requires_grad=True)

        l_ns_total = torch.zeros(1, device=device, dtype=dde_config.real(torch))
        l_cont_total = torch.zeros(1, device=device, dtype=dde_config.real(torch))
        for i in range(num_re):
            branch_i = branch_all[i:i+1]          # [1, 301]
            model_fn = make_model_fn(net, branch_i, device)
            def u_fn(xy, i=i): return make_model_fn(net, branch_all[i:i+1], device)(xy, 0)
            def v_fn(xy, i=i): return make_model_fn(net, branch_all[i:i+1], device)(xy, 1)
            def p_fn(xy, i=i): return make_model_fn(net, branch_all[i:i+1], device)(xy, 2)
            ns_x, ns_y, cont = steady_ns_residuals(u_fn, v_fn, p_fn, xy_phys, re=re_values[i])
            l_ns_total = l_ns_total + torch.mean(ns_x**2) + torch.mean(ns_y**2)
            l_cont_total = l_cont_total + torch.mean(cont**2)
        l_ns_total = l_ns_total / num_re
        l_cont_total = l_cont_total / num_re
        l_physics = l_ns_total + args["physics_continuity_weight"] * l_cont_total

        # ── BC loss (averaged across Re for model_fn; BCs are Re-independent) ──
        # Use first Re case to evaluate BCs (BCs don't depend on Re)
        model_fn_re0 = make_model_fn(net, branch_all[0:1], device)
        l_bc = compute_bc_loss(model_fn=model_fn_re0, n_per_wall=n_bc_per_wall, device=device)
        l_gauge = compute_gauge_loss(model_fn=model_fn_re0, device=device)

        # ── Total loss ──
        l_total = (
            args["data_loss_weight"]    * l_data
            + args["physics_loss_weight"] * l_physics
            + args["bc_loss_weight"]      * l_bc
            + args["gauge_loss_weight"]   * l_gauge
        )
        l_total.backward()
        optimizer.step()
        if scheduler is not None:
            # Clamp LR to min_learning_rate
            scheduler.step()
            for pg in optimizer.param_groups:
                if pg["lr"] < args["min_learning_rate"]:
                    pg["lr"] = args["min_learning_rate"]

        if step % max(1, args["iterations"] // 10) == 0 or step == 1:
            print(f"{step:<8} {l_data.item():>12.4e} {l_physics.item():>12.4e} {l_bc.item():>10.4e} {l_total.item():>12.4e}")

        # ── Checkpoint and validation ──
        if args["checkpoint_period"] > 0 and step % args["checkpoint_period"] == 0:
            ckpt_path = checkpoints_dir / f"ldc_deeponet_step_{step}.pt"
            torch.save(net.state_dict(), str(ckpt_path))

            net.eval()
            with torch.no_grad():
                val_trunk_np, val_ref_np = dataset.sample_val_trunk_all()
                val_trunk = torch.tensor(val_trunk_np, device=device)
                val_ref = torch.tensor(val_ref_np, device=device)
                val_pred = net(branch_all, val_trunk)
                # Mean relative L2 across all Re
                rel_l2 = torch.mean(
                    torch.norm(val_pred - val_ref, dim=1) / (torch.norm(val_ref, dim=1) + 1e-8)
                ).item()
            print(f"  [val @ {step}] mean_rel_l2 = {rel_l2:.4f}")

            if rel_l2 < best_val_metric:
                best_val_metric = rel_l2
                best_path = best_dir / "ldc_deeponet_best.pt"
                torch.save(net.state_dict(), str(best_path))
                best_checkpoint_path = str(best_path)
                write_json(best_dir / "best_validation_summary.json", {
                    "step": step, "val_mean_rel_l2": rel_l2,
                })

    # Final checkpoint
    final_path = artifacts_dir / "ldc_deeponet_final.pt"
    torch.save(net.state_dict(), str(final_path))

    write_json(artifacts_dir / "experiment_manifest.json", {
        "configuration": args,
        "best_checkpoint": best_checkpoint_path,
        "final_checkpoint": str(final_path),
        "final_val_metric": best_val_metric,
    })
    print(f"=== Done. Best val rel_L2 = {best_val_metric:.4f} ===")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """What: Entry point for ldc_train CLI."""
    import argparse
    parser = argparse.ArgumentParser(description="Train PI-DeepONet on steady-state LDC flow.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    cli_args = parser.parse_args()

    config = dict(DEFAULT_LDC_ARGS)
    config.update(load_ldc_config(cli_args.config))
    if cli_args.device is not None:
        config["device"] = cli_args.device

    train_ldc(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test — expect PASS**

```bash
uv run pytest tests/test_ldc.py::test_training_smoke -v
```
Expected: PASS (training runs 3 steps, checkpoint saved)

- [ ] **Step 5: Run all LDC tests**

```bash
uv run pytest tests/test_ldc.py -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/pi_onet/ldc_train.py tests/test_ldc.py
git commit -m "feat: add LDC training loop, config loader, and CLI entry point"
```

---

## Task 5: Config File

**Files:**
- Create: `configs/ldc_re3000_5000.toml`
- Modify: `pyproject.toml` (add `ldc_train` entry point)

- [ ] **Step 1: Create config**

```toml
# configs/ldc_re3000_5000.toml
[train]
data_files = [
  "../data/ldc/cavity_Re3000_256_Uniform.mat",
  "../data/ldc/cavity_Re4000_256_Uniform.mat",
  "../data/ldc/cavity_Re5000_256_Uniform.mat",
]
num_interior_sensors = 80
num_boundary_sensors = 20
num_trunk_points = 2048
num_physics_points = 1024
num_bc_points = 100
branch_hidden_dims = [256, 256]
trunk_hidden_dims = [256, 256]
trunk_rff_features = 128
trunk_rff_sigma = 5.0
latent_width = 128
use_resnet_branch = true
data_loss_weight = 1.0
physics_loss_weight = 0.1
physics_continuity_weight = 1.0
bc_loss_weight = 1.0
gauge_loss_weight = 1.0
iterations = 10000
batch_size = 3
optimizer = "adamw"
learning_rate = 0.001
weight_decay = 0.0001
lr_schedule = "step"
lr_step_size = 5000
lr_step_gamma = 0.5
min_learning_rate = 1e-6
checkpoint_period = 2000
seed = 42
device = "auto"
artifacts_dir = "../artifacts/ldc-resnet-rff"
```

- [ ] **Step 2: Add entry point to `pyproject.toml`**

Open `pyproject.toml` and add `ldc_train` to the `[project.scripts]` section:

```toml
[project.scripts]
pi_onet_train = "pi_onet.train:main"
pi_onet_ldc_train = "pi_onet.ldc_train:main"
```

Verify the existing entry point key — the new line must match the format already present.

- [ ] **Step 3: Reinstall and verify CLI**

```bash
uv sync
uv run python -m pi_onet.ldc_train --help
```
Expected: prints `usage: ldc_train.py [-h] [--config CONFIG] [--device ...]`

- [ ] **Step 4: Run full test suite to check no regressions**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -20
```
Expected: all existing tests still PASS; new LDC tests PASS

- [ ] **Step 5: Commit**

```bash
git add configs/ldc_re3000_5000.toml pyproject.toml
git commit -m "feat: add LDC config and cli entry point; all tests passing"
```

---

## Notes for Implementers

- **`make_model_fn` closure inside the training loop**: The closures `u_fn`, `v_fn`, `p_fn` inside the `for i in range(num_re)` loop must capture `i` by value using default arguments (`i=i`) to avoid Python late-binding bugs.
- **`requires_grad` on physics points**: `xy_phys` needs `requires_grad=True` before passing to `steady_ns_residuals`. Do not set `requires_grad` on the full trunk tensor — only on the spatial coordinate slice.
- **`dde_config.real(torch)`**: Returns `torch.float32` by default. All `torch.zeros`, `torch.linspace`, and buffer creation in LDC code must use this dtype for consistency with model parameters.
- **`batch_size` in config**: Currently unused (all 3 Re always processed together per step). The key is kept in config for forward compatibility.
