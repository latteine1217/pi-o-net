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
