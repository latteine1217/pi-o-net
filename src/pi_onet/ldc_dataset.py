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
import scipy.interpolate

RE_MEAN: float = 4000.0
RE_STD: float = 816.5  # population std of {3000, 4000, 5000}; update if Re set changes


def _fill_nan_nearest(arr2d: np.ndarray) -> np.ndarray:
    """What: Fill NaN values with nearest-neighbour interpolation on a 2-D grid.

    Why: Some mat files have NaN pressure values near boundaries due to CFD solver
         corner divergence. Nearest-neighbour filling preserves physical scale without
         introducing artificial smoothing.
    """
    mask = np.isnan(arr2d)
    if not mask.any():
        return arr2d
    ny, nx = arr2d.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    valid = ~mask
    filled = scipy.interpolate.griddata(
        (ys[valid], xs[valid]), arr2d[valid], (ys, xs), method="nearest"
    )
    return filled


def load_ldc_mat(path: Path) -> dict[str, np.ndarray]:
    """What: Load a single LDC .mat file; return flattened 1-D field arrays.

    Why: NaN values in pressure fields (boundary artefacts) are filled with
         nearest-neighbour interpolation before ravel so downstream code sees
         clean float64 arrays.
    """
    data = scipy.io.loadmat(str(path))
    p_clean = _fill_nan_nearest(data["P_ref"].astype(np.float64))
    return {
        "x": data["X_ref"].ravel().astype(np.float64),
        "y": data["Y_ref"].ravel().astype(np.float64),
        "u": data["U_ref"].ravel().astype(np.float64),
        "v": data["V_ref"].ravel().astype(np.float64),
        "p": p_clean.ravel(),
    }


def _resize_grid(grid: dict[str, np.ndarray], target_size: int) -> dict[str, np.ndarray]:
    """What: Bilinear-interpolate all fields of a square grid to target_size × target_size.

    Why: Mat files for different Re may use different grid resolutions (e.g., Re=4000
         is 256×256 while Re=3000/5000 are 257×257). A shared flat_index space requires
         all grids to have the same ncols; otherwise the same flat_idx decodes to a
         different (row, col) → (x, y) across grids, corrupting sensor and trunk
         coordinates during training.

    Invariants preserved:
      - Domain stays [0, 1] × [0, 1].
      - Boundary values at x=0, x=1, y=0, y=1 are preserved because the target
        grid includes those exact endpoints (linspace).
      - Fields returned as float64 1-D arrays, row-major (flat_idx = row * target_size + col).
    """
    src_size = int(round(len(grid["x"]) ** 0.5))
    if src_size == target_size:
        return grid

    xs_src = np.unique(grid["x"])          # [src_size]
    ys_src = np.unique(grid["y"])          # [src_size]

    x_new = np.linspace(xs_src[0], xs_src[-1], target_size)
    y_new = np.linspace(ys_src[0], ys_src[-1], target_size)
    X2d, Y2d = np.meshgrid(x_new, y_new)  # both [target_size, target_size]

    result: dict[str, np.ndarray] = {
        "x": X2d.ravel().astype(np.float64),
        "y": Y2d.ravel().astype(np.float64),
    }
    query_pts = np.stack([Y2d.ravel(), X2d.ravel()], axis=1)  # (y, x) for RegularGridInterp
    for field in ("u", "v", "p"):
        F2d = grid[field].reshape(src_size, src_size)   # [ny_src, nx_src]
        interp = scipy.interpolate.RegularGridInterpolator(
            (ys_src, xs_src), F2d,
            method="linear", bounds_error=False, fill_value=None,
        )
        result[field] = interp(query_pts).astype(np.float64)
    return result


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
    """What: Return n_per_wall uniformly spaced flat indices per wall (4 walls), no duplicates.

    Why: Boundary sensors give the branch information about how well BCs are satisfied,
         complementing interior sensors that observe the flow structure.
         Left/right walls intentionally use only interior rows (1..grid_size-2) to avoid
         corner duplication; physical corners are represented exclusively in the top/bottom
         wall arrays, guaranteeing exactly 4*n_per_wall unique indices.
    """
    cols = np.linspace(0, grid_size - 1, n_per_wall, dtype=int)
    top    = (grid_size - 1) * grid_size + cols   # row = grid_size-1, all cols
    bottom = cols                                  # row = 0, all cols
    # Interior rows only: avoids corners already claimed by top/bottom
    inner_rows = np.linspace(1, grid_size - 2, n_per_wall, dtype=int)
    left  = inner_rows * grid_size + 0                  # col = 0
    right = inner_rows * grid_size + (grid_size - 1)    # col = grid_size-1
    return np.concatenate([top, bottom, left, right])


@dataclass
class LDCDataset:
    """What: Holds all Re cases and exposes branch/trunk/ref tensors for training.

    Why: Encapsulates sensor sampling and train/val split so ldc_train.py stays clean.

    Attributes:
        branch_all:      [num_re, branch_dim]  — one branch vector per Re case
        re_values:       [num_re]              — raw Re values (not normalised)
        train_idx:       [n_train]             — flat grid indices in train pool (shared across Re)
        val_idx:         [n_val]               — flat grid indices in val pool (shared across Re)
        grids:           list of dicts         — raw x/y/u/v/p arrays per Re case
        sensor_interior: [n_interior]          — flat sensor indices (shared)
        sensor_boundary: [n_boundary*4]        — flat boundary sensor indices (shared)
    """

    branch_all: np.ndarray
    re_values: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
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
        raw_grids = [load_ldc_mat(Path(p)) for p in mat_paths]
        # Align all grids to the largest grid_size via bilinear interpolation.
        # Why: mat files may have inconsistent resolutions (e.g., Re4000 is 256×256 while
        #      Re3000/Re5000 are 257×257). A flat_index has different (row, col) meanings
        #      under different ncols, so sharing sensor/trunk indices across grids requires
        #      a uniform coordinate frame. Upsampling the smaller grid preserves boundary
        #      values exactly (linspace endpoints match) and introduces negligible error
        #      (~O(h²) bilinear residual on a smooth laminar flow field).
        n_pts_max = max(len(g["x"]) for g in raw_grids)
        grid_size = int(round(n_pts_max ** 0.5))
        grids = [_resize_grid(g, grid_size) for g in raw_grids]

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
        n_pts = grid_size * grid_size
        all_idx = np.arange(n_pts)
        rng.shuffle(all_idx)
        n_train = int(len(all_idx) * train_ratio)
        train_idx = all_idx[:n_train]
        val_idx = all_idx[n_train:]
        self.train_idx = train_idx
        self.val_idx = val_idx

    def _build_trunk_and_ref(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """What: Build (x, y, c) trunk points and interleaved u/v/p ref values for given indices.

        Why: Shared trunk-building logic between train sampling and full-val evaluation;
             eliminates duplication and keeps each public method focused on index selection.
             All grids are aligned to the same resolution by __init__ (via _resize_grid),
             so grids[0]["x/y"] is identical to any other grid's x/y — using it for the
             shared trunk is safe and avoids per-Re coordinate divergence.

        Returns:
            trunk_pts: [len(idx)*3, 3]        — (x, y, c) rows, c cycles 0/1/2
            ref_vals:  [num_re, len(idx)*3]   — interleaved u/v/p per Re case
        """
        n = len(idx)
        grid0 = self.grids[0]
        x_rep = np.repeat(grid0["x"][idx].astype(np.float32), 3)
        y_rep = np.repeat(grid0["y"][idx].astype(np.float32), 3)
        c_rep = np.tile([0, 1, 2], n).astype(np.float32)
        trunk_pts = np.stack([x_rep, y_rep, c_rep], axis=1)

        ref_list = []
        for grid in self.grids:
            ref = np.empty(n * 3, dtype=np.float32)
            ref[0::3] = grid["u"][idx].astype(np.float32)
            ref[1::3] = grid["v"][idx].astype(np.float32)
            ref[2::3] = grid["p"][idx].astype(np.float32)
            ref_list.append(ref)
        return trunk_pts, np.stack(ref_list, axis=0)

    def sample_train_trunk(
        self, rng: np.random.Generator, n_per_re: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """What: Sample n_per_re grid points from the shared train pool; replicate for c=0,1,2.

        Returns:
            trunk_pts: [n_per_re * 3, 3]  — (x, y, c) rows, c cycles 0/1/2
            ref_vals:  [num_re, n_per_re * 3]  — reference u/v/p at each point per Re
        """
        idx = rng.choice(self.train_idx, size=n_per_re, replace=False)
        return self._build_trunk_and_ref(idx)

    def sample_val_trunk_all(self) -> tuple[np.ndarray, np.ndarray]:
        """What: Build trunk/ref tensors from the full shared val pool."""
        return self._build_trunk_and_ref(self.val_idx)
