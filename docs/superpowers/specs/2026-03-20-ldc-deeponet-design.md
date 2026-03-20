# Architecture Design: LDC Multi-Re PI-DeepONet

**Date**: 2026-03-20
**Scope**: `src/pi_onet/ldc_dataset.py`, `src/pi_onet/ldc_train.py`
**Status**: Pending Review

---

## Motivation

Lid-Driven Cavity (LDC) flow is a canonical steady-state benchmark for validating PI-DeepONet on time-independent problems. Training on multiple Re values (3000, 4000, 5000) tests the operator's ability to generalize across Reynolds numbers using both sparse sensor data and steady NS physics constraints.

---

## New Files

| File | Responsibility |
|------|---------------|
| `src/pi_onet/ldc_dataset.py` | Load `.mat` files, sample sensors, build branch/trunk tensors |
| `src/pi_onet/ldc_train.py` | Model builder, steady NS physics loss, training loop, CLI |

**Shared from `train.py`** (import only, no modification):
- `ResNetBranchNet`, `SimpleMLP`

`FlattenBranchNet` is **not used** — dataset yields `[batch, 301]` directly.

---

## Data

### Source Files
```
data/ldc/cavity_Re3000_256_Uniform.mat
data/ldc/cavity_Re4000_256_Uniform.mat
data/ldc/cavity_Re5000_256_Uniform.mat
```
Each file contains: `X_ref`, `Y_ref`, `U_ref`, `V_ref`, `P_ref` — shape `(257, 257)`, domain `[0,1]²`.

### Sensor Layout (fixed across all Re, determined by seed)

- **80 interior sensors**: randomly sampled from interior grid points at indices `[1:256, 1:256]` (i.e., excluding boundary rows 0 and 256 and boundary columns 0 and 256), giving a pool of 255×255 = 65,025 points.
- **20 boundary sensors**: 5 points per wall (top, bottom, left, right), uniformly spaced along each edge.

Sensor positions are fixed at dataset construction time and shared across all Re cases.

### Branch Input

One branch vector per Re case, dim = `1 + 80×3 + 20×3 = 301`:

| Slice | Content | Size |
|-------|---------|------|
| `[0]` | `Re_norm` | 1 |
| `[1:241]` | interior sensors: `u_s1…u_s80, v_s1…v_s80, p_s1…p_s80` | 240 |
| `[241:301]` | boundary sensors: `u_b1…u_b20, v_b1…v_b20, p_b1…p_b20` | 60 |

Total: 1 + 240 + 60 = **301**.

Dataset yields branch tensor shape `[batch, 301]` directly (no extra history dimension).

Re normalization: `Re_norm = (Re - 4000) / 816.5`
- mean = 4000 (midpoint of {3000, 4000, 5000})
- std = 816.5 (exact population std of {3000, 4000, 5000}, rounded from √(2/3)×1000)
- **Note**: std is hardcoded to match this fixed 3-Re set. Adding a new Re value requires updating this constant.

Re is also stored as a separate per-case tensor `re_values` (shape `[batch]`, dtype float), passed explicitly to the physics loss. Re is **not** recovered algebraically from the normalised branch vector during physics computation — this avoids unnecessary gradient flow through Re.

### Trunk Input

For each Re, `N_trunk` grid points are sampled per training step. Each point is replicated 3 times for component index c ∈ {0=u, 1=v, 2=p}:
```
(x, y, c)  shape: [N_trunk × 3, 3]
```

### Train / Val Split

- All 3 Re values participate in training (no Re-level split — too few cases)
- Per Re: 80% of the 66,049 grid points → train trunk pool; 20% → val trunk pool
- Val evaluation: full val pool, mean relative L2 error across all 3 Re and all 3 components
- Best checkpoint selection: minimum validation mean relative L2 error, evaluated every `checkpoint_period` steps

---

## Model Architecture

### Branch Net

`ResNetBranchNet(flat_dim=301, hidden_dims=branch_hidden_dims, latent_width=latent_width)` used directly (input shape `[batch, 301]`).

When `use_resnet_branch=False`, falls back to `SimpleMLP(layer_sizes=[301, *branch_hidden_dims, latent_width], activation="tanh", kernel_initializer="Glorot normal")`.

### Trunk Net — `LDCFourierTrunkNet`

New class in `ldc_train.py`. Does **not** modify existing `FourierFeatureTrunkNet`.

**Note on shape divergence**: The existing `FourierFeatureTrunkNet` uses `B` of shape `[3, num_features]` to encode `(x, y, t)`. `LDCFourierTrunkNet` uses `B` of shape `[2, num_features]` to encode `(x, y)` only (no time). Do not copy the existing class without updating this shape.

**Input**: `(x, y, c)` — shape `[N, 3]`

**RFF encoding** (applied to spatial coordinates `(x, y)` only):
- Frequency matrix `B`, shape `[2, num_features]`, entries i.i.d. sampled as `torch.randn(2, num_features, dtype=dde_config.real(torch)) * sigma` and registered as a non-trainable buffer via `register_buffer`.
- Encoding: `γ(z) = [sin(2π·z·B), cos(2π·z·B)]` where `z = inputs[:, :2]`, product `z @ B` has shape `[N, num_features]` — output dim = `2 * num_features`

**Component index `c`**:
- `nn.Embedding(num_embeddings=3, embedding_dim=8)`
- Init: `nn.init.normal_(weight, mean=0.0, std=0.1)`

**Final trunk input**: `[γ(x,y), embed(c)]` → dim = `2 * num_features + 8`

**Trunk core**: `SimpleMLP(layer_sizes=[2*trunk_rff_features+8, *trunk_hidden_dims, latent_width], activation="tanh", kernel_initializer="Glorot normal")`

`trunk_rff_features` must be > 0; raise `ValueError` if zero or negative.

---

## Physics Loss — Steady NS

Steady incompressible Navier-Stokes (no time derivative):

```
NS_x:        u·∂u/∂x + v·∂u/∂y + ∂p/∂x - (1/Re)(∂²u/∂x² + ∂²u/∂y²) = 0
NS_y:        u·∂v/∂x + v·∂v/∂y + ∂p/∂y - (1/Re)(∂²v/∂x² + ∂²v/∂y²) = 0
Continuity:  ∂u/∂x + ∂v/∂y = 0
```

**Physics collocation strategy**:
- Per training step: sample `num_physics_points` collocation points `(x, y)` uniformly from `(0, 1)²` (open interior), `requires_grad=True`
- The **same** set of `num_physics_points` collocation points is reused across all 3 Re cases in a batch. The model is queried once per Re case with these shared coordinates, yielding per-Re `(u, v, p)` fields. NS residuals are computed independently for each Re case using its `re_values` scalar.
- Derivatives via `torch.autograd.grad` (first and second order w.r.t. `x`, `y`)

### Boundary Condition Loss

LDC Dirichlet BCs (do not vary with Re):
- Top wall (y=1): u=1, v=0
- Bottom wall (y=0): u=0, v=0
- Left wall (x=0): u=0, v=0
- Right wall (x=1): u=0, v=0

`num_bc_points` is the **total** number of BC collocation points per training step, split equally across 4 walls (`num_bc_points // 4` per wall). Only `u` and `v` are constrained; pressure is intentionally excluded from `L_bc` because steady NS does not impose Dirichlet pressure at walls — the pressure gauge is handled separately via `L_gauge`.

```
L_bc = MSE(u_pred - u_bc) + MSE(v_pred - v_bc)     # all 4 walls combined
```

### Pressure Gauge

Steady incompressible NS determines pressure only up to an additive constant. To pin the gauge:
```
L_gauge = (p_pred(x=0, y=0) - 0)²
```
where `(x=0, y=0)` is the bottom-left corner, and reference pressure = 0.

### Total Loss

```
L_total = data_loss_weight         × L_data
        + physics_loss_weight      × (L_NS + physics_continuity_weight × L_continuity)
        + bc_loss_weight           × L_bc
        + gauge_loss_weight        × L_gauge
```

All weight keys match config parameter names exactly.

---

## CLI & Config

New entry point: `python -m pi_onet.ldc_train --config configs/ldc_re3000_5000.toml`

Example config:
```toml
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
num_bc_points = 100          # total across all 4 walls (25 per wall)
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
artifacts_dir = "../artifacts/ldc-resnet-rff"
```

Note: `batch_size = 3` means all 3 Re cases per step. `rollout_steps` is not applicable for steady-state and is not a parameter in `ldc_train.py`.

---

## Output Structure

Best checkpoint is saved every `checkpoint_period` steps, triggered by validation evaluation at the same interval.

```
artifacts/ldc-resnet-rff/
  experiment_manifest.json
  checkpoints/
    ldc_deeponet_step_<N>.pt
  best_validation/
    ldc_deeponet_best.pt
    best_validation_summary.json
  final_evaluation.json
```

---

## Implementation Notes

### Import path for shared model classes

```python
from pi_onet.train import ResNetBranchNet, SimpleMLP
```

This import triggers two idempotent `os.environ.setdefault` calls in `train.py` (`DDE_BACKEND` and `PYTORCH_ENABLE_MPS_FALLBACK`). This is acceptable — `ldc_train.py` is a similar entry point that would set these vars anyway. No architectural restructuring of `train.py` is required.

### `L_data` definition

`L_data` is the MSE between model predictions and ground-truth field values at the `N_trunk` trunk points sampled from each Re case's grid per training step:

```
L_data = (1/num_Re) × Σ_Re  MSE(model(branch_Re, trunk_pts), ref_values_Re)
```

where `ref_values_Re` is the ground-truth `(u, v, p)` value at each trunk point read from the `.mat` grid for that Re. Trunk points are sampled from the train pool (80% of 66,049 points) at each step.

### Physics collocation forward pass

During physics loss computation, trunk input is constructed as follows:

```python
# xy: [num_physics_points, 2], requires_grad=True
# c_vals: [num_physics_points * 3] — repeat each point for c=0,1,2
trunk_phys = torch.cat([
    xy.repeat_interleave(3, dim=0),          # [num_physics_points*3, 2]
    torch.tensor([0,1,2]).repeat(num_physics_points).unsqueeze(1)  # c column
], dim=1)  # → [num_physics_points*3, 3]
```

`requires_grad=True` is set on `xy` (the spatial coordinate slice) only, not on the `c` index column.

Model output shape: `[num_physics_points * 3, 1]` → reshaped to `[num_physics_points, 3]`:
- `out[:, 0]` = u predictions
- `out[:, 1]` = v predictions
- `out[:, 2]` = p predictions

Autograd derivatives are taken w.r.t. `xy`:
```python
du_dx, du_dy = autograd(u, xy)
dv_dx, dv_dy = autograd(v, xy)
dp_dx, dp_dy = autograd(p, xy)
d2u_dx2 = autograd(du_dx, xy)[0]
d2u_dy2 = autograd(du_dy, xy)[1]
# analogous for v
```

---

## What Is Not Changed

- `train.py`, `dataset.py`, `evaluate_checkpoint.py` — untouched
- Existing Kolmogorov configs — fully backward compatible
- `ResNetBranchNet`, `SimpleMLP` — imported, not modified
