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
- `ResNetBranchNet`, `FlattenBranchNet`, `SimpleMLP`

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

- **80 interior sensors**: randomly sampled from interior grid points (excluding boundary rows/columns)
- **20 boundary sensors**: 5 points per wall (top, bottom, left, right), uniformly spaced

Sensor positions are fixed at dataset construction time and shared across all Re cases.

### Branch Input

One branch vector per Re case, dim = `1 + 100 × 3 = 301`:
```
[Re_norm, u_s1, v_s1, p_s1, ..., u_s80, v_s80, p_s80,   # 80 interior sensors
          u_b1, v_b1, p_b1, ..., u_b20, v_b20, p_b20]    # 20 boundary sensors
```

Re normalization: `Re_norm = (Re - 4000) / 816`
- mean = 4000 (midpoint of {3000, 4000, 5000})
- std = 816 (population std of {3000, 4000, 5000})

### Trunk Input

For each Re, `N_trunk` grid points are sampled per training step. Each point is replicated 3 times for component index c ∈ {0=u, 1=v, 2=p}:
```
(x, y, c)  shape: [N_trunk × 3, 3]
```

### Train / Val Split

- All 3 Re values participate in training (no Re-level split — too few cases)
- Per Re: 80% of the 66,049 grid points → train trunk pool; 20% → val trunk pool
- Val evaluation: trunk points sampled from val pool

---

## Model Architecture

### Branch Net

`ResNetBranchNet(flat_dim=301, hidden_dims, latent_width)` wrapped in `FlattenBranchNet`.

Input shape to `FlattenBranchNet`: `[batch, 1, 301]` → flattened to `[batch, 301]`.

When `use_resnet_branch=False`, falls back to `SimpleMLP`.

### Trunk Net — `LDCFourierTrunkNet`

New class in `ldc_train.py` (does **not** modify existing `FourierFeatureTrunkNet`).

**Input**: `(x, y, c)` — shape `[N, 3]`

**RFF encoding** (applied to spatial coordinates only):
- Frequency matrix `B`, shape `[2, num_features]`, entries i.i.d. from `N(0, σ²)`. Registered as buffer (non-trainable).
- Encoding: `γ(z) = [sin(2π·z·B), cos(2π·z·B)]` where `z = [x, y]` — output dim = `2 * num_features`

**Component index `c`**:
- `nn.Embedding(num_embeddings=3, embedding_dim=8)`
- Init: `nn.init.normal_(weight, mean=0.0, std=0.1)`

**Final trunk input**: `[γ(x,y), embed(c)]` → dim = `2 * num_features + 8`

**Trunk core**: `SimpleMLP` with Tanh, layers = `[2*trunk_rff_features+8, *trunk_hidden_dims, latent_width]`

When `trunk_rff_features == 0`, trunk core input is `(x, y, c)` directly → `SimpleMLP` layers = `[3, *trunk_hidden_dims, latent_width]`.

---

## Physics Loss — Steady NS

Steady incompressible Navier-Stokes (no time derivative):

```
NS_x:        u·∂u/∂x + v·∂u/∂y + ∂p/∂x - (1/Re)(∂²u/∂x² + ∂²u/∂y²) = 0
NS_y:        u·∂v/∂x + v·∂v/∂y + ∂p/∂y - (1/Re)(∂²v/∂x² + ∂²v/∂y²) = 0
Continuity:  ∂u/∂x + ∂v/∂y = 0
```

**Implementation**:
- Physics trunk points `(x, y)` sampled uniformly from `[0,1]²` (continuous, not restricted to grid), `requires_grad=True`
- `u`, `v`, `p` obtained by querying the model with `c=0,1,2` respectively
- Derivatives computed via `torch.autograd.grad` (first and second order)
- Re recovered from branch input: `Re = Re_norm × 816 + 4000`

**Physics loss**:
```
L_physics = MSE(NS_x) + MSE(NS_y) + continuity_weight × MSE(continuity)
```

**Total loss**:
```
L_total = data_loss_weight × L_data + physics_loss_weight × L_physics
```

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
branch_hidden_dims = [256, 256]
trunk_hidden_dims = [256, 256]
trunk_rff_features = 128
trunk_rff_sigma = 5.0
latent_width = 128
use_resnet_branch = true
data_loss_weight = 1.0
physics_loss_weight = 0.1
physics_continuity_weight = 1.0
iterations = 10000
learning_rate = 0.001
batch_size = 3
seed = 42
artifacts_dir = "../artifacts/ldc-resnet-rff"
```

Note: `batch_size = 3` means all 3 Re cases per step (one full pass over all Re).

---

## Output Structure

Mirrors Kolmogorov pipeline:
```
artifacts/ldc-resnet-rff/
  experiment_manifest.json
  checkpoints/
    kolmogorov_deeponet_step_<N>.pt   # reuse naming convention
  best_validation/
  final_evaluation.json
```

---

## What Is Not Changed

- `train.py`, `dataset.py`, `evaluate_checkpoint.py` — untouched
- Existing Kolmogorov configs — fully backward compatible
- `ResNetBranchNet`, `FlattenBranchNet`, `SimpleMLP` — imported, not modified
