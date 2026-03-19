# Architecture Design: ResNet Branch + RFF Trunk

**Date**: 2026-03-19
**Scope**: `src/pi_onet/train.py`
**Status**: Pending Review

---

## Motivation

Current architecture suffers from two known failure modes:

1. **Spectral bias (F-Principle)**: Standard MLP trunk learns low-frequency components first, struggling with the high-frequency spatial structure of turbulent Kolmogorov flow.
2. **Branch gradient instability**: GatedMLP on a 7500-dim flattened input has no identity path. Deep residual connections provide unobstructed gradient flow across many layers.

---

## Design

### 1. `FourierFeatureTrunkNet` — RFF Encoding for Trunk Input

**Purpose**: Replace `TimeFourierTrunkNet` (time-only Fourier) with full spatial-temporal RFF encoding.

**Input**: `(x, y, t, c)` — shape `[N, 4]`

#### RFF Encoding (applied to continuous coordinates only)

- Apply RFF to `(x, y, t)` ∈ ℝ³ only. Component index `c ∈ {0, 1, 2}` is discrete and must be treated separately.
- Frequency matrix `B`, shape `[3, num_features]` — each entry sampled **i.i.d.** from `N(0, σ²)`. Sampled once at construction, registered as buffer (non-trainable). This is the standard RFF approximation of the Gaussian (RBF) kernel.
- Encoding: `γ(z) = [sin(2π·z·B), cos(2π·z·B)]` where `z` is `[N, 3]` and `z @ B` is `[N, num_features]` → output dim = `2 * num_features`

#### Component Index `c` Handling

- Problem: with `2 * num_features` (e.g. 512) RFF features bounded in `[-1, 1]`, a raw scalar `c ∈ {0,1,2}` would dominate or be dominated depending on scale.
- Solution: pass `c` (integer indices 0/1/2) through `nn.Embedding(num_embeddings=3, embedding_dim=8)`.
  - Initialization: `nn.init.normal_(embedding.weight, mean=0.0, std=0.1)` — small scale to match the `[-1, 1]` range of RFF outputs.
- Final trunk input: `[γ(x,y,t), embed(c)]` → dim = `2 * num_features + 8`

#### Trunk Core

**Always `SimpleMLP with Tanh`**, independent of `use_gated_mlp`. The trunk core MLP type is decoupled from the branch choice. When `FourierFeatureTrunkNet` is active (`trunk_rff_features > 0`), the trunk core always uses `SimpleMLP`.

Trunk layer sizes: `[2 * trunk_rff_features + 8, *trunk_hidden_dims, latent_width]`

This replaces the existing `trunk_core_input_dim = trunk_dim + 2 * time_fourier_modes` calculation, which only applies in the `TimeFourierTrunkNet` path.

#### Backward Compatibility

When `trunk_rff_features == 0`, fall back to existing `TimeFourierTrunkNet` + existing `trunk_core_input_dim` calculation. `--time-fourier-modes` still applies in that path. No existing experiments are broken.

---

### 2. `ResNetBranchNet` — Pre-activation ResNet Branch

**Purpose**: Add as an independent branch option alongside existing GatedMLP and Transformer. Enables ablation studies.

#### Pre-activation ResBlock (He v2)

Each block operates on a vector of dimension `hidden_dim = branch_hidden_dims[0]`:

```
block(x):                                # x: [batch, hidden_dim]
    h = LN(x)                            # LayerNorm(hidden_dim)
    h = Tanh(h)
    h = Linear(hidden_dim → hidden_dim)  # first linear
    h = LN(h)
    h = Tanh(h)
    h = Linear(hidden_dim → hidden_dim)  # second linear
    return h

output = x + block(x)                   # skip connection (no projection needed — dims match)
```

Full pre-activation ordering: `LN → Tanh → Linear → LN → Tanh → Linear`. The identity skip path from input projection to output projection is fully unobstructed.

#### Full Architecture

```
flat input  [batch, flat_dim]             (e.g. flat_dim = 7500)
    ↓  Linear(flat_dim → hidden_dim)      input projection
    ↓  Tanh
    ↓  ResBlock × N                       N = len(branch_hidden_dims)
    ↓  Linear(hidden_dim → latent_width)  output projection
```

#### Parameter Mapping

`branch_hidden_dims` is reused: `hidden_dim = branch_hidden_dims[0]`, `N = len(branch_hidden_dims)`. All values must be equal (same constraint as GatedMLP). A `ValueError` is raised if values differ.

Example: `branch_hidden_dims = [512, 512]` → 2 blocks, hidden_dim = 512.

#### Initialization

- **Input projection** `Linear(flat_dim → hidden_dim)`: `nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='tanh')` — critical for 7500-dim input to prevent gradient explosion at layer 0. Bias: zeros.
- **ResBlock Linear layers**: `nn.init.xavier_uniform_(weight)`. Bias: zeros.
- **Output projection** `Linear(hidden_dim → latent_width)`: `nn.init.xavier_uniform_(weight)`. Bias: zeros.
- **LayerNorm**: default (weight=1, bias=0).

Wrapped in `FlattenBranchNet` (existing class, unchanged).

---

### 3. New CLI Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--use-resnet-branch` | flag | `False` | Enable ResNet branch (independent of `--use-gated-mlp`, `--use-transformer-branch`) |
| `--trunk-rff-features` | int | `0` | RFF feature count. `0` = disabled (falls back to `--time-fourier-modes`) |
| `--trunk-rff-sigma` | float | `1.0` | Gaussian bandwidth σ for frequency sampling. **For turbulence, recommended range: 10.0–50.0** |

These three parameters must be added to **both** `DEFAULT_TRAIN_ARGS` (so `load_train_config` TOML validation passes) **and** `build_arg_parser()` (so CLI flags work). Both updates are required.

**`DEFAULT_TRAIN_ARGS` additions**:
```python
"use_resnet_branch": False,
"trunk_rff_features": 0,
"trunk_rff_sigma": 1.0,
```

**`build_arg_parser()` additions**:
```python
parser.add_argument("--use-resnet-branch", action="store_true", default=None)
parser.add_argument("--trunk-rff-features", type=int, default=None)
parser.add_argument("--trunk-rff-sigma", type=float, default=None)
```

Note: `--trunk-rff-features` and `--trunk-rff-sigma` use `default=None` in `build_arg_parser()` by design. The existing `main()` merge pattern resolves CLI `None` by falling back to the value in `DEFAULT_TRAIN_ARGS` (i.e. `0` and `1.0` respectively). This is the same pattern used by all other numeric parameters in the codebase (e.g. `--physics-time-samples`).

---

### 4. `create_model()` Signature Extension

```python
def create_model(
    ...
    use_resnet_branch: bool = False,       # new
    trunk_rff_features: int = 0,           # new
    trunk_rff_sigma: float = 1.0,          # new
) -> dde.nn.pytorch.deeponet.DeepONetCartesianProd:
```

**Branch selection logic** (priority order):
1. `use_transformer_branch=True` → `TemporalTransformerBranch` (unchanged)
2. `use_resnet_branch=True` → `ResNetBranchNet` + `FlattenBranchNet`
3. `use_gated_mlp=True` → `ModifiedGatedMLP` + `FlattenBranchNet` (unchanged)
4. default → `SimpleMLP` + `FlattenBranchNet` (unchanged)

**Trunk wrapper selection**:
- `trunk_rff_features > 0` → `FourierFeatureTrunkNet` (trunk core = `SimpleMLP`, always; trunk_layers = `[2*trunk_rff_features+8, *trunk_hidden_dims, latent_width]`)
- otherwise → `TimeFourierTrunkNet` (existing path unchanged; trunk core depends on `use_gated_mlp`; trunk_layers = `[trunk_dim + 2*time_fourier_modes, *trunk_hidden_dims, latent_width]`)

---

### 5. New Classes Summary

| Class | Location | Replaces / Adds |
|---|---|---|
| `FourierFeatureTrunkNet` | `train.py` | Adds (replaces `TimeFourierTrunkNet` when `trunk_rff_features > 0`) |
| `ResNetBlock` | `train.py` | New |
| `ResNetBranchNet` | `train.py` | New branch option |

`TimeFourierTrunkNet`, `ModifiedGatedMLP`, `TemporalTransformerBranch`, `SimpleMLP` are all preserved unchanged.

---

## What Is Not Changed

- `PhysicsInformedTripleCartesianProd` physics loss — untouched
- `DatasetConfig`, `build_dataset` — untouched
- `evaluate`, `rollout_evaluate`, `predict_raw` — untouched
- Existing CLI parameters — all preserved, defaults unchanged
- Existing config `.toml` files — fully backward compatible

---

## Recommended Experiment Config (new architecture)

```toml
[train]
use_resnet_branch = true
trunk_rff_features = 256
trunk_rff_sigma = 10.0
use_gated_mlp = false
branch_hidden_dims = [512, 512, 512, 512]
trunk_hidden_dims = [512, 512, 512, 512]
latent_width = 256
```
