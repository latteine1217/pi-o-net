# ResNet Branch + RFF Trunk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ResNetBranchNet` (pre-activation ResNet) and `FourierFeatureTrunkNet` (Random Fourier Features) as new independently-selectable architecture options in the PI-DeepONet training pipeline.

**Architecture:** New branch option selectable via `--use-resnet-branch`; new trunk encoding selectable via `--trunk-rff-features > 0`. All existing options (GatedMLP, Transformer branch, TimeFourier trunk) are preserved unchanged. `create_model()` is refactored to a single-return structure for clarity.

**Tech Stack:** PyTorch, DeepXDE, pytest, uv

**Spec:** `docs/superpowers/specs/2026-03-19-resnet-branch-rff-trunk-design.md`

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `src/pi_onet/train.py` | Modify | Add 3 new classes, extend `create_model()`, wire CLI |
| `src/pi_onet/evaluate_checkpoint.py` | Modify | Pass new params to `create_model()`, update print dict |
| `tests/test_model.py` | Create | Tests for new architecture components |

---

## Task 1: `FourierFeatureTrunkNet` — RFF Trunk Encoding

**Files:**
- Modify: `src/pi_onet/train.py` (insert after class `TimeFourierTrunkNet` ends, before `class CallableTrunkDeepONetCartesianProd`)
- Create: `tests/test_model.py`

- [ ] **Step 1: Create `tests/test_model.py` with ALL imports upfront and Task 1 tests**

Note: All imports from `pi_onet.train` for Tasks 1–3 must be declared together at the top of the file. This prevents collection errors when running tests for earlier tasks before later classes exist. Task 2 and 3 will only **add test classes**, not modify the imports.

Create `tests/test_model.py`:

```python
"""Tests for train.py neural network architecture components."""
from __future__ import annotations

import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np
import torch
import pytest

# All imports from train.py — declared upfront so the file structure is stable
# across all tasks. ImportError here is the expected TDD "red" state for not-yet-implemented classes.
from pi_onet.train import (
    FourierFeatureTrunkNet,
    ResNetBlock,
    ResNetBranchNet,
    FlattenBranchNet,
    SimpleMLP,
    create_model,
)


def _make_rff_trunk(num_features: int = 16, sigma: float = 1.0) -> FourierFeatureTrunkNet:
    """Helper: build a small FourierFeatureTrunkNet for testing."""
    core = SimpleMLP(
        [2 * num_features + 8, 32, 16], activation="tanh", kernel_initializer="Glorot normal"
    )
    return FourierFeatureTrunkNet(num_features=num_features, sigma=sigma, core_net=core)


class TestFourierFeatureTrunkNet:
    def test_output_shape(self):
        trunk = _make_rff_trunk(num_features=16)
        x = torch.zeros(10, 4)
        x[:, 3] = torch.arange(10) % 3
        out = trunk(x)
        assert out.shape == (10, 16)

    def test_B_is_buffer_not_parameter(self):
        trunk = _make_rff_trunk()
        param_names = {n for n, _ in trunk.named_parameters()}
        buffer_names = {n for n, _ in trunk.named_buffers()}
        assert "B" not in param_names, "B must NOT be a trainable parameter"
        assert "B" in buffer_names, "B must be a registered buffer"

    def test_B_shape(self):
        trunk = _make_rff_trunk(num_features=32)
        assert trunk.B.shape == (3, 32)

    def test_embedding_weight_scale_small(self):
        trunk = _make_rff_trunk()
        std = trunk.component_embedding.weight.std().item()
        assert std < 0.3, f"Embedding std {std:.3f} is too large — should be ~0.1"

    def test_invalid_num_features_zero_raises(self):
        core = SimpleMLP([8, 16], activation="tanh", kernel_initializer="Glorot normal")
        with pytest.raises(ValueError, match="num_features"):
            FourierFeatureTrunkNet(num_features=0, sigma=1.0, core_net=core)

    def test_invalid_sigma_zero_raises(self):
        core = SimpleMLP([2 * 8 + 8, 16], activation="tanh", kernel_initializer="Glorot normal")
        with pytest.raises(ValueError, match="sigma"):
            FourierFeatureTrunkNet(num_features=8, sigma=0.0, core_net=core)

    def test_invalid_input_dim_raises(self):
        trunk = _make_rff_trunk()
        x = torch.randn(5, 3)
        with pytest.raises(ValueError):
            trunk(x)

    def test_forward_differentiable(self):
        trunk = _make_rff_trunk()
        x = torch.zeros(4, 4)
        x[:, 3] = torch.tensor([0.0, 1.0, 2.0, 0.0])
        out = trunk(x)
        out.sum().backward()
```

- [ ] **Step 2: Run tests to confirm Task 1 tests fail**

```bash
uv run pytest tests/test_model.py::TestFourierFeatureTrunkNet -x -q 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'FourierFeatureTrunkNet'`

- [ ] **Step 3: Add `FourierFeatureTrunkNet` to `train.py`**

Insert after class `TimeFourierTrunkNet` ends (line ~474), before `class CallableTrunkDeepONetCartesianProd`:

```python
class FourierFeatureTrunkNet(torch.nn.Module):
    """What: 以 Random Fourier Features (RFF) 對空間-時間座標 (x,y,t) 做全座標編碼。

    Why: 解決 MLP trunk 的 spectral bias (F-Principle)，使高頻湍流特徵可被學習。
         B ~ N(0, σ²) 各元素 i.i.d. 採樣，逼近 Gaussian (RBF) kernel 的特徵映射。
         離散 component index c 另以可學習 Embedding 處理，避免被大維度 RFF 特徵淹沒。
    """

    def __init__(self, num_features: int, sigma: float, core_net: torch.nn.Module) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features 必須為正整數。")
        if sigma <= 0.0:
            raise ValueError("sigma 必須為正數。")
        self.num_features = int(num_features)
        self.core_net = core_net

        # Component index c ∈ {0,1,2} → learnable 8-dim embedding
        # std=0.1 to match the scale of RFF outputs bounded in [-1, 1]
        self.component_embedding = torch.nn.Embedding(3, 8)
        torch.nn.init.normal_(self.component_embedding.weight, mean=0.0, std=0.1)

        # RFF frequency matrix B ~ N(0, σ²) i.i.d., shape [3, num_features]
        # dtype=dde_config.real(torch) ensures consistency with model parameters
        B = torch.randn(3, self.num_features, dtype=dde_config.real(torch)) * float(sigma)
        self.register_buffer("B", B)

    def encoded_dim(self) -> int:
        """What: 返回 RFF + embedding 後的總維度。"""
        return 2 * self.num_features + 8

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """What: RFF 編碼 (x,y,t)，embed c，拼接後交給 core_net。"""
        if inputs.shape[1] != 4:
            raise ValueError(
                f"FourierFeatureTrunkNet 僅支援 (x,y,t,c) 4 維輸入，但收到 {inputs.shape[1]} 維。"
            )
        xyz = inputs[:, :3]  # [N, 3]
        c_idx = inputs[:, 3].long()  # [N] — float index → int for embedding lookup

        # γ(z) = [sin(2π·z·B), cos(2π·z·B)], z @ B: [N,3] × [3,F] → [N,F]
        proj = 2.0 * np.pi * (xyz @ self.B.to(dtype=inputs.dtype))  # [N, num_features]
        rff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)  # [N, 2*num_features]

        c_emb = self.component_embedding(c_idx).to(dtype=inputs.dtype)  # [N, 8]
        encoded = torch.cat([rff, c_emb], dim=1)  # [N, 2*num_features + 8]
        return self.core_net(encoded)
```

- [ ] **Step 4: Run tests — expect all Task 1 tests to pass**

```bash
uv run pytest tests/test_model.py::TestFourierFeatureTrunkNet -v 2>&1
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_model.py src/pi_onet/train.py
git commit -m "feat: add FourierFeatureTrunkNet with RFF encoding for trunk input"
```

---

## Task 2: `ResNetBlock` + `ResNetBranchNet`

**Files:**
- Modify: `src/pi_onet/train.py` (insert after `class FlattenBranchNet`, before `class TemporalTransformerBranch`)
- Modify: `tests/test_model.py` (append new test classes — imports already declared in Task 1)

- [ ] **Step 1: Append `TestResNetBlock` and `TestResNetBranchNet` to `tests/test_model.py`**

Do NOT modify the existing import block. Simply append to the end of the file:

```python
class TestResNetBlock:
    def test_output_shape_matches_input(self):
        block = ResNetBlock(hidden_dim=64)
        x = torch.randn(8, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_skip_connection_active(self):
        """With x non-zero, output = x + block(x) must exceed block(x) magnitude."""
        torch.manual_seed(0)
        block = ResNetBlock(hidden_dim=32)
        x = torch.ones(4, 32)  # use ones so x is definitely non-zero
        out = block(x)
        # block(x) alone — computed without x
        with torch.no_grad():
            h = torch.tanh(block.norm1(x))
            h = block.linear1(h)
            h = torch.tanh(block.norm2(h))
            h = block.linear2(h)
        # out = x + h, so out - h should equal x (all-ones)
        diff = (out.detach() - h).abs().mean().item()
        assert diff > 0.1, f"Skip connection appears inactive (mean diff={diff:.4f})"

    def test_backward_through_block(self):
        block = ResNetBlock(hidden_dim=16)
        x = torch.randn(4, 16, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestResNetBranchNet:
    def test_output_shape(self):
        net = ResNetBranchNet(flat_dim=100, hidden_dims=[64, 64], latent_width=32)
        x = torch.randn(8, 100)
        out = net(x)
        assert out.shape == (8, 32)

    def test_single_block(self):
        net = ResNetBranchNet(flat_dim=50, hidden_dims=[32], latent_width=16)
        x = torch.randn(4, 50)
        out = net(x)
        assert out.shape == (4, 16)

    def test_unequal_hidden_dims_raises(self):
        with pytest.raises(ValueError, match="全部相同"):
            ResNetBranchNet(flat_dim=100, hidden_dims=[64, 128], latent_width=32)

    def test_empty_hidden_dims_raises(self):
        with pytest.raises(ValueError):
            ResNetBranchNet(flat_dim=100, hidden_dims=[], latent_width=32)

    def test_input_projection_kaiming_init_scale(self):
        """Kaiming Normal (fan_in, tanh): expected std ≈ sqrt(2/fan_in)."""
        flat_dim = 400
        net = ResNetBranchNet(flat_dim=flat_dim, hidden_dims=[64], latent_width=16)
        expected_std = (2.0 / flat_dim) ** 0.5  # ~0.071
        actual_std = net.input_proj.weight.std().item()
        assert actual_std < expected_std * 3.0, (
            f"Input proj std {actual_std:.4f} exceeds 3× Kaiming expected {expected_std:.4f}"
        )

    def test_backward(self):
        net = ResNetBranchNet(flat_dim=40, hidden_dims=[16, 16], latent_width=8)
        x = torch.randn(4, 40)
        out = net(x)
        out.sum().backward()
        assert net.input_proj.weight.grad is not None
```

- [ ] **Step 2: Run tests to confirm Task 2 tests fail**

```bash
uv run pytest tests/test_model.py::TestResNetBlock tests/test_model.py::TestResNetBranchNet -x -q 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'ResNetBlock'`

- [ ] **Step 3: Add `ResNetBlock` and `ResNetBranchNet` to `train.py`**

Insert after class `FlattenBranchNet` ends (line ~367), before `class TemporalTransformerBranch`:

```python
class ResNetBlock(torch.nn.Module):
    """What: Pre-activation ResNet block (He v2)。

    Why: Identity path `x + block(x)` 確保梯度從 output 直通至 input projection，
         不受深度影響。LayerNorm → Tanh → Linear × 2 是 He v2 的標準順序。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        dim = int(hidden_dim)
        self.norm1 = torch.nn.LayerNorm(dim, dtype=dde_config.real(torch))
        self.linear1 = torch.nn.Linear(dim, dim, dtype=dde_config.real(torch))
        self.norm2 = torch.nn.LayerNorm(dim, dtype=dde_config.real(torch))
        self.linear2 = torch.nn.Linear(dim, dim, dtype=dde_config.real(torch))
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear1.bias)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.norm1(x))
        h = self.linear1(h)
        h = torch.tanh(self.norm2(h))
        h = self.linear2(h)
        return x + h


class ResNetBranchNet(torch.nn.Module):
    """What: Pre-activation ResNet branch encoder。

    Why: 高維感測器序列（~7500 維）的梯度不穩定問題，需要 identity path 解決。
         Kaiming Normal 初始化 input projection，確保第一層不出現梯度爆炸。
         branch_hidden_dims 全部相同時，skip connection 不需額外 projection 層。
    """

    def __init__(self, flat_dim: int, hidden_dims: list[int], latent_width: int) -> None:
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("ResNetBranchNet 需要至少一個 hidden dim。")
        if len(set(hidden_dims)) != 1:
            raise ValueError(
                "ResNetBranchNet 要求 branch_hidden_dims 全部相同，例如 [512, 512]。"
            )
        hidden_dim = int(hidden_dims[0])
        num_blocks = len(hidden_dims)

        # Input projection: flat_dim → hidden_dim
        self.input_proj = torch.nn.Linear(
            int(flat_dim), hidden_dim, dtype=dde_config.real(torch)
        )
        torch.nn.init.kaiming_normal_(
            self.input_proj.weight, mode="fan_in", nonlinearity="tanh"
        )
        torch.nn.init.zeros_(self.input_proj.bias)

        self.blocks = torch.nn.ModuleList(
            [ResNetBlock(hidden_dim) for _ in range(num_blocks)]
        )

        # Output projection: hidden_dim → latent_width
        self.output_proj = torch.nn.Linear(
            hidden_dim, int(latent_width), dtype=dde_config.real(torch)
        )
        torch.nn.init.xavier_uniform_(self.output_proj.weight)
        torch.nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.input_proj(x))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)
```

- [ ] **Step 4: Run Task 2 tests**

```bash
uv run pytest tests/test_model.py::TestResNetBlock tests/test_model.py::TestResNetBranchNet -v 2>&1
```

Expected: 9 passed.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
uv run pytest tests/ -x -q 2>&1
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_model.py src/pi_onet/train.py
git commit -m "feat: add ResNetBlock and ResNetBranchNet (He v2 pre-activation)"
```

---

## Task 3: Extend `create_model()` + Integration Tests

**Files:**
- Modify: `src/pi_onet/train.py` — `create_model()` function (lines 503–562)
- Modify: `tests/test_model.py` — append integration tests

- [ ] **Step 1: Append `TestCreateModelNewOptions` to `tests/test_model.py`**

```python
class TestCreateModelNewOptions:
    """Integration tests: create_model builds without error and produces correct outputs."""

    _BRANCH_SHAPE = (2, 3)  # (history_steps=2, features_per_step=3)
    _TRUNK_DIM = 4           # (x, y, t, c)
    _HIDDEN = [16, 16]
    _LATENT = 8

    def _branch_input(self) -> np.ndarray:
        return np.random.randn(5, 2, 3).astype(np.float64)

    def _trunk_input(self, num_trunk: int = 12) -> np.ndarray:
        trunk = np.random.randn(num_trunk, 4)
        trunk[:, 3] = np.tile([0, 1, 2], num_trunk // 3 + 1)[:num_trunk]
        return trunk.astype(np.float64)

    def test_resnet_branch_builds_and_runs(self):
        model = create_model(
            branch_shape=self._BRANCH_SHAPE,
            trunk_dim=self._TRUNK_DIM,
            branch_hidden_dims=self._HIDDEN,
            trunk_hidden_dims=self._HIDDEN,
            latent_width=self._LATENT,
            use_resnet_branch=True,
        )
        pred = model.predict((self._branch_input(), self._trunk_input()))
        assert pred.shape[0] == 5

    def test_rff_trunk_builds_and_runs(self):
        model = create_model(
            branch_shape=self._BRANCH_SHAPE,
            trunk_dim=self._TRUNK_DIM,
            branch_hidden_dims=self._HIDDEN,
            trunk_hidden_dims=self._HIDDEN,
            latent_width=self._LATENT,
            trunk_rff_features=16,
            trunk_rff_sigma=1.0,
        )
        pred = model.predict((self._branch_input(), self._trunk_input()))
        assert pred.shape[0] == 5

    def test_resnet_branch_plus_rff_trunk(self):
        model = create_model(
            branch_shape=self._BRANCH_SHAPE,
            trunk_dim=self._TRUNK_DIM,
            branch_hidden_dims=self._HIDDEN,
            trunk_hidden_dims=self._HIDDEN,
            latent_width=self._LATENT,
            use_resnet_branch=True,
            trunk_rff_features=16,
            trunk_rff_sigma=5.0,
        )
        pred = model.predict((self._branch_input(), self._trunk_input()))
        assert pred.shape[0] == 5

    def test_backward_compat_all_new_defaults(self):
        """New params at defaults must not change behavior vs original call."""
        model = create_model(
            branch_shape=self._BRANCH_SHAPE,
            trunk_dim=self._TRUNK_DIM,
            branch_hidden_dims=self._HIDDEN,
            trunk_hidden_dims=self._HIDDEN,
            latent_width=self._LATENT,
            use_resnet_branch=False,
            trunk_rff_features=0,
            trunk_rff_sigma=1.0,
        )
        pred = model.predict((self._branch_input(), self._trunk_input()))
        assert pred.shape[0] == 5
```

- [ ] **Step 2: Run tests to see them fail**

```bash
uv run pytest tests/test_model.py::TestCreateModelNewOptions -x -q 2>&1 | head -10
```

Expected: `TypeError: create_model() got an unexpected keyword argument 'use_resnet_branch'`

- [ ] **Step 3: Replace `create_model()` in `train.py` (lines 503–562)**

```python
def create_model(
    branch_shape: tuple[int, ...],
    trunk_dim: int,
    branch_hidden_dims: list[int],
    trunk_hidden_dims: list[int],
    latent_width: int,
    use_gated_mlp: bool = False,
    time_fourier_modes: int = 0,
    use_transformer_branch: bool = False,
    transformer_model_dim: int = 128,
    transformer_num_heads: int = 4,
    transformer_num_layers: int = 2,
    transformer_ff_dim: int = 256,
    transformer_dropout: float = 0.0,
    use_resnet_branch: bool = False,
    trunk_rff_features: int = 0,
    trunk_rff_sigma: float = 1.0,
) -> dde.nn.pytorch.deeponet.DeepONetCartesianProd:
    """What: 建立文獻對齊 DeepONet 模型。

    Why: Branch 與 Trunk 的架構選擇完全解耦，便於消融實驗。
         Branch 優先順序：Transformer > ResNet > GatedMLP > SimpleMLP。
         Trunk：trunk_rff_features > 0 時使用 RFF，否則退回 TimeFourier。
    """
    if len(branch_shape) == 0:
        raise ValueError("branch_shape 不可為空。")
    branch_feature_dim = int(branch_shape[-1])
    branch_flat_dim = int(np.prod(np.asarray(branch_shape, dtype=np.int64)))
    branch_layers = [branch_flat_dim, *branch_hidden_dims, latent_width]

    # --- Branch ---
    if use_transformer_branch:
        branch_net = TemporalTransformerBranch(
            token_dim=branch_feature_dim,
            sequence_length=int(branch_shape[0]) if len(branch_shape) > 1 else 1,
            latent_width=latent_width,
            model_dim=transformer_model_dim,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
            ff_dim=transformer_ff_dim,
            dropout=transformer_dropout,
        )
    elif use_resnet_branch:
        branch_core = ResNetBranchNet(branch_flat_dim, branch_hidden_dims, latent_width)
        branch_net = FlattenBranchNet(branch_core)
    elif use_gated_mlp:
        branch_core = ModifiedGatedMLP(branch_layers, activation="tanh", kernel_initializer="Glorot normal")
        branch_net = FlattenBranchNet(branch_core)
    else:
        branch_core = SimpleMLP(branch_layers, activation="tanh", kernel_initializer="Glorot normal")
        branch_net = FlattenBranchNet(branch_core)

    # --- Trunk ---
    if trunk_rff_features > 0:
        trunk_core_input_dim = 2 * int(trunk_rff_features) + 8
        trunk_layers = [trunk_core_input_dim, *trunk_hidden_dims, latent_width]
        trunk_core = SimpleMLP(trunk_layers, activation="tanh", kernel_initializer="Glorot normal")
        trunk_net = FourierFeatureTrunkNet(trunk_rff_features, trunk_rff_sigma, trunk_core)
    elif use_gated_mlp:
        trunk_core_input_dim = trunk_dim + 2 * int(time_fourier_modes)
        trunk_layers = [trunk_core_input_dim, *trunk_hidden_dims, latent_width]
        trunk_core = ModifiedGatedMLP(trunk_layers, activation="tanh", kernel_initializer="Glorot normal")
        trunk_net = TimeFourierTrunkNet(time_fourier_modes, trunk_core)
    else:
        trunk_core_input_dim = trunk_dim + 2 * int(time_fourier_modes)
        trunk_layers = [trunk_core_input_dim, *trunk_hidden_dims, latent_width]
        trunk_core = SimpleMLP(trunk_layers, activation="tanh", kernel_initializer="Glorot normal")
        trunk_net = TimeFourierTrunkNet(time_fourier_modes, trunk_core)

    return CallableTrunkDeepONetCartesianProd(
        (branch_feature_dim, branch_net),
        (trunk_dim, trunk_net),
        activation="tanh",
        kernel_initializer="Glorot normal",
    )
```

- [ ] **Step 4: Run integration tests**

```bash
uv run pytest tests/test_model.py::TestCreateModelNewOptions -v 2>&1
```

Expected: 4 passed.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -x -q 2>&1
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_model.py src/pi_onet/train.py
git commit -m "feat: extend create_model() with use_resnet_branch and trunk_rff_features"
```

---

## Task 4: CLI Wiring

**Files:**
- Modify: `src/pi_onet/train.py` (5 locations)
- Modify: `src/pi_onet/evaluate_checkpoint.py` (5 locations)

Note on `default=None` pattern: `--use-resnet-branch`, `--trunk-rff-features`, `--trunk-rff-sigma` must use `default=None` in `build_arg_parser()` (consistent with `--use-gated-mlp` and all other numeric args). `DEFAULT_TRAIN_ARGS` holds the actual defaults. The `parse_args()` merge logic resolves `None` to the `DEFAULT_TRAIN_ARGS` value.

### `train.py` changes

- [ ] **Step 1: Add 3 keys to `DEFAULT_TRAIN_ARGS` (line ~73)**

After `"use_transformer_branch": False,`, add:

```python
    "use_resnet_branch": False,
    "trunk_rff_features": 0,
    "trunk_rff_sigma": 1.0,
```

- [ ] **Step 2: Add 3 args to `build_arg_parser()` (line ~201)**

After `parser.add_argument("--use-transformer-branch", action="store_true", default=None)`, add:

```python
    parser.add_argument("--use-resnet-branch", action="store_true", default=None)
    parser.add_argument("--trunk-rff-features", type=int, default=None)
    parser.add_argument("--trunk-rff-sigma", type=float, default=None)
```

- [ ] **Step 3: Add validation in `parse_args()` (line ~277)**

`parse_args()` (NOT `load_train_config`) contains the validation block that ends with `return argparse.Namespace(**merged)`. Add before that return:

```python
    if int(merged["trunk_rff_features"]) < 0:
        raise ValueError("trunk_rff_features 不可小於 0。")
    if float(merged["trunk_rff_sigma"]) <= 0.0:
        raise ValueError("trunk_rff_sigma 必須為正數。")
```

- [ ] **Step 4: Update `create_model()` call in `main()` (line ~1402)**

After `transformer_dropout=args.transformer_dropout,`, add:

```python
        use_resnet_branch=args.use_resnet_branch,
        trunk_rff_features=args.trunk_rff_features,
        trunk_rff_sigma=args.trunk_rff_sigma,
```

- [ ] **Step 5: Update Model print dict in `main()` (line ~1419)**

Replace the existing `branch_encoder` line and `trunk_layers` line; add new fields:

```python
                "branch_encoder": (
                    "transformer" if args.use_transformer_branch
                    else "resnet" if args.use_resnet_branch
                    else "gated_mlp" if args.use_gated_mlp
                    else "mlp"
                ),
                "trunk_encoder": "rff" if args.trunk_rff_features > 0 else "time_fourier",
                "trunk_rff_features": args.trunk_rff_features,
                "trunk_rff_sigma": args.trunk_rff_sigma,
                "trunk_layers": (
                    [2 * args.trunk_rff_features + 8, *args.trunk_hidden_dims, args.latent_width]
                    if args.trunk_rff_features > 0
                    else [int(X_train[1].shape[1]) + 2 * args.time_fourier_modes, *args.trunk_hidden_dims, args.latent_width]
                ),
                "use_resnet_branch": args.use_resnet_branch,
```

- [ ] **Step 6: Update Configuration print dict (line ~1336) and manifest `"configuration"` dict (line ~1515)**

In **both dicts**, after `"use_transformer_branch": args.use_transformer_branch,`, add:

```python
                "use_resnet_branch": args.use_resnet_branch,
                "trunk_rff_features": args.trunk_rff_features,
                "trunk_rff_sigma": args.trunk_rff_sigma,
```

In the manifest's `"model"` sub-dict (line ~1555), update the two `branch_encoder` lines:

```python
                "branch_encoder": (
                    "transformer" if args.use_transformer_branch
                    else "resnet" if args.use_resnet_branch
                    else "gated_mlp" if args.use_gated_mlp
                    else "mlp"
                ),
                "trunk_encoder": "rff" if args.trunk_rff_features > 0 else "time_fourier",
```

And update the manifest's `trunk_layers` line with the same conditional as Step 5.

### `evaluate_checkpoint.py` changes

- [ ] **Step 7: Update `evaluate_checkpoint.py`**

**In `parse_args()`**, after `--use-transformer-branch`, add:

```python
    parser.add_argument("--use-resnet-branch", action="store_true", default=None)
    parser.add_argument("--trunk-rff-features", type=int, default=None)
    parser.add_argument("--trunk-rff-sigma", type=float, default=None)
```

**In `main()`**, after `transformer_dropout = float(get_config_value(...))`, add:

```python
    use_resnet_branch = bool(get_config_value(args, defaults, "use_resnet_branch"))
    trunk_rff_features_value = get_config_value(args, defaults, "trunk_rff_features")
    trunk_rff_features = 0 if trunk_rff_features_value is None else int(trunk_rff_features_value)
    trunk_rff_sigma_value = get_config_value(args, defaults, "trunk_rff_sigma")
    trunk_rff_sigma = 1.0 if trunk_rff_sigma_value is None else float(trunk_rff_sigma_value)
```

**In the `print(json.dumps({...}))` diagnostic block** in `main()` (lines 164–198), after `"transformer_dropout": transformer_dropout,`, add:

```python
                "use_resnet_branch": use_resnet_branch,
                "trunk_rff_features": trunk_rff_features,
                "trunk_rff_sigma": trunk_rff_sigma,
```

**In the `create_model()` call** in `main()` (line ~209), after `transformer_dropout=transformer_dropout,`, add:

```python
        use_resnet_branch=use_resnet_branch,
        trunk_rff_features=trunk_rff_features,
        trunk_rff_sigma=trunk_rff_sigma,
```

### Verification

- [ ] **Step 8: Run full test suite**

```bash
uv run pytest tests/ -x -q 2>&1
```

Expected: all pass.

- [ ] **Step 9: Smoke-test CLI help**

```bash
uv run python -m pi_onet.train --help 2>&1 | grep -E "resnet|rff"
```

Expected:
```
  --use-resnet-branch
  --trunk-rff-features TRUNK_RFF_FEATURES
  --trunk-rff-sigma TRUNK_RFF_SIGMA
```

- [ ] **Step 10: Smoke-test backward compat with existing config**

```bash
uv run python -c "
import os; os.environ['DDE_BACKEND'] = 'pytorch'
from pi_onet.train import load_train_config
from pathlib import Path
cfg = load_train_config(Path('configs/paper_aligned_step.toml'))
print('use_resnet_branch:', cfg.use_resnet_branch)
print('trunk_rff_features:', cfg.trunk_rff_features)
print('trunk_rff_sigma:', cfg.trunk_rff_sigma)
"
```

Expected: `use_resnet_branch: False`, `trunk_rff_features: 0`, `trunk_rff_sigma: 1.0`

- [ ] **Step 11: Commit**

```bash
git add src/pi_onet/train.py src/pi_onet/evaluate_checkpoint.py
git commit -m "feat: wire use_resnet_branch and trunk_rff_features through CLI, main(), and evaluate_checkpoint"
```
