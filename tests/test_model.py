"""Tests for train.py neural network architecture components."""
from __future__ import annotations

import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np
import torch
import pytest

# All imports from train.py — declared upfront so the file structure is stable
# across all tasks. ImportError for not-yet-implemented classes (Tasks 2-3) is the
# expected TDD "red" state; those names are set to None so the module can be collected.
from pi_onet.train import FourierFeatureTrunkNet, FlattenBranchNet, SimpleMLP

try:
    from pi_onet.train import ResNetBlock
except ImportError:
    ResNetBlock = None  # type: ignore[assignment,misc]

try:
    from pi_onet.train import ResNetBranchNet
except ImportError:
    ResNetBranchNet = None  # type: ignore[assignment,misc]

try:
    from pi_onet.train import create_model
except ImportError:
    create_model = None  # type: ignore[assignment]


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
