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
