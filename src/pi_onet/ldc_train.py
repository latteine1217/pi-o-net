# src/pi_onet/ldc_train.py
"""Train PI-DeepONet on steady-state multi-Re Lid-Driven Cavity flow.

What:
    以 PI-DeepONet 訓練多 Re LDC 穩態問題（無時間維度）。
    Branch: Re_norm + 感測器讀值; Trunk: (x,y,c) with RFF + Embedding。
Why:
    作為 Kolmogorov 暫態 pipeline 的互補驗證基準，時間無關穩態問題。
"""
from __future__ import annotations

import json  # noqa: F401 - used in training loop (Task 4)
import os
import time  # noqa: F401 - used in training loop (Task 4)
import tomllib  # noqa: F401 - used in training loop (Task 4)
from pathlib import Path
from typing import Any

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from deepxde import config as dde_config

from pi_onet.train import ResNetBranchNet, SimpleMLP, configure_torch_runtime  # noqa: F401 - configure_torch_runtime used in training loop (Task 4)


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
        """What: Forward pass; branch_input [batch_Re, branch_dim], trunk_input [N_pts, 3]."""
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
