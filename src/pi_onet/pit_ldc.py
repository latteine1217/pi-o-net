# src/pi_onet/pit_ldc.py
"""PiT Cross-Attention Operator for LDC multi-Re steady-state flow.

What: Physics-informed Transformer operator — replaces branch-trunk dot-product
      with cross-attention from query points over sensor token sequence.
Why:  Cross-attention makes information gain spatially adaptive; each query point
      attends only to nearby / relevant sensors rather than weighting all equally.
"""
from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
from deepxde import config as dde_config

from pi_onet.ldc_train import (
    steady_ns_residuals,
    compute_bc_loss,
    compute_gauge_loss,
    count_parameters,
    write_json,
)
from pi_onet.train import configure_torch_runtime
from pi_onet.ldc_dataset import RE_MEAN, RE_STD


# ── RFF helper ────────────────────────────────────────────────────────────────

def rff_encode(z: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """What: Random Fourier Feature encoding of 2-D spatial coordinates.

    Why: Provides a rich frequency basis so the model can represent fine-grained
         spatial patterns without needing very deep networks.
    z: [N, 2], B: [2, rff_features] (non-trainable buffer)
    Returns: [N, 2*rff_features] = [sin(2π·z·B), cos(2π·z·B)]
    """
    proj = 2.0 * torch.pi * z @ B          # [N, rff_features]
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)


# ── Model components ──────────────────────────────────────────────────────────

class SensorEncoder(nn.Module):
    """What: Encode sensor tokens [RFF(x,y), u, v, p] + Re token via self-attention.

    Why: Self-attention over sensors lets each sensor aggregate spatial context
         before acting as keys/values in the downstream cross-attention.
    """

    def __init__(
        self,
        rff_features: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_encoder_layers: int,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        sensor_input_dim = 2 * rff_features + 3   # rff(x,y) + u, v, p
        self.sensor_proj = nn.Linear(sensor_input_dim, d_model)
        self.re_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(
        self,
        sensors: torch.Tensor,
        re_norm: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """
        sensors: [N_s, 5] = [x_s, y_s, u_s, v_s, p_s]
        re_norm: [1, 1] normalised Re value
        B: [2, rff_features] shared RFF matrix
        Returns: [N_s+1, d_model]
        """
        rff_s = rff_encode(sensors[:, :2], B)                       # [N_s, 2*rff_features]
        token_input = torch.cat([rff_s, sensors[:, 2:]], dim=1)     # [N_s, 2*rff_features+3]
        tokens = self.sensor_proj(token_input)                       # [N_s, d_model]
        re_token = self.re_proj(re_norm)                             # [1, d_model]
        tokens = torch.cat([re_token, tokens], dim=0)                # [N_s+1, d_model]
        tokens = tokens.unsqueeze(0)                                 # [1, N_s+1, d_model]
        out = self.encoder(tokens)                                   # [1, N_s+1, d_model]
        return out.squeeze(0)                                        # [N_s+1, d_model]


class QueryEncoder(nn.Module):
    """What: Encode query points (x, y) with shared RFF and component index c via Embedding.

    Why: Shared RFF matrix B with SensorEncoder ensures spatial frequency alignment
         between keys/values and queries — critical for meaningful cross-attention.
    """

    def __init__(self, rff_features: int, d_model: int) -> None:
        super().__init__()
        query_input_dim = 2 * rff_features + 8   # rff(x,y) + embed(c,8)
        self.component_emb = nn.Embedding(3, 8)
        nn.init.normal_(self.component_emb.weight, mean=0.0, std=0.1)
        self.query_proj = nn.Linear(query_input_dim, d_model)

    def forward(
        self,
        xy: torch.Tensor,
        c: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """
        xy: [N_q, 2]
        c: [N_q] long tensor (0=u, 1=v, 2=p)
        B: [2, rff_features] shared RFF matrix
        Returns: [N_q, d_model]
        """
        rff_q = rff_encode(xy, B)                                   # [N_q, 2*rff_features]
        emb_c = self.component_emb(c)                               # [N_q, 8]
        q_input = torch.cat([rff_q, emb_c], dim=1)                 # [N_q, 2*rff_features+8]
        return self.query_proj(q_input)                             # [N_q, d_model]


class CrossAttentionOperator(nn.Module):
    """What: PiT operator — sensor tokens as K/V, query points as Q, cross-attention output.

    Why: Each query point's prediction is a learned weighted average over all sensor
         tokens, with weights determined by (query, sensor) spatial relationship.
         This is architecturally distinct from DeepONet's branch@trunk.T.
    """

    def __init__(
        self,
        rff_features: int,
        rff_sigma: float,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_encoder_layers: int,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        if rff_features <= 0:
            raise ValueError("rff_features 必須為正整數。")
        B = torch.randn(2, rff_features) * rff_sigma
        self.register_buffer("B", B)

        self.sensor_encoder = SensorEncoder(
            rff_features=rff_features,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            attn_dropout=attn_dropout,
        )
        self.query_encoder = QueryEncoder(rff_features=rff_features, d_model=d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1, bias=True)
        # Per-component learnable scale/bias — handles u/v/p magnitude disparity
        self.component_scale = nn.Parameter(torch.ones(3))
        self.component_bias = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        sensors: torch.Tensor,
        re_norm: float,
        xy: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        sensors: [N_s, 5]
        re_norm: Python float — converted to Tensor[1,1] here
        xy: [N_q, 2], may have requires_grad=True for physics loss
        c: [N_q] long tensor
        Returns: [N_q, 1]
        """
        re_norm_t = torch.tensor([[re_norm]], dtype=torch.float32, device=sensors.device)
        kv = self.sensor_encoder(sensors, re_norm_t, self.B)         # [N_s+1, d_model]
        q = self.query_encoder(xy, c, self.B)                        # [N_q, d_model]

        kv = kv.unsqueeze(0)                                         # [1, N_s+1, d_model]
        q = q.unsqueeze(0)                                           # [1, N_q, d_model]
        attn_out, _ = self.cross_attn(q, kv, kv)                    # [1, N_q, d_model]
        attn_out = attn_out.squeeze(0)                               # [N_q, d_model]

        feat = self.norm(attn_out)
        out = self.output_head(feat)                                 # [N_q, 1]
        out = out * self.component_scale[c].unsqueeze(1) + self.component_bias[c].unsqueeze(1)
        return out


def create_pit_model(cfg: dict) -> CrossAttentionOperator:
    """What: Factory — build CrossAttentionOperator from config dict."""
    return CrossAttentionOperator(
        rff_features=int(cfg["rff_features"]),
        rff_sigma=float(cfg["rff_sigma"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        num_encoder_layers=int(cfg["num_encoder_layers"]),
        attn_dropout=float(cfg["attn_dropout"]),
    )


# ── Physics utilities ─────────────────────────────────────────────────────────

def make_pit_model_fn(
    net: CrossAttentionOperator,
    sensors_t: torch.Tensor,
    re_norm: float,
    device: torch.device,
) -> Callable:
    """What: Return a closure (xy, c) -> [N,1] for one Re case.

    Why: Physics/BC/gauge losses call model_fn(xy, c=0/1/2) with keyword arg c.
         Closure captures sensors and re_norm for a specific Re case.
    re_norm: Python float (normalised Re value).

    Note: net and sensors_t may reside on a different device than xy (e.g. when
    deepxde sets the default device to MPS but compute_bc_loss builds CPU tensors).
    We resolve the target device from the net's buffer B to ensure consistency.
    """
    net_device = next(iter(net.buffers())).device

    def model_fn(xy: torch.Tensor, c: int) -> torch.Tensor:
        src_device = xy.device
        xy_d = xy.to(net_device)
        out = net(
            sensors_t, re_norm, xy_d,
            torch.full((xy_d.shape[0],), c, dtype=torch.long, device=net_device),
        )
        return out.to(src_device)
    return model_fn


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_PIT_ARGS: dict[str, Any] = {
    "data_files": None,
    "num_interior_sensors": 80,
    "num_boundary_sensors": 20,
    "num_query_points": 2048,
    "num_physics_points": 1024,
    "num_bc_points": 100,
    "d_model": 128,
    "nhead": 4,
    "num_encoder_layers": 2,
    "dim_feedforward": 256,
    "attn_dropout": 0.0,
    "rff_features": 64,
    "rff_sigma": 5.0,
    "data_loss_weight": 1.0,
    "physics_loss_weight": 0.1,
    "physics_continuity_weight": 1.0,
    "bc_loss_weight": 1.0,
    "gauge_loss_weight": 1.0,
    "iterations": 10000,
    "batch_size": 3,  # TODO: currently unused — all Re cases processed together per step
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_schedule": "step",
    "lr_step_size": 5000,
    "lr_step_gamma": 0.5,
    "min_learning_rate": 1e-6,
    "max_grad_norm": 1.0,
    "checkpoint_period": 2000,
    "seed": 42,
    "device": "auto",
    "artifacts_dir": "../artifacts/pit-ldc",
}


def load_pit_config(config_path: Path | None) -> dict[str, Any]:
    """What: Load and validate TOML config against DEFAULT_PIT_ARGS.

    Why: Single entry point for config parsing; catches unsupported keys early
         and resolves relative paths against the config file location.
    """
    if config_path is None:
        return {}
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config_data = payload.get("train", payload)
    normalized = dict(config_data)
    unknown = sorted(set(normalized) - set(DEFAULT_PIT_ARGS))
    if unknown:
        raise ValueError(f"PiT config 含有不支援的欄位: {unknown}")
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
