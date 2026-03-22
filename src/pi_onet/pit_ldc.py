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


# ── Training loop ─────────────────────────────────────────────────────────────

def train_pit_ldc(args: dict[str, Any]) -> None:
    """What: Full PiT LDC training loop.

    Why: Pure PyTorch loop (no DeepXDE training infra) — steady-state LDC has
         no temporal structure compatible with DeepXDE's DataSet abstraction.
    """
    from pi_onet.ldc_dataset import LDCDataset

    device = configure_torch_runtime(args["device"])
    torch.manual_seed(args["seed"])
    rng = np.random.default_rng(args["seed"])

    artifacts_dir = Path(args["artifacts_dir"])
    checkpoints_dir = artifacts_dir / "checkpoints"
    best_dir = artifacts_dir / "best_validation"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = LDCDataset(
        mat_paths=args["data_files"],
        num_interior_sensors=args["num_interior_sensors"],
        num_boundary_sensors=args["num_boundary_sensors"],
        train_ratio=0.8,
        seed=args["seed"],
    )
    num_re = len(dataset.re_values)

    # Pre-assemble sensor tensors and re_norm_list (static across all steps)
    idx = np.concatenate([dataset.sensor_interior, dataset.sensor_boundary])
    sensors_list: list[torch.Tensor] = []
    re_norm_list: list[float] = []
    for i, g in enumerate(dataset.grids):
        s = np.stack(
            [g["x"][idx], g["y"][idx], g["u"][idx], g["v"][idx], g["p"][idx]], axis=1
        )
        sensors_list.append(torch.tensor(s, dtype=torch.float32, device=device))
        re_norm_list.append(float((dataset.re_values[i] - RE_MEAN) / RE_STD))

    # Model
    net = create_pit_model(args).to(device)

    print("=== Configuration ===")
    print(json.dumps(
        {k: v for k, v in args.items() if k != "data_files"}, indent=2, ensure_ascii=False
    ))
    print(f"trainable_parameters: {count_parameters(net)}")

    # Optimizer
    if args["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"]
        )
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args["learning_rate"])

    # LR scheduler
    scheduler = None
    if args["lr_schedule"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args["lr_step_size"], gamma=args["lr_step_gamma"]
        )

    best_val_metric = float("inf")
    best_checkpoint_path: str | None = None
    n_bc_per_wall = max(1, args["num_bc_points"] // 4)

    print("=== Training ===")
    print(f"{'Step':<8} {'L_data':>12} {'L_phys':>12} {'L_bc':>10} {'L_total':>12}")

    for step in range(1, args["iterations"] + 1):
        net.train()
        optimizer.zero_grad()

        # ── Data loss ──
        trunk_np, ref_np = dataset.sample_train_trunk(
            rng=rng, n_per_re=args["num_query_points"]
        )
        xy_data = torch.tensor(trunk_np[:, :2], dtype=torch.float32, device=device)
        c_data = torch.tensor(trunk_np[:, 2], dtype=torch.long, device=device)

        l_data = torch.zeros(1, device=device, dtype=dde_config.real(torch))
        for i in range(num_re):
            pred = net(sensors_list[i], re_norm_list[i], xy_data, c_data).squeeze(1)
            ref = torch.tensor(ref_np[i], dtype=torch.float32, device=device)
            l_data = l_data + torch.mean((pred - ref) ** 2)
        l_data = l_data / num_re

        # ── Physics loss ──
        xy_phys_np = rng.uniform(0.0, 1.0, (args["num_physics_points"], 2)).astype(np.float32)
        xy_phys = torch.tensor(xy_phys_np, device=device, requires_grad=True)

        # eval mode: disables attn_dropout noise for deterministic autograd gradients
        net.eval()
        l_ns_total = torch.zeros(1, device=device, dtype=dde_config.real(torch))
        l_cont_total = torch.zeros(1, device=device, dtype=dde_config.real(torch))
        for i in range(num_re):
            model_fn_i = make_pit_model_fn(net, sensors_list[i], re_norm_list[i], device)
            u_fn = lambda xy, fn=model_fn_i: fn(xy, c=0)
            v_fn = lambda xy, fn=model_fn_i: fn(xy, c=1)
            p_fn = lambda xy, fn=model_fn_i: fn(xy, c=2)
            ns_x, ns_y, cont = steady_ns_residuals(
                u_fn, v_fn, p_fn, xy_phys, re=dataset.re_values[i]
            )
            l_ns_total = l_ns_total + torch.mean(ns_x ** 2) + torch.mean(ns_y ** 2)
            l_cont_total = l_cont_total + torch.mean(cont ** 2)

        # BC and gauge: Re-independent BCs, use Re case 0
        model_fn_re0 = make_pit_model_fn(net, sensors_list[0], re_norm_list[0], device)
        l_bc = compute_bc_loss(model_fn=model_fn_re0, n_per_wall=n_bc_per_wall, device=device)
        l_gauge = compute_gauge_loss(model_fn=model_fn_re0, device=device)
        net.train()

        l_ns_total = l_ns_total / num_re
        l_cont_total = l_cont_total / num_re
        l_physics = l_ns_total + args["physics_continuity_weight"] * l_cont_total

        l_total = (
            args["data_loss_weight"] * l_data
            + args["physics_loss_weight"] * l_physics
            + args["bc_loss_weight"] * l_bc
            + args["gauge_loss_weight"] * l_gauge
        )
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(args["max_grad_norm"]))
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
            for pg in optimizer.param_groups:
                if pg["lr"] < args["min_learning_rate"]:
                    pg["lr"] = args["min_learning_rate"]

        if step % max(1, args["iterations"] // 10) == 0 or step == 1:
            print(
                f"{step:<8} {l_data.item():>12.4e} {l_physics.item():>12.4e}"
                f" {l_bc.item():>10.4e} {l_total.item():>12.4e}"
            )

        # ── Checkpoint + validation ──
        if args["checkpoint_period"] > 0 and step % args["checkpoint_period"] == 0:
            ckpt_path = checkpoints_dir / f"pit_ldc_step_{step}.pt"
            torch.save(net.state_dict(), str(ckpt_path))

            trunk_pts, ref_vals = dataset.sample_val_trunk_all()
            xy_val = torch.tensor(trunk_pts[:, :2], dtype=torch.float32, device=device)
            c_val = torch.tensor(trunk_pts[:, 2], dtype=torch.long, device=device)

            with torch.no_grad():
                net.eval()
                total_rel_l2 = 0.0
                for i in range(num_re):
                    pred_val = net(
                        sensors_list[i], re_norm_list[i], xy_val, c_val
                    ).squeeze(1)
                    ref_t = torch.tensor(ref_vals[i], dtype=torch.float32, device=device)
                    rel_l2_i = (
                        torch.norm(pred_val - ref_t) / (torch.norm(ref_t) + 1e-8)
                    ).item()
                    total_rel_l2 += rel_l2_i
                mean_rel_l2 = total_rel_l2 / num_re
                net.train()

            print(f"  [val @ {step}] mean_rel_l2 = {mean_rel_l2:.4f}")

            if mean_rel_l2 < best_val_metric:
                best_val_metric = mean_rel_l2
                best_path = best_dir / "pit_ldc_best.pt"
                torch.save(net.state_dict(), str(best_path))
                best_checkpoint_path = str(best_path)
                write_json(best_dir / "best_validation_summary.json", {
                    "step": step, "val_mean_rel_l2": mean_rel_l2,
                })

    final_path = artifacts_dir / "pit_ldc_final.pt"
    torch.save(net.state_dict(), str(final_path))
    write_json(artifacts_dir / "experiment_manifest.json", {
        "configuration": {k: v for k, v in args.items() if k != "data_files"},
        "best_checkpoint": best_checkpoint_path,
        "final_checkpoint": str(final_path),
        "final_val_metric": best_val_metric,
    })
    print(f"=== Done. Best val rel_L2 = {best_val_metric:.4f} ===")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """What: Entry point for pit_ldc CLI."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Train PiT Cross-Attention Operator on steady-state LDC flow."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    cli_args = parser.parse_args()

    config = dict(DEFAULT_PIT_ARGS)
    config.update(load_pit_config(cli_args.config))
    if cli_args.device is not None:
        config["device"] = cli_args.device

    train_pit_ldc(config)


if __name__ == "__main__":
    main()
