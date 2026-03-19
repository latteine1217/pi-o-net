"""Train physics-informed DeepONet aligned with arXiv:2103.10974.

What:
    以文獻對齊流程訓練 PI-DeepONet：branch 輸入初始條件 + Re，trunk 輸入 (x,y,t,c)，
    損失函數使用 L_data + L_physics。
Why:
    專案已全面遷移為文獻導向版本，不保留舊有單步回歸 / Fourier trunk / GradNorm / L-BFGS 流程。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import tomllib
from pathlib import Path
from typing import Any

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import deepxde as dde
import numpy as np
import torch
from deepxde import config as dde_config
from deepxde.nn import activations, initializers

from pi_onet.dataset import (
    DatasetConfig,
    PhysicsInformedTripleCartesianProd,
    PreparedDataset,
    build_dataset,
    resolve_data_files,
)


DEFAULT_TRAIN_ARGS: dict[str, Any] = {
    "data_file": None,
    "field": "uvp",
    "num_sensors": 50,
    "history_steps": 1,
    "horizon_steps": 10,
    "temporal_stride": 1,
    "burn_in_steps": 100,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "physics_stride": 32,
    "data_loss_weight": 20.0,
    "physics_loss_weight": 1.0,
    "physics_time_samples": 4,
    "physics_branch_batch_size": None,
    "physics_continuity_weight": 10.0,
    "physics_causal_epsilon": 1.0,
    "time_fourier_modes": 8,
    "iterations": 20000,
    "batch_size": 32,
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "lr_warmup_steps": 0,
    "lr_warmup_start_factor": 0.1,
    "weight_decay": 1e-4,
    "lr_schedule": "step",
    "min_learning_rate": 1e-6,
    "lr_step_size": 10000,
    "lr_step_gamma": 0.5,
    "rollout_steps": 10,
    "checkpoint_period": 5000,
    "branch_hidden_dims": [512, 512],
    "trunk_hidden_dims": [512, 512],
    "latent_width": 256,
    "use_gated_mlp": True,
    "use_transformer_branch": False,
    "use_resnet_branch": False,
    "trunk_rff_features": 0,
    "trunk_rff_sigma": 1.0,
    "transformer_model_dim": 128,
    "transformer_num_heads": 4,
    "transformer_num_layers": 2,
    "transformer_ff_dim": 256,
    "transformer_dropout": 0.0,
    "early_stop_total_loss": 1e-4,
    "seed": 42,
    "device": "auto",
    "artifacts_dir": Path("artifacts"),
}

PASS_FAIL_THRESHOLDS: dict[str, float] = {
    "validation_mean_relative_l2": 0.2,
    "test_mean_relative_l2": 0.2,
    "rollout_sensor_relative_l2_mean": 0.3,
    "rollout_physics_relative_l2_mean": 0.4,
}


def resolve_torch_device(device_preference: str) -> torch.device:
    """What: 解析使用者指定的裝置偏好並回傳可用裝置。"""

    preference = device_preference.lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("指定 --device cuda，但目前環境沒有可用 CUDA。")
        return torch.device("cuda")
    if preference == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("指定 --device mps，但目前環境沒有可用 Metal (MPS)。")
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    raise ValueError(f"不支援的 device: {device_preference}")


def configure_torch_runtime(device_preference: str) -> torch.device:
    """What: 啟用 PyTorch 執行環境並回傳實際使用裝置。"""

    torch.set_float32_matmul_precision("high")
    device = resolve_torch_device(device_preference)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


def _resolve_config_path(base_dir: Path, value: str) -> Path:
    """What: 將 config 相對路徑解析為相對於 config 所在目錄。"""

    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def load_train_config(config_path: Path | None) -> dict[str, Any]:
    """What: 載入 TOML 設定檔。"""

    if config_path is None:
        return {}

    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config_data = payload.get("train", payload)
    if not isinstance(config_data, dict):
        raise ValueError("訓練 config 必須是 TOML table。")

    normalized = dict(config_data)
    unknown_keys = sorted(set(normalized) - set(DEFAULT_TRAIN_ARGS))
    if unknown_keys:
        raise ValueError(f"訓練 config 含有不支援的欄位: {unknown_keys}")

    if "data_file" in normalized:
        data_file = normalized["data_file"]
        if isinstance(data_file, str):
            normalized["data_file"] = [str(_resolve_config_path(config_path.parent, data_file))]
        elif isinstance(data_file, list):
            normalized["data_file"] = [str(_resolve_config_path(config_path.parent, str(path))) for path in data_file]
        else:
            raise ValueError("data_file 必須是字串或字串陣列。")
    if "artifacts_dir" in normalized:
        normalized["artifacts_dir"] = _resolve_config_path(config_path.parent, str(normalized["artifacts_dir"]))
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    """What: 建立文獻對齊版訓練 CLI。"""

    parser = argparse.ArgumentParser(description="Train literature-aligned PI-DeepONet on Kolmogorov DNS.")
    parser.add_argument("--config", type=Path, default=None, help="TOML 設定檔路徑。")
    parser.add_argument("--data-file", action="append", default=None, help="可重複指定 DNS `.npy`。")
    parser.add_argument("--field", type=str, choices=["uvp"], default=None)
    parser.add_argument("--num-sensors", type=int, default=None)
    parser.add_argument("--history-steps", type=int, default=None)
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--temporal-stride", type=int, default=None)
    parser.add_argument("--burn-in-steps", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--physics-stride", type=int, default=None)
    parser.add_argument("--data-loss-weight", type=float, default=None)
    parser.add_argument("--physics-loss-weight", type=float, default=None)
    parser.add_argument("--physics-time-samples", type=int, default=None)
    parser.add_argument("--physics-branch-batch-size", type=int, default=None)
    parser.add_argument("--physics-continuity-weight", type=float, default=None)
    parser.add_argument("--physics-causal-epsilon", type=float, default=None)
    parser.add_argument("--time-fourier-modes", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    parser.add_argument("--lr-warmup-start-factor", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=["none", "cosine", "step"], default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--lr-step-size", type=int, default=None)
    parser.add_argument("--lr-step-gamma", type=float, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--checkpoint-period", type=int, default=None)
    parser.add_argument("--branch-hidden-dims", type=str, default=None, help="例如: 512,512")
    parser.add_argument("--trunk-hidden-dims", type=str, default=None, help="例如: 512,512")
    parser.add_argument("--latent-width", type=int, default=None)
    parser.add_argument("--use-gated-mlp", action="store_true", default=None)
    parser.add_argument("--use-transformer-branch", action="store_true", default=None)
    parser.add_argument("--use-resnet-branch", action="store_true", default=None)
    parser.add_argument("--trunk-rff-features", type=int, default=None)
    parser.add_argument("--trunk-rff-sigma", type=float, default=None)
    parser.add_argument("--transformer-model-dim", type=int, default=None)
    parser.add_argument("--transformer-num-heads", type=int, default=None)
    parser.add_argument("--transformer-num-layers", type=int, default=None)
    parser.add_argument("--transformer-ff-dim", type=int, default=None)
    parser.add_argument("--transformer-dropout", type=float, default=None)
    parser.add_argument("--early-stop-total-loss", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    return parser


def parse_hidden_dims(value: Any, field_name: str) -> list[int]:
    """What: 將 hidden dims 正規化成整數列表。"""

    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        dims = [int(token) for token in tokens]
    elif isinstance(value, list):
        dims = [int(item) for item in value]
    else:
        raise ValueError(f"{field_name} 必須是字串或整數陣列。")

    if len(dims) == 0:
        raise ValueError(f"{field_name} 至少要有一層 hidden layer。")
    if any(dimension <= 0 for dimension in dims):
        raise ValueError(f"{field_name} 每一層維度都必須大於 0。")
    return dims


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """What: 合併 default / config / CLI 參數。"""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config_defaults = load_train_config(args.config)

    merged = dict(DEFAULT_TRAIN_ARGS)
    merged.update(config_defaults)
    merged.update({key: value for key, value in vars(args).items() if key != "config" and value is not None})
    merged["branch_hidden_dims"] = parse_hidden_dims(merged["branch_hidden_dims"], "branch_hidden_dims")
    merged["trunk_hidden_dims"] = parse_hidden_dims(merged["trunk_hidden_dims"], "trunk_hidden_dims")
    merged["latent_width"] = int(merged["latent_width"])
    merged["config"] = args.config
    if merged["latent_width"] <= 0:
        raise ValueError("latent_width 必須大於 0。")
    if int(merged["history_steps"]) <= 0:
        raise ValueError("history_steps 必須大於 0。")
    if merged["data_loss_weight"] <= 0.0:
        raise ValueError("data_loss_weight 必須大於 0。")
    if merged["physics_loss_weight"] < 0.0:
        raise ValueError("physics_loss_weight 不可小於 0。")
    if merged["physics_continuity_weight"] <= 0.0:
        raise ValueError("physics_continuity_weight 必須大於 0。")
    if merged["physics_causal_epsilon"] < 0.0:
        raise ValueError("physics_causal_epsilon 不可小於 0。")
    if int(merged["time_fourier_modes"]) < 0:
        raise ValueError("time_fourier_modes 不可小於 0。")
    if int(merged["trunk_rff_features"]) < 0:
        raise ValueError("trunk_rff_features 不可小於 0。")
    if float(merged["trunk_rff_sigma"]) <= 0.0:
        raise ValueError("trunk_rff_sigma 必須為正數。")
    if int(merged["lr_warmup_steps"]) < 0:
        raise ValueError("lr_warmup_steps 不可小於 0。")
    if not (0.0 < float(merged["lr_warmup_start_factor"]) <= 1.0):
        raise ValueError("lr_warmup_start_factor 必須介於 (0, 1]。")
    if merged["physics_branch_batch_size"] is not None and int(merged["physics_branch_batch_size"]) <= 0:
        raise ValueError("physics_branch_batch_size 必須大於 0。")
    if bool(merged["use_transformer_branch"]):
        if int(merged["transformer_model_dim"]) <= 0:
            raise ValueError("transformer_model_dim 必須大於 0。")
        if int(merged["transformer_num_heads"]) <= 0:
            raise ValueError("transformer_num_heads 必須大於 0。")
        if int(merged["transformer_num_layers"]) <= 0:
            raise ValueError("transformer_num_layers 必須大於 0。")
        if int(merged["transformer_ff_dim"]) <= 0:
            raise ValueError("transformer_ff_dim 必須大於 0。")
        if not (0.0 <= float(merged["transformer_dropout"]) < 1.0):
            raise ValueError("transformer_dropout 必須介於 [0, 1)。")
        if int(merged["transformer_model_dim"]) % int(merged["transformer_num_heads"]) != 0:
            raise ValueError("transformer_model_dim 必須能被 transformer_num_heads 整除。")
    return argparse.Namespace(**merged)


class ModifiedGatedMLP(torch.nn.Module):
    """What: Appendix G 的 modified MLP gate 架構。"""

    def __init__(self, layer_sizes: list[int], activation: str, kernel_initializer: str) -> None:
        super().__init__()
        if len(layer_sizes) < 3:
            raise ValueError("use_gated_mlp 需要至少一層 hidden layer。")
        hidden_sizes = layer_sizes[1:-1]
        if len(set(hidden_sizes)) != 1:
            raise ValueError("use_gated_mlp 目前要求 hidden dims 相同，例如 512,512。")

        input_dim = int(layer_sizes[0])
        hidden_dim = int(hidden_sizes[0])
        output_dim = int(layer_sizes[-1])
        num_hidden_layers = len(hidden_sizes)

        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        def make_linear(in_dim: int, out_dim: int) -> torch.nn.Linear:
            linear = torch.nn.Linear(in_dim, out_dim, dtype=dde_config.real(torch))
            initializer(linear.weight)
            initializer_zero(linear.bias)
            return linear

        self.u_linear = make_linear(input_dim, hidden_dim)
        self.v_linear = make_linear(input_dim, hidden_dim)
        self.h0_linear = make_linear(input_dim, hidden_dim)
        self.z_layers = torch.nn.ModuleList([make_linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.output_linear = make_linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        u = self.activation(self.u_linear(inputs))
        v = self.activation(self.v_linear(inputs))
        h = self.activation(self.h0_linear(inputs))
        for z_layer in self.z_layers:
            z = self.activation(z_layer(h))
            h = (1.0 - z) * u + z * v
        return self.output_linear(h)


class SimpleMLP(torch.nn.Module):
    """What: 簡化版 MLP（tanh）供非 gated trunk 使用。"""

    def __init__(self, layer_sizes: list[int], activation: str, kernel_initializer: str) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("SimpleMLP 需要至少 input/output 兩層。")
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        layers: list[torch.nn.Linear] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            linear = torch.nn.Linear(int(in_dim), int(out_dim), dtype=dde_config.real(torch))
            initializer(linear.weight)
            initializer_zero(linear.bias)
            layers.append(linear)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = inputs
        for index, layer in enumerate(self.layers):
            h = layer(h)
            if index < len(self.layers) - 1:
                h = self.activation(h)
        return h


class FlattenBranchNet(torch.nn.Module):
    """What: 將歷史序列 branch 攤平成單向量後交給既有 branch encoder。"""

    def __init__(self, core_net: torch.nn.Module) -> None:
        super().__init__()
        self.core_net = core_net

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            flat = inputs
        elif inputs.ndim == 3:
            flat = inputs.reshape(inputs.shape[0], -1)
        else:
            raise ValueError("FlattenBranchNet 只支援 2D 或 3D branch tensor。")
        return self.core_net(flat)


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


class TemporalTransformerBranch(torch.nn.Module):
    """What: 將 branch 的時間窗視為 token 序列，使用 Transformer encoder 抽取時序表徵。"""

    def __init__(
        self,
        token_dim: int,
        sequence_length: int,
        latent_width: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or sequence_length <= 0:
            raise ValueError("TemporalTransformerBranch 需要正的 token_dim 與 sequence_length。")
        if model_dim % num_heads != 0:
            raise ValueError("Transformer model_dim 必須能被 num_heads 整除。")

        self.token_dim = int(token_dim)
        self.sequence_length = int(sequence_length)
        self.model_dim = int(model_dim)

        self.input_proj = torch.nn.Linear(self.token_dim, self.model_dim, dtype=dde_config.real(torch))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.model_dim, dtype=dde_config.real(torch)))
        self.positional_embedding = torch.nn.Parameter(
            torch.zeros(1, self.sequence_length + 1, self.model_dim, dtype=dde_config.real(torch))
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
            dtype=dde_config.real(torch),
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.output_norm = torch.nn.LayerNorm(self.model_dim, dtype=dde_config.real(torch))
        self.output_proj = torch.nn.Linear(self.model_dim, int(latent_width), dtype=dde_config.real(torch))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.input_proj.weight)
        torch.nn.init.zeros_(self.input_proj.bias)
        torch.nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)
        torch.nn.init.xavier_uniform_(self.output_proj.weight)
        torch.nn.init.zeros_(self.output_proj.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3:
            raise ValueError("TemporalTransformerBranch 只支援 [batch, seq, dim] branch tensor。")
        if inputs.shape[1] != self.sequence_length:
            raise ValueError(
                f"TemporalTransformerBranch 預期 sequence_length={self.sequence_length}，但收到 {inputs.shape[1]}。"
            )
        if inputs.shape[2] != self.token_dim:
            raise ValueError(f"TemporalTransformerBranch 預期 token_dim={self.token_dim}，但收到 {inputs.shape[2]}。")

        tokens = self.input_proj(inputs)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = torch.cat([cls, tokens], dim=1) + self.positional_embedding
        encoded = self.encoder(encoded)
        pooled = self.output_norm(encoded[:, 0, :])
        return self.output_proj(pooled)


class TimeFourierTrunkNet(torch.nn.Module):
    """What: 將 trunk 的時間維度做 Fourier 特徵編碼後交給底層 MLP。"""

    def __init__(self, num_modes: int, core_net: torch.nn.Module) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        if self.num_modes < 0:
            raise ValueError("num_modes 不可小於 0。")
        self.core_net = core_net
        if self.num_modes > 0:
            frequencies = torch.arange(1, self.num_modes + 1, dtype=torch.float32)
            self.register_buffer("frequencies", frequencies)
        else:
            self.register_buffer("frequencies", torch.zeros((0,), dtype=torch.float32))

    def encoded_dim(self, input_dim: int) -> int:
        if input_dim != 4:
            raise ValueError("目前 trunk raw input 必須為 4 維 `(x,y,t,c)`。")
        return input_dim + 2 * self.num_modes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.num_modes <= 0:
            return self.core_net(inputs)
        if inputs.shape[1] != 4:
            raise ValueError("TimeFourierTrunkNet 目前僅支援 `(x,y,t,c)`。")

        x = inputs[:, 0:1]
        y = inputs[:, 1:2]
        t = inputs[:, 2:3]
        c = inputs[:, 3:4]
        omega_t = 2.0 * np.pi * t * self.frequencies.to(device=inputs.device, dtype=inputs.dtype).unsqueeze(0)
        encoded = torch.cat([x, y, t, torch.sin(omega_t), torch.cos(omega_t), c], dim=1)
        return self.core_net(encoded)


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


class CallableTrunkDeepONetCartesianProd(dde.nn.pytorch.deeponet.DeepONetCartesianProd):
    """What: 讓 trunk 支援 callable module。"""

    def build_trunk_net(self, layer_sizes_trunk):
        if len(layer_sizes_trunk) > 1 and callable(layer_sizes_trunk[1]):
            return layer_sizes_trunk[1]
        return super().build_trunk_net(layer_sizes_trunk)

    def forward(self, inputs):
        """What: 保證 branch/trunk 輸入與模型參數位於相同裝置。"""

        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype
        x_func, x_loc = inputs
        if not torch.is_tensor(x_func):
            x_func = torch.as_tensor(x_func, dtype=dtype, device=device)
        else:
            x_func = x_func.to(device=device, dtype=dtype)
        if not torch.is_tensor(x_loc):
            x_loc = torch.as_tensor(x_loc, dtype=dtype, device=device)
        else:
            x_loc = x_loc.to(device=device, dtype=dtype)
        return super().forward((x_func, x_loc))


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


def count_trainable_parameters(net: torch.nn.Module) -> int:
    """What: 計算可訓練參數數量。"""

    return int(sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad))


def build_optimizer(
    net: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> str | torch.optim.Optimizer:
    """What: 建立 optimizer 設定。"""

    if optimizer_name == "adam":
        return "adam"
    if optimizer_name == "adamw":
        if weight_decay <= 0.0:
            raise ValueError("使用 AdamW 時，weight_decay 必須大於 0。")
        return torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"不支援的 optimizer: {optimizer_name}")


def build_lr_decay_config(
    schedule: str,
    iterations: int,
    min_learning_rate: float,
    step_size: int = 10000,
    step_gamma: float = 0.5,
) -> tuple[str, int, float] | None:
    """What: 建立 DeepXDE 可接受的 lr decay 設定。"""

    if schedule == "none":
        return None
    if schedule == "cosine":
        if iterations <= 0:
            raise ValueError("iterations 必須大於 0，才能使用 cosine scheduler。")
        return ("cosine", iterations, min_learning_rate)
    if schedule == "step":
        if step_size <= 0:
            raise ValueError("lr_step_size 必須大於 0。")
        if not (0.0 < step_gamma < 1.0):
            raise ValueError("lr_step_gamma 必須介於 0 與 1。")
        return ("step", step_size, step_gamma)
    raise ValueError(f"不支援的 lr schedule: {schedule}")


def compile_model(
    model: dde.Model,
    optimizer: str | torch.optim.Optimizer,
    learning_rate: float | None,
    lr_decay: tuple[str, int, float] | None,
    loss_weights: list[float],
) -> None:
    """What: 以統一設定編譯 model。"""

    model.compile(
        optimizer,
        lr=learning_rate,
        decay=lr_decay,
        loss="MSE",
        loss_weights=loss_weights,
        metrics=["mean l2 relative error"],
    )


def train_model(
    model: dde.Model,
    args: argparse.Namespace,
    callbacks: list[dde.callbacks.Callback],
) -> tuple[dde.model.LossHistory, dde.model.TrainState, dict[str, float]]:
    """What: 單階段訓練（文獻對齊）。"""

    loss_weights = [args.data_loss_weight, args.physics_loss_weight]
    decay = build_lr_decay_config(
        schedule=args.lr_schedule,
        iterations=args.iterations,
        min_learning_rate=args.min_learning_rate,
        step_size=args.lr_step_size,
        step_gamma=args.lr_step_gamma,
    )
    optimizer = build_optimizer(
        net=model.net,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    compile_model(
        model=model,
        optimizer=optimizer,
        learning_rate=None if isinstance(optimizer, torch.optim.Optimizer) else args.learning_rate,
        lr_decay=decay,
        loss_weights=loss_weights,
    )
    stage_start = time.perf_counter()
    losshistory, train_state = model.train(
        iterations=args.iterations,
        batch_size=args.batch_size,
        display_every=max(1, args.iterations // 10),
        callbacks=callbacks,
    )
    stage_end = time.perf_counter()
    return losshistory, train_state, {"stage1_seconds": float(stage_end - stage_start)}


def save_training_history(
    losshistory: dde.model.LossHistory,
    train_state: dde.model.TrainState,
    artifacts_dir: Path,
    evaluation_metric: float,
    training_wall_time_seconds: float,
) -> None:
    """What: 輸出訓練歷史與摘要。"""

    np.savez(
        artifacts_dir / "training_history.npz",
        steps=np.asarray(losshistory.steps, dtype=np.int64),
        loss_train=np.asarray(losshistory.loss_train, dtype=np.float32),
        loss_test=np.asarray(losshistory.loss_test, dtype=np.float32),
        metrics_test=np.asarray(losshistory.metrics_test, dtype=np.float32),
    )
    summary = {
        "best_step": int(train_state.best_step),
        "best_loss_train": np.asarray(train_state.best_loss_train).astype(float).tolist(),
        "best_loss_test": np.asarray(train_state.best_loss_test).astype(float).tolist(),
        "best_metrics": np.asarray(train_state.best_metrics).astype(float).tolist(),
        "test_mean_relative_l2": evaluation_metric,
        "training_wall_time_seconds": float(training_wall_time_seconds),
        "seconds_per_iteration": float(training_wall_time_seconds / max(1, int(train_state.step))),
    }
    (artifacts_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def predict_raw(
    model: dde.Model,
    X: tuple[np.ndarray, np.ndarray],
    target_means: np.ndarray,
    target_stds: np.ndarray,
) -> np.ndarray:
    """What: 將標準化輸出還原為物理量（per-component）。

    Why:
        trunk 座標的 component 欄位決定每段輸出對應哪個物理量（u/v/p），
        因此需要用各自的 mean/std 做反正規化，而非共用全局統計。
    """

    pred_norm = model.predict(X)  # (N, 3*n)
    n = pred_norm.shape[1] // 3
    pred = np.empty_like(pred_norm)
    for c, (m, s) in enumerate(zip(target_means, target_stds)):
        pred[:, c * n : (c + 1) * n] = pred_norm[:, c * n : (c + 1) * n] * s + m
    return pred


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    """What: 計算平均相對 L2 誤差。"""

    numerator = np.linalg.norm(prediction - target, axis=1)
    denominator = np.linalg.norm(target, axis=1) + 1e-8
    return float(np.mean(numerator / denominator))


def build_component_trunk_coords(coords_xy: np.ndarray, time_value: float) -> np.ndarray:
    """What: 將 `(x,y)` 與時間擴成 `(x,y,t,c)`，component `c in {0,1,2}`。"""

    xy = np.asarray(coords_xy, dtype=np.float32)
    n_points = int(xy.shape[0])
    t_col = np.full((n_points, 1), np.float32(time_value), dtype=np.float32)
    base = np.concatenate([xy, t_col], axis=1)
    chunks: list[np.ndarray] = []
    for comp in (0.0, 1.0, 2.0):
        c_col = np.full((n_points, 1), np.float32(comp), dtype=np.float32)
        chunks.append(np.concatenate([base, c_col], axis=1))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def evaluate(
    model: dde.Model,
    X: tuple[np.ndarray, np.ndarray],
    y: np.ndarray,
    target_means: np.ndarray,
    target_stds: np.ndarray,
) -> float:
    """What: 在指定資料集上評估平均相對 L2（per-component 反正規化）。"""

    prediction = predict_raw(model, X, target_means, target_stds)
    n = y.shape[1] // 3
    target = np.empty_like(y)
    for c, (m, s) in enumerate(zip(target_means, target_stds)):
        target[:, c * n : (c + 1) * n] = y[:, c * n : (c + 1) * n] * s + m
    return relative_l2(prediction, target)


def compute_unweighted_losses(
    model: dde.Model,
    data: PhysicsInformedTripleCartesianProd,
    X: tuple[np.ndarray, np.ndarray],
    y: np.ndarray,
    delta_t: np.ndarray,
    nu: np.ndarray,
    forcing_u: np.ndarray,
    forcing_v: np.ndarray,
) -> dict[str, float]:
    """What: 計算未加權 L_data 與 L_physics。"""

    parameter = next(model.net.parameters())
    branch_inputs = torch.as_tensor(X[0], dtype=parameter.dtype, device=parameter.device)
    trunk_inputs = torch.as_tensor(X[1], dtype=parameter.dtype, device=parameter.device)
    targets = torch.as_tensor(y, dtype=parameter.dtype, device=parameter.device)

    with torch.no_grad():
        pred_norm = model.net((branch_inputs, trunk_inputs))
        supervised_loss = torch.mean((pred_norm - targets) ** 2)
    cached = data._ensure_tensor_cache(parameter.device, parameter.dtype)
    grid_size = data.physics_grid_shape[0]
    physics_output = data._physics_loss(
        model=model,
        branch_inputs=branch_inputs,
        physics_coords_xy=cached["physics_coords_xy"],
        delta_t_tensor=torch.as_tensor(delta_t, dtype=parameter.dtype, device=parameter.device).reshape(-1),
        nu_tensor=torch.as_tensor(nu, dtype=parameter.dtype, device=parameter.device).reshape(-1),
        forcing_u_tensor=torch.as_tensor(forcing_u, dtype=parameter.dtype, device=parameter.device).reshape(
            -1, grid_size, grid_size
        ),
        forcing_v_tensor=torch.as_tensor(forcing_v, dtype=parameter.dtype, device=parameter.device).reshape(
            -1, grid_size, grid_size
        ),
        return_components=True,
    )
    if not isinstance(physics_output, tuple):
        raise RuntimeError("預期 _physics_loss(return_components=True) 回傳 (total, components)。")
    physics_loss, physics_components = physics_output
    return {
        "data_mse": float(supervised_loss.detach().cpu().item()),
        "physics_residual_mse": float(physics_loss.detach().cpu().item()),
        **physics_components,
    }


def split_uvp(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What: 將 flatten field 拆成 `u,v,p` 三分量。"""

    total_dim = int(field.shape[1])
    if total_dim % 3 != 0:
        raise ValueError(f"field 維度 {total_dim} 不能被 3 整除，無法拆成 u/v/p。")
    points = total_dim // 3
    u = field[:, :points]
    v = field[:, points : 2 * points]
    p = field[:, 2 * points :]
    return u, v, p


def kinetic_energy(field: np.ndarray) -> np.ndarray:
    """What: 由 `u,v` 直接計算 2D 動能。"""

    u, v, _ = split_uvp(field)
    return 0.5 * np.mean(u**2 + v**2, axis=1)


def enstrophy(field: np.ndarray, domain_length: float = 2.0 * np.pi) -> np.ndarray:
    """What: 由 `u,v` 在規則網格近似計算 enstrophy。

    Why:
        dx 必須使用實際物理間距 `domain_length / grid_size`（Kolmogorov flow 域長為 2π）。
        使用 1/grid_size 會低估導數，使 enstrophy 偏差 (2π)² 倍。
    """

    u_flat, v_flat, _ = split_uvp(field)
    grid_size = int(round(np.sqrt(u_flat.shape[1])))
    if grid_size * grid_size != u_flat.shape[1]:
        raise ValueError("uv points 必須對應正方形網格。")
    u = u_flat.reshape(-1, grid_size, grid_size)
    v = v_flat.reshape(-1, grid_size, grid_size)
    dx = float(domain_length) / float(grid_size)
    dvdx = np.gradient(v, dx, axis=1, edge_order=2)
    dudy = np.gradient(u, dx, axis=2, edge_order=2)
    omega = dvdx - dudy
    return 0.5 * np.mean(omega**2, axis=(-2, -1))


def rollout_evaluate(model: dde.Model, dataset: PreparedDataset, rollout_steps: int) -> dict[str, object]:
    """What: 固定初值、直接查詢不同時間的 operator rollout。"""

    if rollout_steps <= 0:
        return {"num_cases": 0, "evaluated_steps": 0}

    sensor_step_errors: list[list[float]] = []
    physics_step_errors: list[list[float]] = []
    energy_step_errors: list[list[float]] = []
    enstrophy_step_errors: list[list[float]] = []

    for case in dataset.rollout_cases:
        max_steps = min(rollout_steps, len(case["future_sensor"]))
        if max_steps == 0:
            continue

        reynolds_norm = (float(case["reynolds"]) - dataset.reynolds_mean) / dataset.reynolds_std
        initial_history = np.asarray(case["initial_history"], dtype=np.float32).copy()
        step_dt = float(np.asarray(case["future_dt"], dtype=np.float32)[0])
        case_sensor_errors: list[float] = []
        case_physics_errors: list[float] = []
        case_energy_errors: list[float] = []
        case_enstrophy_errors: list[float] = []

        branch = np.zeros((1, initial_history.shape[0], initial_history.shape[1] + 1), dtype=np.float32)
        branch[0, :, :-1] = (initial_history - dataset.branch_mean) / dataset.branch_std
        branch[0, :, -1] = np.float32(reynolds_norm)

        for step in range(1, max_steps + 1):
            time_value = np.float32(step * step_dt)
            sensor_coords_t = build_component_trunk_coords(dataset.sensor_coords, time_value)
            physics_coords_t = build_component_trunk_coords(dataset.physics_coords, time_value)

            sensor_pred = predict_raw(
                model,
                (branch, sensor_coords_t),
                dataset.target_means,
                dataset.target_stds,
            )[0]
            physics_pred = predict_raw(
                model,
                (branch, physics_coords_t),
                dataset.target_means,
                dataset.target_stds,
            )[0:1]

            sensor_truth = np.asarray(case["future_sensor"][step - 1], dtype=np.float32)
            physics_truth = np.asarray(case["future_physics"][step - 1 : step], dtype=np.float32)
            case_sensor_errors.append(relative_l2(sensor_pred[None, :], sensor_truth[None, :]))
            case_physics_errors.append(relative_l2(physics_pred, physics_truth))

            pred_energy = kinetic_energy(physics_pred)[0]
            truth_energy = kinetic_energy(physics_truth)[0]
            pred_enstrophy = enstrophy(physics_pred, domain_length=dataset.physics_domain_length)[0]
            truth_enstrophy = enstrophy(physics_truth, domain_length=dataset.physics_domain_length)[0]
            case_energy_errors.append(float(abs(pred_energy - truth_energy) / (abs(truth_energy) + 1e-8)))
            case_enstrophy_errors.append(
                float(abs(pred_enstrophy - truth_enstrophy) / (abs(truth_enstrophy) + 1e-8))
            )

        sensor_step_errors.append(case_sensor_errors)
        physics_step_errors.append(case_physics_errors)
        energy_step_errors.append(case_energy_errors)
        enstrophy_step_errors.append(case_enstrophy_errors)

    if not sensor_step_errors:
        return {"num_cases": 0, "evaluated_steps": 0}

    sensor_arr = np.asarray(sensor_step_errors, dtype=np.float32)
    physics_arr = np.asarray(physics_step_errors, dtype=np.float32)
    energy_arr = np.asarray(energy_step_errors, dtype=np.float32)
    enstrophy_arr = np.asarray(enstrophy_step_errors, dtype=np.float32)
    return {
        "num_cases": int(sensor_arr.shape[0]),
        "evaluated_steps": int(sensor_arr.shape[1]),
        "sensor_relative_l2_mean": float(np.mean(sensor_arr)),
        "sensor_relative_l2_by_step": sensor_arr.mean(axis=0).astype(float).tolist(),
        "physics_relative_l2_mean": float(np.mean(physics_arr)),
        "physics_relative_l2_by_step": physics_arr.mean(axis=0).astype(float).tolist(),
        "energy_relative_error_mean": float(np.mean(energy_arr)),
        "energy_relative_error_by_step": energy_arr.mean(axis=0).astype(float).tolist(),
        "enstrophy_relative_error_mean": float(np.mean(enstrophy_arr)),
        "enstrophy_relative_error_by_step": enstrophy_arr.mean(axis=0).astype(float).tolist(),
    }


def build_pass_fail_summary(
    *,
    validation_mean_relative_l2: float | None,
    test_mean_relative_l2: float | None,
    rollout_summary: dict[str, object],
    include_test_metric: bool,
) -> dict[str, Any]:
    """What: 依固定門檻產生 PASS/FAIL 診斷結果。"""

    checks: list[dict[str, Any]] = []

    def append_check(metric: str, value: float | None, threshold: float) -> None:
        passed = (value is not None) and (float(value) < threshold)
        checks.append(
            {
                "metric": metric,
                "value": None if value is None else float(value),
                "threshold": float(threshold),
                "passed": bool(passed),
            }
        )

    append_check(
        "validation_mean_relative_l2",
        validation_mean_relative_l2,
        PASS_FAIL_THRESHOLDS["validation_mean_relative_l2"],
    )
    if include_test_metric:
        append_check(
            "test_mean_relative_l2",
            test_mean_relative_l2,
            PASS_FAIL_THRESHOLDS["test_mean_relative_l2"],
        )

    rollout_sensor = rollout_summary.get("sensor_relative_l2_mean")
    rollout_physics = rollout_summary.get("physics_relative_l2_mean")
    append_check(
        "rollout_sensor_relative_l2_mean",
        None if rollout_sensor is None else float(rollout_sensor),
        PASS_FAIL_THRESHOLDS["rollout_sensor_relative_l2_mean"],
    )
    append_check(
        "rollout_physics_relative_l2_mean",
        None if rollout_physics is None else float(rollout_physics),
        PASS_FAIL_THRESHOLDS["rollout_physics_relative_l2_mean"],
    )

    failed_metrics = [entry["metric"] for entry in checks if not entry["passed"]]
    return {
        "status": "PASS" if not failed_metrics else "FAIL",
        "scope": "full" if include_test_metric else "mid",
        "thresholds": dict(PASS_FAIL_THRESHOLDS),
        "checks": checks,
        "failed_metrics": failed_metrics,
    }


def build_mid_evaluation_summary(
    model: dde.Model,
    data: PhysicsInformedTripleCartesianProd,
    dataset: PreparedDataset,
    X_val: tuple[np.ndarray, np.ndarray],
    y_val: np.ndarray,
    rollout_steps: int,
    step: int,
) -> dict[str, Any]:
    """What: 產生中途評估摘要。"""

    rollout_summary = rollout_evaluate(model, dataset, rollout_steps)
    summary = {
        "step": int(step),
        "validation_mean_relative_l2": evaluate(model, X_val, y_val, dataset.target_means, dataset.target_stds),
        "validation_unweighted_losses": compute_unweighted_losses(
            model,
            data,
            X_val,
            y_val,
            dataset.val_dt,
            dataset.val_nu,
            dataset.val_forcing_u,
            dataset.val_forcing_v,
        ),
        "loss_weights": None if model.loss_weights is None else [float(value) for value in model.loss_weights],
        "rollout": rollout_summary,
    }
    summary["pass_fail"] = build_pass_fail_summary(
        validation_mean_relative_l2=summary["validation_mean_relative_l2"],
        test_mean_relative_l2=None,
        rollout_summary=rollout_summary,
        include_test_metric=False,
    )
    return summary


def build_full_evaluation_summary(
    model: dde.Model,
    data: PhysicsInformedTripleCartesianProd,
    dataset: PreparedDataset,
    X_val: tuple[np.ndarray, np.ndarray],
    y_val: np.ndarray,
    X_test: tuple[np.ndarray, np.ndarray],
    y_test: np.ndarray,
    rollout_steps: int,
) -> dict[str, Any]:
    """What: 產生完整 validation / test / rollout 摘要。"""

    validation_metric = evaluate(model, X_val, y_val, dataset.target_means, dataset.target_stds)
    test_metric = evaluate(model, X_test, y_test, dataset.target_means, dataset.target_stds)
    validation_losses = compute_unweighted_losses(
        model,
        data,
        X_val,
        y_val,
        dataset.val_dt,
        dataset.val_nu,
        dataset.val_forcing_u,
        dataset.val_forcing_v,
    )
    test_losses = compute_unweighted_losses(
        model,
        data,
        X_test,
        y_test,
        dataset.test_dt,
        dataset.test_nu,
        dataset.test_forcing_u,
        dataset.test_forcing_v,
    )
    rollout_summary = rollout_evaluate(model, dataset, rollout_steps)
    summary = {
        "validation_mean_relative_l2": validation_metric,
        "test_mean_relative_l2": test_metric,
        "validation_unweighted_losses": validation_losses,
        "test_unweighted_losses": test_losses,
        "final_loss_weights": None if model.loss_weights is None else [float(value) for value in model.loss_weights],
        "rollout": rollout_summary,
    }
    summary["pass_fail"] = build_pass_fail_summary(
        validation_mean_relative_l2=validation_metric,
        test_mean_relative_l2=test_metric,
        rollout_summary=rollout_summary,
        include_test_metric=True,
    )
    return summary


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """What: JSON 輸出工具。"""

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class LearningRateWarmupCallback(dde.callbacks.Callback):
    """What: 對 optimizer 套用線性 learning-rate warm-up。"""

    def __init__(
        self,
        warmup_steps: int,
        start_factor: float,
        history_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        self.start_factor = float(start_factor)
        self.history_path = history_path
        self.base_lrs: list[float] | None = None
        self.history: list[dict[str, Any]] = []
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps 不可小於 0。")
        if not (0.0 < self.start_factor <= 1.0):
            raise ValueError("start_factor 必須介於 (0, 1]。")

    def _resolve_optimizer(self) -> torch.optim.Optimizer | None:
        optimizer = getattr(self.model, "opt", None)
        if optimizer is None or not hasattr(optimizer, "param_groups"):
            return None
        return optimizer

    def on_train_begin(self) -> None:
        optimizer = self._resolve_optimizer()
        if optimizer is None:
            return
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if self.warmup_steps == 0:
            return
        initial_factor = self.start_factor
        for group, base_lr in zip(optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * initial_factor
        event = {
            "step": 0,
            "factor": float(initial_factor),
            "lrs": [float(group["lr"]) for group in optimizer.param_groups],
        }
        self.history.append(event)
        if self.history_path is not None:
            write_json(self.history_path, self.history)

    def on_epoch_begin(self) -> None:
        if self.warmup_steps <= 0 or self.base_lrs is None:
            return
        optimizer = self._resolve_optimizer()
        if optimizer is None:
            return
        step = int(self.model.train_state.step)
        if step > self.warmup_steps:
            return

        progress = min(1.0, step / float(max(1, self.warmup_steps)))
        factor = self.start_factor + (1.0 - self.start_factor) * progress
        for group, base_lr in zip(optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

        if step in (0, self.warmup_steps):
            event = {
                "step": step,
                "factor": float(factor),
                "lrs": [float(group["lr"]) for group in optimizer.param_groups],
            }
            self.history.append(event)
            if self.history_path is not None:
                write_json(self.history_path, self.history)
            print_section(f"LR Warmup @ {step}")
            print(json.dumps(event, indent=2, ensure_ascii=False))


class MidEvaluationCheckpointCallback(dde.callbacks.Callback):
    """What: 定期 checkpoint + 中途評估。"""

    def __init__(
        self,
        period: int,
        artifacts_dir: Path,
        data: PhysicsInformedTripleCartesianProd,
        dataset: PreparedDataset,
        X_val: tuple[np.ndarray, np.ndarray],
        y_val: np.ndarray,
        rollout_steps: int,
    ) -> None:
        super().__init__()
        self.period = period
        self.artifacts_dir = artifacts_dir
        self.data = data
        self.dataset = dataset
        self.X_val = X_val
        self.y_val = y_val
        self.rollout_steps = rollout_steps
        self.history: list[dict[str, Any]] = []
        self.checkpoints_dir = artifacts_dir / "checkpoints"
        self.mid_evals_dir = artifacts_dir / "mid_evals"
        self.best_dir = artifacts_dir / "best_validation"
        self.best_validation_metric = float("inf")
        self.best_summary: dict[str, Any] | None = None
        self.best_checkpoint_path: str | None = None
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.mid_evals_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self) -> None:
        if self.period <= 0:
            return
        step = int(self.model.train_state.step)
        if step == 0 or step % self.period != 0:
            return

        summary = build_mid_evaluation_summary(
            model=self.model,
            data=self.data,
            dataset=self.dataset,
            X_val=self.X_val,
            y_val=self.y_val,
            rollout_steps=self.rollout_steps,
            step=step,
        )
        checkpoint_prefix = self.checkpoints_dir / f"kolmogorov_deeponet_step_{step}"
        checkpoint_path = str(self.model.save(str(checkpoint_prefix)))
        summary["checkpoint_path"] = checkpoint_path
        write_json(self.mid_evals_dir / f"eval_step_{step}.json", summary)
        self.history.append(summary)

        validation_metric = float(summary["validation_mean_relative_l2"])
        if validation_metric < self.best_validation_metric:
            self.best_validation_metric = validation_metric
            self.best_summary = dict(summary)
            best_prefix = self.best_dir / "kolmogorov_deeponet_best"
            self.best_checkpoint_path = str(self.model.save(str(best_prefix)))
            self.best_summary["best_checkpoint_path"] = self.best_checkpoint_path
            write_json(self.best_dir / "best_validation_summary.json", self.best_summary)

        print_section(f"Mid Evaluation @ {step}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


class TotalLossThresholdStopCallback(dde.callbacks.Callback):
    """What: weighted total train loss 門檻早停。"""

    def __init__(self, threshold: float) -> None:
        super().__init__()
        if threshold <= 0.0:
            raise ValueError("early_stop_total_loss 必須大於 0。")
        self.threshold = float(threshold)
        self.stop_step: int | None = None
        self.stop_loss: float | None = None

    def on_epoch_end(self) -> None:
        current = float(np.sum(np.asarray(self.model.train_state.loss_train, dtype=np.float64)))
        if current <= self.threshold:
            self.stop_step = int(self.model.train_state.step)
            self.stop_loss = current
            self.model.stop_training = True
            print_section("Early Stop Triggered")
            print(
                json.dumps(
                    {
                        "step": self.stop_step,
                        "train_total_loss": self.stop_loss,
                        "threshold": self.threshold,
                    },
                    ensure_ascii=False,
                )
            )

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "threshold": self.threshold,
            "triggered": self.stop_step is not None,
            "step": self.stop_step,
            "train_total_loss": self.stop_loss,
        }


def export_best_checkpoint(
    artifacts_dir: Path,
    final_checkpoint_path: str,
    best_validation_checkpoint_path: str | None,
) -> str:
    """What: 導出固定檔名 best checkpoint。"""

    source = best_validation_checkpoint_path or final_checkpoint_path
    target = artifacts_dir / "best_checkpoint.pt"
    shutil.copy2(source, target)
    write_json(
        artifacts_dir / "best_checkpoint_export.json",
        {
            "source_checkpoint": source,
            "exported_checkpoint": str(target),
            "from_validation_best": best_validation_checkpoint_path is not None,
        },
    )
    return str(target)


def print_section(title: str) -> None:
    """What: 輸出區段標題。"""

    print(f"=== {title} ===")


def main() -> None:
    """What: 文獻對齊 PI-DeepONet 訓練入口。"""

    args = parse_args()
    runtime_device = configure_torch_runtime(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = DatasetConfig(
        field=args.field,
        num_sensors=args.num_sensors,
        history_steps=args.history_steps,
        horizon_steps=args.horizon_steps,
        temporal_stride=args.temporal_stride,
        burn_in_steps=args.burn_in_steps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        physics_stride=args.physics_stride,
        seed=args.seed,
    )
    data_files = resolve_data_files(args.data_file)
    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print_section("Configuration")
    print(
        json.dumps(
            {
                "data_files": [str(path) for path in data_files],
                "config": None if args.config is None else str(args.config),
                "field": config.field,
                "num_sensors": config.num_sensors,
                "history_steps": config.history_steps,
                "horizon_steps": config.horizon_steps,
                "temporal_stride": config.temporal_stride,
                "burn_in_steps": config.burn_in_steps,
                "train_ratio": config.train_ratio,
                "val_ratio": config.val_ratio,
                "physics_stride": config.physics_stride,
                "data_loss_weight": args.data_loss_weight,
                "physics_loss_weight": args.physics_loss_weight,
                "physics_time_samples": args.physics_time_samples,
                "physics_branch_batch_size": args.physics_branch_batch_size,
                "physics_continuity_weight": args.physics_continuity_weight,
                "physics_causal_epsilon": args.physics_causal_epsilon,
                "time_fourier_modes": args.time_fourier_modes,
                "iterations": args.iterations,
                "batch_size": args.batch_size,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "lr_warmup_steps": args.lr_warmup_steps,
                "lr_warmup_start_factor": args.lr_warmup_start_factor,
                "weight_decay": args.weight_decay,
                "lr_schedule": args.lr_schedule,
                "min_learning_rate": args.min_learning_rate,
                "lr_step_size": args.lr_step_size,
                "lr_step_gamma": args.lr_step_gamma,
                "rollout_steps": args.rollout_steps,
                "checkpoint_period": args.checkpoint_period,
                "branch_hidden_dims": args.branch_hidden_dims,
                "trunk_hidden_dims": args.trunk_hidden_dims,
                "latent_width": args.latent_width,
                "use_gated_mlp": args.use_gated_mlp,
                "use_transformer_branch": args.use_transformer_branch,
                "use_resnet_branch": args.use_resnet_branch,
                "trunk_rff_features": args.trunk_rff_features,
                "trunk_rff_sigma": args.trunk_rff_sigma,
                "transformer_model_dim": args.transformer_model_dim,
                "transformer_num_heads": args.transformer_num_heads,
                "transformer_num_layers": args.transformer_num_layers,
                "transformer_ff_dim": args.transformer_ff_dim,
                "transformer_dropout": args.transformer_dropout,
                "early_stop_total_loss": args.early_stop_total_loss,
                "device": args.device,
                "resolved_device": str(runtime_device),
                "artifacts_dir": str(artifacts_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print_section("Dataset")
    dataset = build_dataset(data_files=data_files, config=config)
    X_train = dataset.train_x
    y_train = dataset.train_y
    X_val = dataset.val_x
    y_val = dataset.val_y
    X_test = dataset.test_x
    y_test = dataset.test_y
    print(
        json.dumps(
            {
                "train_samples": int(len(X_train[0])),
                "val_samples": int(len(X_val[0])),
                "test_samples": int(len(X_test[0])),
                "branch_shape": list(X_train[0].shape[1:]),
                "trunk_points": int(X_train[1].shape[0]),
                "encoded_trunk_dim": int(X_train[1].shape[1]),
                "target_dim": int(y_train.shape[1]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    data = PhysicsInformedTripleCartesianProd(
        dataset,
        physics_time_samples=args.physics_time_samples,
        physics_branch_batch_size=args.physics_branch_batch_size,
        disable_physics_loss=(args.physics_loss_weight == 0.0),
        physics_continuity_weight=args.physics_continuity_weight,
        physics_causal_epsilon=args.physics_causal_epsilon,
    )
    net = create_model(
        branch_shape=tuple(int(dim) for dim in X_train[0].shape[1:]),
        trunk_dim=int(X_train[1].shape[1]),
        branch_hidden_dims=args.branch_hidden_dims,
        trunk_hidden_dims=args.trunk_hidden_dims,
        latent_width=args.latent_width,
        use_gated_mlp=args.use_gated_mlp,
        time_fourier_modes=args.time_fourier_modes,
        use_transformer_branch=args.use_transformer_branch,
        transformer_model_dim=args.transformer_model_dim,
        transformer_num_heads=args.transformer_num_heads,
        transformer_num_layers=args.transformer_num_layers,
        transformer_ff_dim=args.transformer_ff_dim,
        transformer_dropout=args.transformer_dropout,
        use_resnet_branch=args.use_resnet_branch,
        trunk_rff_features=args.trunk_rff_features,
        trunk_rff_sigma=args.trunk_rff_sigma,
    )
    net = net.to(runtime_device)
    print_section("Model")
    print(
        json.dumps(
            {
                "branch_shape": list(X_train[0].shape[1:]),
                "branch_encoder": (
                    "transformer" if args.use_transformer_branch
                    else "resnet" if args.use_resnet_branch
                    else "gated_mlp" if args.use_gated_mlp
                    else "mlp"
                ),
                "branch_layers": [int(np.prod(np.asarray(X_train[0].shape[1:], dtype=np.int64))), *args.branch_hidden_dims, args.latent_width],
                "trunk_raw_input_dim": int(X_train[1].shape[1]),
                "time_fourier_modes": args.time_fourier_modes,
                "trunk_encoder": "rff" if args.trunk_rff_features > 0 else "time_fourier",
                "trunk_rff_features": args.trunk_rff_features,
                "trunk_rff_sigma": args.trunk_rff_sigma,
                "trunk_layers": (
                    [2 * args.trunk_rff_features + 8, *args.trunk_hidden_dims, args.latent_width]
                    if args.trunk_rff_features > 0
                    else [int(X_train[1].shape[1]) + 2 * args.time_fourier_modes, *args.trunk_hidden_dims, args.latent_width]
                ),
                "trainable_parameters": count_trainable_parameters(net),
                "use_gated_mlp": args.use_gated_mlp,
                "use_transformer_branch": args.use_transformer_branch,
                "use_resnet_branch": args.use_resnet_branch,
                "device": str(next(net.parameters()).device),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    model = dde.Model(data, net)
    mid_eval_callback = MidEvaluationCheckpointCallback(
        period=args.checkpoint_period,
        artifacts_dir=artifacts_dir,
        data=data,
        dataset=dataset,
        X_val=X_val,
        y_val=y_val,
        rollout_steps=args.rollout_steps,
    )
    lr_warmup_callback = (
        LearningRateWarmupCallback(
            warmup_steps=args.lr_warmup_steps,
            start_factor=args.lr_warmup_start_factor,
            history_path=artifacts_dir / "lr_warmup_history.json",
        )
        if args.lr_warmup_steps > 0
        else None
    )
    early_stop_callback = (
        TotalLossThresholdStopCallback(args.early_stop_total_loss) if args.early_stop_total_loss > 0.0 else None
    )
    callbacks: list[dde.callbacks.Callback] = []
    if lr_warmup_callback is not None:
        callbacks.append(lr_warmup_callback)
    if args.checkpoint_period > 0:
        callbacks.append(mid_eval_callback)
    if early_stop_callback is not None:
        callbacks.append(early_stop_callback)

    print_section("Training")
    train_start_time = time.perf_counter()
    losshistory, train_state, stage_timings = train_model(model, args, callbacks)
    train_end_time = time.perf_counter()

    evaluation_summary = build_full_evaluation_summary(
        model=model,
        data=data,
        dataset=dataset,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        rollout_steps=args.rollout_steps,
    )
    print_section("Evaluation")
    print(json.dumps(evaluation_summary, indent=2, ensure_ascii=False))

    np.savez(artifacts_dir / "dataset_stats.npz", **dataset.metadata)
    save_training_history(
        losshistory=losshistory,
        train_state=train_state,
        artifacts_dir=artifacts_dir,
        evaluation_metric=evaluation_summary["test_mean_relative_l2"],
        training_wall_time_seconds=train_end_time - train_start_time,
    )
    write_json(artifacts_dir / "evaluation_summary.json", evaluation_summary)
    write_json(
        artifacts_dir / "experiment_manifest.json",
        {
            "configuration": {
                "data_file": [str(path) for path in data_files],
                "config": None if args.config is None else str(args.config),
                "num_sensors": args.num_sensors,
                "history_steps": args.history_steps,
                "horizon_steps": args.horizon_steps,
                "temporal_stride": args.temporal_stride,
                "burn_in_steps": args.burn_in_steps,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "physics_stride": args.physics_stride,
                "data_loss_weight": args.data_loss_weight,
                "physics_loss_weight": args.physics_loss_weight,
                "physics_time_samples": args.physics_time_samples,
                "physics_branch_batch_size": args.physics_branch_batch_size,
                "physics_continuity_weight": args.physics_continuity_weight,
                "physics_causal_epsilon": args.physics_causal_epsilon,
                "time_fourier_modes": args.time_fourier_modes,
                "iterations": args.iterations,
                "batch_size": args.batch_size,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "lr_warmup_steps": args.lr_warmup_steps,
                "lr_warmup_start_factor": args.lr_warmup_start_factor,
                "weight_decay": args.weight_decay,
                "lr_schedule": args.lr_schedule,
                "min_learning_rate": args.min_learning_rate,
                "lr_step_size": args.lr_step_size,
                "lr_step_gamma": args.lr_step_gamma,
                "rollout_steps": args.rollout_steps,
                "checkpoint_period": args.checkpoint_period,
                "branch_hidden_dims": args.branch_hidden_dims,
                "trunk_hidden_dims": args.trunk_hidden_dims,
                "latent_width": args.latent_width,
                "use_gated_mlp": args.use_gated_mlp,
                "use_transformer_branch": args.use_transformer_branch,
                "use_resnet_branch": args.use_resnet_branch,
                "trunk_rff_features": args.trunk_rff_features,
                "trunk_rff_sigma": args.trunk_rff_sigma,
                "transformer_model_dim": args.transformer_model_dim,
                "transformer_num_heads": args.transformer_num_heads,
                "transformer_num_layers": args.transformer_num_layers,
                "transformer_ff_dim": args.transformer_ff_dim,
                "transformer_dropout": args.transformer_dropout,
                "early_stop_total_loss": args.early_stop_total_loss,
                "seed": args.seed,
                "device": args.device,
                "resolved_device": str(runtime_device),
            },
            "dataset": {
                "train_samples": int(len(X_train[0])),
                "val_samples": int(len(X_val[0])),
                "test_samples": int(len(X_test[0])),
                "branch_shape": list(X_train[0].shape[1:]),
                "trunk_points": int(X_train[1].shape[0]),
                "encoded_trunk_dim": int(X_train[1].shape[1]),
                "physics_points": int(len(dataset.physics_coords)),
            },
            "model": {
                "branch_shape": list(X_train[0].shape[1:]),
                "branch_encoder": (
                    "transformer" if args.use_transformer_branch
                    else "resnet" if args.use_resnet_branch
                    else "gated_mlp" if args.use_gated_mlp
                    else "mlp"
                ),
                "trunk_encoder": "rff" if args.trunk_rff_features > 0 else "time_fourier",
                "branch_layers": [int(np.prod(np.asarray(X_train[0].shape[1:], dtype=np.int64))), *args.branch_hidden_dims, args.latent_width],
                "trunk_raw_input_dim": int(X_train[1].shape[1]),
                "time_fourier_modes": args.time_fourier_modes,
                "trunk_layers": (
                    [2 * args.trunk_rff_features + 8, *args.trunk_hidden_dims, args.latent_width]
                    if args.trunk_rff_features > 0
                    else [int(X_train[1].shape[1]) + 2 * args.time_fourier_modes, *args.trunk_hidden_dims, args.latent_width]
                ),
                "trainable_parameters": count_trainable_parameters(net),
                "use_gated_mlp": args.use_gated_mlp,
                "use_transformer_branch": args.use_transformer_branch,
                "use_resnet_branch": args.use_resnet_branch,
                "trunk_rff_features": args.trunk_rff_features,
                "trunk_rff_sigma": args.trunk_rff_sigma,
            },
            "runtime": {
                "training_wall_time_seconds": float(train_end_time - train_start_time),
                "seconds_per_iteration": float((train_end_time - train_start_time) / max(1, int(train_state.step))),
                **stage_timings,
            },
        },
    )
    write_json(artifacts_dir / "mid_eval_history.json", mid_eval_callback.history)
    if early_stop_callback is not None:
        write_json(artifacts_dir / "early_stop_summary.json", early_stop_callback.summary())

    final_checkpoint_path = str(model.save(str(artifacts_dir / "kolmogorov_deeponet")))
    exported_best_checkpoint_path = export_best_checkpoint(
        artifacts_dir=artifacts_dir,
        final_checkpoint_path=final_checkpoint_path,
        best_validation_checkpoint_path=mid_eval_callback.best_checkpoint_path,
    )
    source_checkpoint_for_eval = mid_eval_callback.best_checkpoint_path or final_checkpoint_path
    if source_checkpoint_for_eval != final_checkpoint_path:
        model.restore(source_checkpoint_for_eval, verbose=0)
        best_checkpoint_evaluation: dict[str, Any] = build_full_evaluation_summary(
            model=model,
            data=data,
            dataset=dataset,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            rollout_steps=args.rollout_steps,
        )
    else:
        best_checkpoint_evaluation = dict(evaluation_summary)
    best_checkpoint_evaluation["checkpoint"] = source_checkpoint_for_eval
    best_checkpoint_evaluation["exported_checkpoint"] = exported_best_checkpoint_path
    write_json(artifacts_dir / "best_checkpoint_evaluation.json", best_checkpoint_evaluation)


if __name__ == "__main__":
    main()
