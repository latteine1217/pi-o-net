"""Train physics-informed DeepONet aligned with arXiv:2103.10974.

What:
    以文獻對齊流程訓練 PI-DeepONet：branch 輸入初始條件 + Re，trunk 輸入 (x,y,t)，
    損失函數使用 L_IC + L_physics。
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
    "field": "omega",
    "num_sensors": 50,
    "horizon_steps": 10,
    "temporal_stride": 1,
    "burn_in_steps": 100,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "physics_stride": 32,
    "ic_loss_weight": 20.0,
    "physics_loss_weight": 1.0,
    "physics_time_samples": 4,
    "physics_branch_batch_size": None,
    "iterations": 20000,
    "batch_size": 32,
    "optimizer": "adamw",
    "learning_rate": 1e-3,
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
    "early_stop_total_loss": 1e-4,
    "seed": 42,
    "artifacts_dir": Path("artifacts"),
}


def configure_torch_runtime() -> None:
    """What: 啟用 PyTorch 的穩健效能選項。"""

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


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

    unknown_keys = sorted(set(config_data) - set(DEFAULT_TRAIN_ARGS))
    if unknown_keys:
        raise ValueError(f"訓練 config 含有不支援的欄位: {unknown_keys}")

    normalized = dict(config_data)
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
    parser.add_argument("--field", type=str, choices=["omega"], default=None)
    parser.add_argument("--num-sensors", type=int, default=None)
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--temporal-stride", type=int, default=None)
    parser.add_argument("--burn-in-steps", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--physics-stride", type=int, default=None)
    parser.add_argument("--ic-loss-weight", type=float, default=None)
    parser.add_argument("--physics-loss-weight", type=float, default=None)
    parser.add_argument("--physics-time-samples", type=int, default=None)
    parser.add_argument("--physics-branch-batch-size", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
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
    parser.add_argument("--early-stop-total-loss", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
    if merged["ic_loss_weight"] <= 0.0:
        raise ValueError("ic_loss_weight 必須大於 0。")
    if merged["physics_loss_weight"] <= 0.0:
        raise ValueError("physics_loss_weight 必須大於 0。")
    if merged["physics_branch_batch_size"] is not None and int(merged["physics_branch_batch_size"]) <= 0:
        raise ValueError("physics_branch_batch_size 必須大於 0。")
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


class CallableTrunkDeepONetCartesianProd(dde.nn.pytorch.deeponet.DeepONetCartesianProd):
    """What: 讓 trunk 支援 callable module。"""

    def build_trunk_net(self, layer_sizes_trunk):
        if len(layer_sizes_trunk) > 1 and callable(layer_sizes_trunk[1]):
            return layer_sizes_trunk[1]
        return super().build_trunk_net(layer_sizes_trunk)


def create_model(
    branch_dim: int,
    trunk_dim: int,
    branch_hidden_dims: list[int],
    trunk_hidden_dims: list[int],
    latent_width: int,
    use_gated_mlp: bool = False,
) -> dde.nn.pytorch.deeponet.DeepONetCartesianProd:
    """What: 建立文獻對齊 DeepONet 模型。"""

    branch_layers = [branch_dim, *branch_hidden_dims, latent_width]
    trunk_layers = [trunk_dim, *trunk_hidden_dims, latent_width]
    if use_gated_mlp:
        branch_net = ModifiedGatedMLP(branch_layers, activation="tanh", kernel_initializer="Glorot normal")
        trunk_net = ModifiedGatedMLP(trunk_layers, activation="tanh", kernel_initializer="Glorot normal")
        return CallableTrunkDeepONetCartesianProd(
            (branch_dim, branch_net),
            (trunk_dim, trunk_net),
            activation="tanh",
            kernel_initializer="Glorot normal",
        )
    return dde.nn.DeepONetCartesianProd(
        branch_layers,
        trunk_layers,
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

    loss_weights = [args.ic_loss_weight, args.physics_loss_weight]
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
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    """What: 將標準化輸出還原為物理量。"""

    return model.predict(X) * target_std + target_mean


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    """What: 計算平均相對 L2 誤差。"""

    numerator = np.linalg.norm(prediction - target, axis=1)
    denominator = np.linalg.norm(target, axis=1) + 1e-8
    return float(np.mean(numerator / denominator))


def evaluate(
    model: dde.Model,
    X: tuple[np.ndarray, np.ndarray],
    y: np.ndarray,
    target_mean: float,
    target_std: float,
) -> float:
    """What: 在指定資料集上評估平均相對 L2。"""

    prediction = predict_raw(model, X, target_mean, target_std)
    target = y * target_std + target_mean
    return relative_l2(prediction, target)


def compute_unweighted_losses(
    model: dde.Model,
    data: PhysicsInformedTripleCartesianProd,
    X: tuple[np.ndarray, np.ndarray],
    y: np.ndarray,
    delta_t: np.ndarray,
    nu: np.ndarray,
    forcing: np.ndarray,
) -> dict[str, float]:
    """What: 計算未加權 L_IC 與 L_physics。"""

    parameter = next(model.net.parameters())
    branch_inputs = torch.as_tensor(X[0], dtype=parameter.dtype, device=parameter.device)
    trunk_inputs = torch.as_tensor(X[1], dtype=parameter.dtype, device=parameter.device)
    targets = torch.as_tensor(y, dtype=parameter.dtype, device=parameter.device)

    with torch.no_grad():
        pred_norm = model.net((branch_inputs, trunk_inputs))
        supervised_loss = torch.mean((pred_norm - targets) ** 2)
    cached = data._ensure_tensor_cache(parameter.device, parameter.dtype)
    grid_size = data.physics_grid_shape[0]
    physics_loss = data._physics_loss(
        model=model,
        branch_inputs=branch_inputs,
        physics_coords_xy=cached["physics_coords_xy"],
        delta_t_tensor=torch.as_tensor(delta_t, dtype=parameter.dtype, device=parameter.device).reshape(-1),
        nu_tensor=torch.as_tensor(nu, dtype=parameter.dtype, device=parameter.device).reshape(-1),
        forcing_tensor=torch.as_tensor(forcing, dtype=parameter.dtype, device=parameter.device).reshape(
            -1, grid_size, grid_size
        ),
    )
    return {
        "ic_mse": float(supervised_loss.detach().cpu().item()),
        "physics_residual_mse": float(physics_loss.detach().cpu().item()),
    }


def kinetic_energy(field: np.ndarray) -> np.ndarray:
    """What: 用頻域 streamfunction 估算 2D 動能。"""

    grid_size = int(round(np.sqrt(field.shape[1])))
    omega = field.reshape(-1, grid_size, grid_size)
    freq = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=1.0 / grid_size)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    k2 = kx**2 + ky**2

    omega_hat = np.fft.fft2(omega, axes=(-2, -1))
    psi_hat = np.zeros_like(omega_hat, dtype=np.complex64)
    nonzero = k2 > 0.0
    psi_hat[:, nonzero] = omega_hat[:, nonzero] / k2[nonzero]
    u = np.fft.ifft2(1j * ky * psi_hat, axes=(-2, -1)).real
    v = np.fft.ifft2(-1j * kx * psi_hat, axes=(-2, -1)).real
    return 0.5 * np.mean(u**2 + v**2, axis=(-2, -1))


def enstrophy(field: np.ndarray) -> np.ndarray:
    """What: 計算 enstrophy。"""

    grid_size = int(round(np.sqrt(field.shape[1])))
    omega = field.reshape(-1, grid_size, grid_size)
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
        initial_sensor = np.asarray(case["initial_sensor"], dtype=np.float32).copy()
        step_dt = float(np.asarray(case["future_dt"], dtype=np.float32)[0])
        case_sensor_errors: list[float] = []
        case_physics_errors: list[float] = []
        case_energy_errors: list[float] = []
        case_enstrophy_errors: list[float] = []

        branch = np.zeros((1, len(initial_sensor) + 1), dtype=np.float32)
        branch[0, :-1] = (initial_sensor - dataset.branch_mean) / dataset.branch_std
        branch[0, -1] = np.float32(reynolds_norm)

        for step in range(1, max_steps + 1):
            time_value = np.float32(step * step_dt)
            sensor_coords_t = np.concatenate(
                [dataset.sensor_coords, np.full((len(dataset.sensor_coords), 1), time_value, dtype=np.float32)],
                axis=1,
            )
            physics_coords_t = np.concatenate(
                [dataset.physics_coords, np.full((len(dataset.physics_coords), 1), time_value, dtype=np.float32)],
                axis=1,
            )

            sensor_pred = predict_raw(
                model,
                (branch, sensor_coords_t),
                dataset.target_mean,
                dataset.target_std,
            )[0]
            physics_pred = predict_raw(
                model,
                (branch, physics_coords_t),
                dataset.target_mean,
                dataset.target_std,
            )[0:1]

            sensor_truth = np.asarray(case["future_sensor"][step - 1], dtype=np.float32)
            physics_truth = np.asarray(case["future_physics"][step - 1 : step], dtype=np.float32)
            case_sensor_errors.append(relative_l2(sensor_pred[None, :], sensor_truth[None, :]))
            case_physics_errors.append(relative_l2(physics_pred, physics_truth))

            pred_energy = kinetic_energy(physics_pred)[0]
            truth_energy = kinetic_energy(physics_truth)[0]
            pred_enstrophy = enstrophy(physics_pred)[0]
            truth_enstrophy = enstrophy(physics_truth)[0]
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

    return {
        "step": int(step),
        "validation_mean_relative_l2": evaluate(model, X_val, y_val, dataset.target_mean, dataset.target_std),
        "validation_unweighted_losses": compute_unweighted_losses(
            model,
            data,
            X_val,
            y_val,
            dataset.val_dt,
            dataset.val_nu,
            dataset.val_forcing,
        ),
        "loss_weights": None if model.loss_weights is None else [float(value) for value in model.loss_weights],
        "rollout": rollout_evaluate(model, dataset, rollout_steps),
    }


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

    validation_metric = evaluate(model, X_val, y_val, dataset.target_mean, dataset.target_std)
    test_metric = evaluate(model, X_test, y_test, dataset.target_mean, dataset.target_std)
    validation_losses = compute_unweighted_losses(
        model,
        data,
        X_val,
        y_val,
        dataset.val_dt,
        dataset.val_nu,
        dataset.val_forcing,
    )
    test_losses = compute_unweighted_losses(
        model,
        data,
        X_test,
        y_test,
        dataset.test_dt,
        dataset.test_nu,
        dataset.test_forcing,
    )
    return {
        "validation_mean_relative_l2": validation_metric,
        "test_mean_relative_l2": test_metric,
        "validation_unweighted_losses": validation_losses,
        "test_unweighted_losses": test_losses,
        "final_loss_weights": None if model.loss_weights is None else [float(value) for value in model.loss_weights],
        "rollout": rollout_evaluate(model, dataset, rollout_steps),
    }


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """What: JSON 輸出工具。"""

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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

    configure_torch_runtime()
    args = parse_args()
    np.random.seed(args.seed)
    config = DatasetConfig(
        field=args.field,
        num_sensors=args.num_sensors,
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
                "horizon_steps": config.horizon_steps,
                "temporal_stride": config.temporal_stride,
                "burn_in_steps": config.burn_in_steps,
                "train_ratio": config.train_ratio,
                "val_ratio": config.val_ratio,
                "physics_stride": config.physics_stride,
                "ic_loss_weight": args.ic_loss_weight,
                "physics_loss_weight": args.physics_loss_weight,
                "physics_time_samples": args.physics_time_samples,
                "physics_branch_batch_size": args.physics_branch_batch_size,
                "iterations": args.iterations,
                "batch_size": args.batch_size,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
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
                "early_stop_total_loss": args.early_stop_total_loss,
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
                "branch_dim": int(X_train[0].shape[1]),
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
    )
    net = create_model(
        branch_dim=int(X_train[0].shape[1]),
        trunk_dim=int(X_train[1].shape[1]),
        branch_hidden_dims=args.branch_hidden_dims,
        trunk_hidden_dims=args.trunk_hidden_dims,
        latent_width=args.latent_width,
        use_gated_mlp=args.use_gated_mlp,
    )
    print_section("Model")
    print(
        json.dumps(
            {
                "branch_layers": [int(X_train[0].shape[1]), *args.branch_hidden_dims, args.latent_width],
                "trunk_layers": [int(X_train[1].shape[1]), *args.trunk_hidden_dims, args.latent_width],
                "trainable_parameters": count_trainable_parameters(net),
                "use_gated_mlp": args.use_gated_mlp,
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
    early_stop_callback = (
        TotalLossThresholdStopCallback(args.early_stop_total_loss) if args.early_stop_total_loss > 0.0 else None
    )
    callbacks: list[dde.callbacks.Callback] = []
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
                "data_files": [str(path) for path in data_files],
                "config": None if args.config is None else str(args.config),
                "num_sensors": args.num_sensors,
                "horizon_steps": args.horizon_steps,
                "temporal_stride": args.temporal_stride,
                "burn_in_steps": args.burn_in_steps,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "physics_stride": args.physics_stride,
                "ic_loss_weight": args.ic_loss_weight,
                "physics_loss_weight": args.physics_loss_weight,
                "physics_time_samples": args.physics_time_samples,
                "physics_branch_batch_size": args.physics_branch_batch_size,
                "iterations": args.iterations,
                "batch_size": args.batch_size,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
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
                "early_stop_total_loss": args.early_stop_total_loss,
                "seed": args.seed,
            },
            "dataset": {
                "train_samples": int(len(X_train[0])),
                "val_samples": int(len(X_val[0])),
                "test_samples": int(len(X_test[0])),
                "trunk_points": int(X_train[1].shape[0]),
                "encoded_trunk_dim": int(X_train[1].shape[1]),
                "physics_points": int(len(dataset.physics_coords)),
            },
            "model": {
                "branch_layers": [int(X_train[0].shape[1]), *args.branch_hidden_dims, args.latent_width],
                "trunk_layers": [int(X_train[1].shape[1]), *args.trunk_hidden_dims, args.latent_width],
                "trainable_parameters": count_trainable_parameters(net),
                "use_gated_mlp": args.use_gated_mlp,
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
