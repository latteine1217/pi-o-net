"""DNS sensor loading and physics-informed data utilities.

What:
    從 Kolmogorov DNS 時間序列中抽取固定隨機 sensors，建立 DeepONet 的觀測資料，
    並同時準備在規則 collocation grid 上計算 vorticity equation residual 所需的物理量。
Why:
    使用者要的是「DNS + sparse sensors + physics loss」，不是全場監督。
    這個模組把觀測 loss 與 physics loss 所需資料一起整理，讓訓練端能維持單一入口。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import deepxde as dde
import numpy as np
import torch
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler


Array = np.ndarray


@dataclass(slots=True)
class DatasetConfig:
    """What: 控制 DNS 資料抽樣與時間切分。

    Why:
        這個問題真正敏感的是 sensor 數量、時間步距與 physics collocation grid，
        必須把這些設計假設集中管理，才能清楚知道模型學的是哪一個 operator。
    """

    field: str = "omega"
    num_sensors: int = 1000
    horizon_steps: int = 1
    temporal_stride: int = 1
    burn_in_steps: int = 0
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    physics_stride: int = 32
    seed: int = 42


@dataclass(slots=True)
class LoadedDnsTrajectory:
    """What: 單條 DNS 軌跡經 sensor/collocation 抽樣後的表示。"""

    source_file: str
    reynolds: float
    nu: float
    dt: Array
    sensor_coords: Array
    sensor_values: Array
    physics_coords: Array
    physics_values: Array
    forcing: Array
    physics_grid_shape: tuple[int, int]
    domain_length: float


@dataclass(slots=True)
class PreparedDataset:
    """What: DeepONet 觀測資料與 physics loss 輔助資料。"""

    train_x: tuple[Array, Array]
    train_y: Array
    val_x: tuple[Array, Array]
    val_y: Array
    test_x: tuple[Array, Array]
    test_y: Array
    train_current_physics: Array
    val_current_physics: Array
    test_current_physics: Array
    train_dt: Array
    val_dt: Array
    test_dt: Array
    train_nu: Array
    val_nu: Array
    test_nu: Array
    train_forcing: Array
    val_forcing: Array
    test_forcing: Array
    physics_coords: Array
    physics_grid_shape: tuple[int, int]
    physics_domain_length: float
    sensor_indices: Array
    sensor_coords: Array
    branch_mean: float
    branch_std: float
    reynolds_mean: float
    reynolds_std: float
    target_mean: float
    target_std: float
    rollout_cases: list[dict[str, object]]
    metadata: dict[str, object]
    horizon_steps: int
    seed: int


def resolve_data_files(explicit_files: list[str] | None) -> list[Path]:
    """What: 解析要使用的 DNS 檔案清單。

    Why:
        physics loss 寫死在 Kolmogorov vorticity equation，上游資料就應明確限定為 DNS。
    """

    if explicit_files:
        paths = [Path(path) for path in explicit_files]
    else:
        paths = sorted(Path("data/kolmogorov_dns").glob("*.npy"))
    if not paths:
        raise FileNotFoundError("找不到任何 DNS `.npy` 檔，請檢查 `data/kolmogorov_dns/` 或使用 `--data-file`。")
    return paths


def _extract_reynolds(payload: dict, source_file: Path) -> float:
    """What: 從檔案內容或檔名推回雷諾數。"""

    config = payload.get("config", {})
    if isinstance(config, dict) and config.get("nu"):
        return float(round(1.0 / float(config["nu"])))
    match = re.search(r"(\d+)", source_file.stem)
    if match:
        return float(match.group(1))
    raise ValueError(f"無法從 {source_file} 推出雷諾數。")


def _validate_dns_payload(payload: dict, source_file: Path, field: str) -> None:
    """What: 驗證 DNS 檔案是否包含 physics loss 需要的欄位。"""

    required = {"x", "y", "time", "config", field}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"{source_file} 缺少 DNS 欄位: {sorted(missing)}")


def _build_full_coords(x: Array, y: Array) -> Array:
    """What: 建立完整規則網格座標。"""

    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack([xx, yy], axis=-1)


def _sample_sensor_indices(num_points: int, num_sensors: int, seed: int) -> Array:
    """What: 固定隨機抽取 sensors。

    Why:
        DeepONet 的 branch sensors 應該在整個訓練過程保持固定，
        否則模型實際上會在學一個不停換 basis 的 operator。
    """

    if num_sensors > num_points:
        raise ValueError(f"要求 {num_sensors} 個 sensors，但網格總點數只有 {num_points}。")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, size=num_sensors, replace=False)).astype(np.int64)


def _forcing_vorticity(config: dict, physics_coords_grid: Array) -> Array:
    """What: 依 DNS 設定建立 Kolmogorov forcing 的渦度形式。"""

    amplitude = float(config.get("A", 0.1))
    forcing_wavenumber = float(config.get("k_f", 4.0))
    y = physics_coords_grid[..., 1]
    return (-amplitude * forcing_wavenumber * np.cos(forcing_wavenumber * y)).astype(np.float32)


def load_dns_trajectory(
    source_file: Path,
    config: DatasetConfig,
    sensor_indices: Array | None = None,
) -> tuple[LoadedDnsTrajectory, Array]:
    """What: 載入單個 DNS 檔並抽取 sensors 與 physics grid。"""

    payload = np.load(source_file, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError(f"{source_file} 不是預期的 dict-based DNS `.npy` 格式。")
    _validate_dns_payload(payload, source_file, config.field)

    x = np.asarray(payload["x"], dtype=np.float32)
    y = np.asarray(payload["y"], dtype=np.float32)
    time = np.asarray(payload["time"], dtype=np.float32)
    field = np.asarray(payload[config.field], dtype=np.float32)
    dns_config = dict(payload["config"])

    full_coords = _build_full_coords(x, y)
    flat_coords = full_coords.reshape(-1, 2)
    flat_field = field.reshape(field.shape[0], -1)

    if sensor_indices is None:
        sensor_indices = _sample_sensor_indices(
            num_points=flat_coords.shape[0],
            num_sensors=config.num_sensors,
            seed=config.seed,
        )
    sensor_coords = flat_coords[sensor_indices]
    sensor_values = flat_field[:, sensor_indices]

    physics_stride = max(1, int(config.physics_stride))
    physics_field = field[:, ::physics_stride, ::physics_stride]
    physics_coords_grid = full_coords[::physics_stride, ::physics_stride]
    forcing = _forcing_vorticity(dns_config, physics_coords_grid)

    dt = time[1:] - time[:-1]
    if np.any(dt <= 0.0):
        raise ValueError(f"{source_file} 的時間軸不是嚴格遞增。")

    trajectory = LoadedDnsTrajectory(
        source_file=str(source_file),
        reynolds=_extract_reynolds(payload, source_file),
        nu=float(dns_config["nu"]),
        dt=dt.astype(np.float32),
        sensor_coords=sensor_coords.astype(np.float32),
        sensor_values=sensor_values.astype(np.float32),
        physics_coords=physics_coords_grid.reshape(-1, 2).astype(np.float32),
        physics_values=physics_field.reshape(physics_field.shape[0], -1).astype(np.float32),
        forcing=forcing.reshape(-1).astype(np.float32),
        physics_grid_shape=(physics_field.shape[1], physics_field.shape[2]),
        domain_length=float(dns_config.get("L", float(x[-1] - x[0] + (x[1] - x[0])))),
    )
    return trajectory, sensor_indices


def _split_indices(
    num_pairs: int,
    train_ratio: float,
    val_ratio: float,
    temporal_stride: int,
    burn_in_steps: int,
) -> tuple[Array, Array, Array]:
    """What: 建立 train/val/test 的時間索引。"""

    if burn_in_steps < 0:
        raise ValueError("burn_in_steps 不可為負數。")
    start_index = min(int(burn_in_steps), num_pairs)
    indices = np.arange(start_index, num_pairs, max(1, int(temporal_stride)), dtype=np.int64)
    if len(indices) < 3:
        raise ValueError("時間樣本不足，無法同時切出 train/val/test 資料。")
    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("需要滿足 `0 < train_ratio`, `0 < val_ratio`, 且 `train_ratio + val_ratio < 1`。")

    train_end = max(1, int(len(indices) * train_ratio))
    val_end = max(train_end + 1, int(len(indices) * (train_ratio + val_ratio)))
    train_end = min(train_end, len(indices) - 2)
    val_end = min(val_end, len(indices) - 1)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _build_sample_arrays(
    trajectory: LoadedDnsTrajectory,
    pair_indices: Array,
    target_offset: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """What: 將時間配對轉成觀測資料與 physics loss 所需陣列。"""

    branch = np.zeros((len(pair_indices), trajectory.sensor_values.shape[1] + 1), dtype=np.float32)
    target = np.zeros((len(pair_indices), trajectory.sensor_values.shape[1]), dtype=np.float32)
    current_physics = np.zeros((len(pair_indices), trajectory.physics_values.shape[1]), dtype=np.float32)
    dt = np.zeros((len(pair_indices), 1), dtype=np.float32)
    nu = np.full((len(pair_indices), 1), trajectory.nu, dtype=np.float32)

    for row, start in enumerate(pair_indices):
        branch[row, :-1] = trajectory.sensor_values[start]
        branch[row, -1] = np.float32(trajectory.reynolds)
        target[row] = trajectory.sensor_values[start + target_offset]
        current_physics[row] = trajectory.physics_values[start]
        dt[row, 0] = trajectory.dt[start]
    forcing = np.repeat(trajectory.forcing[None, :], len(pair_indices), axis=0)
    return branch, target, current_physics, dt, nu, forcing.astype(np.float32)


def _normalize_dataset(
    train_branch: Array,
    val_branch: Array,
    test_branch: Array,
    train_target: Array,
    val_target: Array,
    test_target: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, dict[str, float]]:
    """What: 用訓練集統計量正規化觀測資料。"""

    branch_mean = float(np.mean(train_branch[:, :-1]))
    branch_std = float(np.std(train_branch[:, :-1]) + 1e-6)
    target_mean = float(np.mean(train_target))
    target_std = float(np.std(train_target) + 1e-6)
    reynolds_mean = float(np.mean(train_branch[:, -1]))
    reynolds_std = float(np.std(train_branch[:, -1]) + 1e-6)

    train_branch = train_branch.copy()
    val_branch = val_branch.copy()
    test_branch = test_branch.copy()
    train_target = train_target.copy()
    val_target = val_target.copy()
    test_target = test_target.copy()

    train_branch[:, :-1] = (train_branch[:, :-1] - branch_mean) / branch_std
    val_branch[:, :-1] = (val_branch[:, :-1] - branch_mean) / branch_std
    test_branch[:, :-1] = (test_branch[:, :-1] - branch_mean) / branch_std
    train_branch[:, -1] = (train_branch[:, -1] - reynolds_mean) / reynolds_std
    val_branch[:, -1] = (val_branch[:, -1] - reynolds_mean) / reynolds_std
    test_branch[:, -1] = (test_branch[:, -1] - reynolds_mean) / reynolds_std
    train_target = (train_target - target_mean) / target_std
    val_target = (val_target - target_mean) / target_std
    test_target = (test_target - target_mean) / target_std

    stats = {
        "branch_mean": branch_mean,
        "branch_std": branch_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "reynolds_mean": reynolds_mean,
        "reynolds_std": reynolds_std,
    }
    return train_branch, val_branch, test_branch, train_target, val_target, test_target, stats


def build_dataset(data_files: list[Path], config: DatasetConfig) -> PreparedDataset:
    """What: 建立 DNS sparse-sensor + physics grid 資料集。"""

    train_branch_parts: list[Array] = []
    train_target_parts: list[Array] = []
    train_current_parts: list[Array] = []
    train_dt_parts: list[Array] = []
    train_nu_parts: list[Array] = []
    train_forcing_parts: list[Array] = []

    val_branch_parts: list[Array] = []
    val_target_parts: list[Array] = []
    val_current_parts: list[Array] = []
    val_dt_parts: list[Array] = []
    val_nu_parts: list[Array] = []
    val_forcing_parts: list[Array] = []

    test_branch_parts: list[Array] = []
    test_target_parts: list[Array] = []
    test_current_parts: list[Array] = []
    test_dt_parts: list[Array] = []
    test_nu_parts: list[Array] = []
    test_forcing_parts: list[Array] = []

    sensor_indices: Array | None = None
    sensor_coords: Array | None = None
    physics_coords: Array | None = None
    physics_grid_shape: tuple[int, int] | None = None
    domain_length: float | None = None
    dt_reference: float | None = None
    per_file_metadata: list[dict[str, object]] = []
    rollout_cases: list[dict[str, object]] = []

    for source_file in data_files:
        trajectory, sensor_indices = load_dns_trajectory(source_file, config, sensor_indices=sensor_indices)
        if sensor_coords is None:
            sensor_coords = trajectory.sensor_coords
            physics_coords = trajectory.physics_coords
            physics_grid_shape = trajectory.physics_grid_shape
            domain_length = trajectory.domain_length
        else:
            if not np.allclose(sensor_coords, trajectory.sensor_coords):
                raise ValueError("所有 DNS 檔必須共享同一組 sensor coordinates。")
            if not np.allclose(physics_coords, trajectory.physics_coords):
                raise ValueError("所有 DNS 檔必須共享同一組 physics grid。")
            if physics_grid_shape != trajectory.physics_grid_shape:
                raise ValueError("所有 DNS 檔必須共享同一個 physics grid shape。")
            if domain_length != trajectory.domain_length:
                raise ValueError("所有 DNS 檔必須共享同一個 domain length。")

        num_pairs = len(trajectory.sensor_values) - config.horizon_steps
        if num_pairs <= 0:
            raise ValueError(
                f"{trajectory.source_file} 可用時間樣本不足：horizon_steps={config.horizon_steps} 超過序列長度。"
            )
        local_dt = float(np.mean(trajectory.dt))
        if dt_reference is None:
            dt_reference = local_dt
        elif abs(local_dt - dt_reference) > 1e-7:
            raise ValueError("目前要求所有 DNS 檔具有相同時間步距，才能共享 trunk 的時間座標。")
        train_indices, val_indices, test_indices = _split_indices(
            num_pairs,
            config.train_ratio,
            config.val_ratio,
            config.temporal_stride,
            config.burn_in_steps,
        )

        train_branch, train_target, train_current, train_dt, train_nu, train_forcing = _build_sample_arrays(
            trajectory, train_indices, target_offset=0
        )
        val_branch, val_target, val_current, val_dt, val_nu, val_forcing = _build_sample_arrays(
            trajectory, val_indices, target_offset=config.horizon_steps
        )
        test_branch, test_target, test_current, test_dt, test_nu, test_forcing = _build_sample_arrays(
            trajectory, test_indices, target_offset=config.horizon_steps
        )

        train_branch_parts.append(train_branch)
        train_target_parts.append(train_target)
        train_current_parts.append(train_current)
        train_dt_parts.append(train_dt)
        train_nu_parts.append(train_nu)
        train_forcing_parts.append(train_forcing)

        val_branch_parts.append(val_branch)
        val_target_parts.append(val_target)
        val_current_parts.append(val_current)
        val_dt_parts.append(val_dt)
        val_nu_parts.append(val_nu)
        val_forcing_parts.append(val_forcing)

        test_branch_parts.append(test_branch)
        test_target_parts.append(test_target)
        test_current_parts.append(test_current)
        test_dt_parts.append(test_dt)
        test_nu_parts.append(test_nu)
        test_forcing_parts.append(test_forcing)

        per_file_metadata.append(
            {
                "source_file": trajectory.source_file,
                "reynolds": trajectory.reynolds,
                "burn_in_steps": int(config.burn_in_steps),
                "train_pairs": int(len(train_indices)),
                "val_pairs": int(len(val_indices)),
                "test_pairs": int(len(test_indices)),
            }
        )
        first_test_index = int(test_indices[0])
        rollout_cases.append(
            {
                "source_file": trajectory.source_file,
                "reynolds": float(trajectory.reynolds),
                "initial_sensor": trajectory.sensor_values[first_test_index].astype(np.float32),
                "future_sensor": trajectory.sensor_values[
                    first_test_index + 1 : first_test_index + config.horizon_steps + 1
                ].astype(np.float32),
                "future_physics": trajectory.physics_values[
                    first_test_index + 1 : first_test_index + config.horizon_steps + 1
                ].astype(np.float32),
                "future_dt": trajectory.dt[first_test_index : first_test_index + config.horizon_steps].astype(np.float32),
            }
        )

    if sensor_coords is None or physics_coords is None or physics_grid_shape is None or domain_length is None:
        raise ValueError("沒有可用的 DNS 資料。")
    if dt_reference is None:
        raise ValueError("無法推斷統一時間步距。")

    train_branch = np.concatenate(train_branch_parts, axis=0)
    train_target = np.concatenate(train_target_parts, axis=0)
    val_branch = np.concatenate(val_branch_parts, axis=0)
    val_target = np.concatenate(val_target_parts, axis=0)
    test_branch = np.concatenate(test_branch_parts, axis=0)
    test_target = np.concatenate(test_target_parts, axis=0)

    train_current_physics = np.concatenate(train_current_parts, axis=0)
    val_current_physics = np.concatenate(val_current_parts, axis=0)
    test_current_physics = np.concatenate(test_current_parts, axis=0)
    train_dt = np.concatenate(train_dt_parts, axis=0)
    val_dt = np.concatenate(val_dt_parts, axis=0)
    test_dt = np.concatenate(test_dt_parts, axis=0)
    train_nu = np.concatenate(train_nu_parts, axis=0)
    val_nu = np.concatenate(val_nu_parts, axis=0)
    test_nu = np.concatenate(test_nu_parts, axis=0)
    train_forcing = np.concatenate(train_forcing_parts, axis=0)
    val_forcing = np.concatenate(val_forcing_parts, axis=0)
    test_forcing = np.concatenate(test_forcing_parts, axis=0)

    train_branch, val_branch, test_branch, train_target, val_target, test_target, stats = _normalize_dataset(
        train_branch, val_branch, test_branch, train_target, val_target, test_target
    )

    supervised_eval_time = float(dt_reference * config.horizon_steps)
    sensor_coords_t0 = np.concatenate(
        [sensor_coords.astype(np.float32), np.zeros((len(sensor_coords), 1), dtype=np.float32)],
        axis=1,
    )
    sensor_coords_teval = np.concatenate(
        [sensor_coords.astype(np.float32), np.full((len(sensor_coords), 1), supervised_eval_time, dtype=np.float32)],
        axis=1,
    )

    metadata = {
        "config": json.dumps(asdict(config), ensure_ascii=False),
        "files": json.dumps([str(path) for path in data_files], ensure_ascii=False),
        "per_file": json.dumps(per_file_metadata, ensure_ascii=False),
        "train_samples": int(train_branch.shape[0]),
        "val_samples": int(val_branch.shape[0]),
        "test_samples": int(test_branch.shape[0]),
        "branch_dim": int(train_branch.shape[1]),
        "trunk_points": int(sensor_coords.shape[0]),
        "target_dim": int(train_target.shape[1]),
        "physics_points": int(physics_coords.shape[0]),
        "physics_grid_shape": np.asarray(physics_grid_shape, dtype=np.int64),
        "sensor_indices": sensor_indices,
        "sensor_coords": sensor_coords,
        "physics_coords": physics_coords,
        "physics_domain_length": float(domain_length),
        "supervised_eval_time": supervised_eval_time,
        **stats,
    }

    return PreparedDataset(
        train_x=(train_branch.astype(np.float32), sensor_coords_t0),
        train_y=train_target.astype(np.float32),
        val_x=(val_branch.astype(np.float32), sensor_coords_teval),
        val_y=val_target.astype(np.float32),
        test_x=(test_branch.astype(np.float32), sensor_coords_teval),
        test_y=test_target.astype(np.float32),
        train_current_physics=train_current_physics.astype(np.float32),
        val_current_physics=val_current_physics.astype(np.float32),
        test_current_physics=test_current_physics.astype(np.float32),
        train_dt=train_dt.astype(np.float32),
        val_dt=val_dt.astype(np.float32),
        test_dt=test_dt.astype(np.float32),
        train_nu=train_nu.astype(np.float32),
        val_nu=val_nu.astype(np.float32),
        test_nu=test_nu.astype(np.float32),
        train_forcing=train_forcing.astype(np.float32),
        val_forcing=val_forcing.astype(np.float32),
        test_forcing=test_forcing.astype(np.float32),
        physics_coords=physics_coords.astype(np.float32),
        physics_grid_shape=physics_grid_shape,
        physics_domain_length=float(domain_length),
        sensor_indices=sensor_indices.astype(np.int64),
        sensor_coords=sensor_coords.astype(np.float32),
        branch_mean=stats["branch_mean"],
        branch_std=stats["branch_std"],
        reynolds_mean=stats["reynolds_mean"],
        reynolds_std=stats["reynolds_std"],
        target_mean=stats["target_mean"],
        target_std=stats["target_std"],
        rollout_cases=rollout_cases,
        metadata=metadata,
        horizon_steps=int(config.horizon_steps),
        seed=int(config.seed),
    )


class PhysicsInformedTripleCartesianProd(Data):
    """What: 結合 sparse supervised loss 與 vorticity residual 的 DeepONet 資料類。

    Why:
        `DeepXDE` 已經負責 optimizer、checkpoint 與 metrics；
        只要把 physics residual 包進 `Data.losses()`，就能沿用既有訓練流程。
    """

    def __init__(
        self,
        dataset: PreparedDataset,
        physics_time_samples: int = 4,
        physics_branch_batch_size: int | None = None,
    ):
        self.train_x = dataset.train_x
        self.train_y = dataset.train_y
        self.test_x = dataset.val_x
        self.test_y = dataset.val_y
        self.train_current_physics = dataset.train_current_physics
        self.test_current_physics = dataset.val_current_physics
        self.train_dt = dataset.train_dt
        self.test_dt = dataset.val_dt
        self.train_nu = dataset.train_nu
        self.test_nu = dataset.val_nu
        self.train_forcing = dataset.train_forcing
        self.test_forcing = dataset.val_forcing
        self.physics_coords = dataset.physics_coords
        self.physics_grid_shape = dataset.physics_grid_shape
        self.physics_domain_length = dataset.physics_domain_length
        self.target_mean = float(dataset.target_mean)
        self.target_std = float(dataset.target_std)
        self.horizon_steps = int(dataset.horizon_steps)
        self.seed = int(dataset.seed)
        self.physics_time_samples = int(physics_time_samples)
        if self.physics_time_samples <= 0:
            raise ValueError("physics_time_samples 必須大於 0。")
        if physics_branch_batch_size is not None and int(physics_branch_batch_size) <= 0:
            raise ValueError("physics_branch_batch_size 必須大於 0。")
        self.physics_branch_batch_size = (
            None if physics_branch_batch_size is None else int(physics_branch_batch_size)
        )

        if len(self.train_x[0]) != self.train_y.shape[0] or len(self.train_x[1]) != self.train_y.shape[1]:
            raise ValueError("訓練資料不符合 Cartesian product 格式。")
        if len(self.test_x[0]) != self.test_y.shape[0] or len(self.test_x[1]) != self.test_y.shape[1]:
            raise ValueError("測試資料不符合 Cartesian product 格式。")

        self.branch_sampler = BatchSampler(len(self.train_x[0]), shuffle=True)
        self._last_train_indices = np.arange(len(self.train_x[0]), dtype=np.int64)
        self._physics_rng = np.random.default_rng(self.seed)

        self._spectral_kx: torch.Tensor | None = None
        self._spectral_ky: torch.Tensor | None = None
        self._spectral_k2: torch.Tensor | None = None
        self._spectral_nonzero_mask: torch.Tensor | None = None
        self._spectral_kx_complex: torch.Tensor | None = None
        self._spectral_ky_complex: torch.Tensor | None = None
        self._spectral_k2_complex: torch.Tensor | None = None
        self._tensor_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    def _ensure_spectral_cache(self, device: torch.device, dtype: torch.dtype) -> None:
        """What: 在指定 device 上建立頻域常數。"""

        if self._spectral_kx is not None and self._spectral_kx.device == device and self._spectral_kx.dtype == dtype:
            return

        grid_x, grid_y = self.physics_grid_shape
        if grid_x != grid_y:
            raise ValueError("physics grid 必須是正方形，才能使用目前的 spectral residual。")
        dx = self.physics_domain_length / grid_x
        freq = 2.0 * np.pi * np.fft.fftfreq(grid_x, d=dx)
        kx, ky = np.meshgrid(freq, freq, indexing="ij")
        k2 = kx**2 + ky**2
        nonzero = k2 > 0.0

        self._spectral_kx = torch.as_tensor(kx, dtype=dtype, device=device)
        self._spectral_ky = torch.as_tensor(ky, dtype=dtype, device=device)
        self._spectral_k2 = torch.as_tensor(k2, dtype=dtype, device=device)
        self._spectral_nonzero_mask = torch.as_tensor(nonzero, dtype=torch.bool, device=device)
        complex_dtype = (
            torch.complex64 if dtype in (torch.float16, torch.float32, torch.bfloat16) else torch.complex128
        )
        self._spectral_kx_complex = self._spectral_kx.to(complex_dtype)
        self._spectral_ky_complex = self._spectral_ky.to(complex_dtype)
        self._spectral_k2_complex = self._spectral_k2.to(complex_dtype)

    def _ensure_tensor_cache(self, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        """What: 快取 physics loss 需要的靜態 tensor，避免每步重複轉換。"""

        key = (str(device), str(dtype))
        cached = self._tensor_cache.get(key)
        if cached is not None:
            return cached

        grid_size = self.physics_grid_shape[0]
        cached = {
            "physics_coords_xy": torch.as_tensor(self.physics_coords, dtype=dtype, device=device),
            "train_dt": torch.as_tensor(self.train_dt.reshape(-1), dtype=dtype, device=device),
            "test_dt": torch.as_tensor(self.test_dt.reshape(-1), dtype=dtype, device=device),
            "train_nu": torch.as_tensor(self.train_nu.reshape(-1), dtype=dtype, device=device),
            "test_nu": torch.as_tensor(self.test_nu.reshape(-1), dtype=dtype, device=device),
            "train_forcing": torch.as_tensor(
                self.train_forcing.reshape(-1, grid_size, grid_size), dtype=dtype, device=device
            ),
            "test_forcing": torch.as_tensor(
                self.test_forcing.reshape(-1, grid_size, grid_size), dtype=dtype, device=device
            ),
        }
        self._tensor_cache[key] = cached
        return cached

    def _physics_loss(
        self,
        model: dde.Model,
        branch_inputs: torch.Tensor,
        physics_coords_xy: torch.Tensor,
        delta_t_tensor: torch.Tensor,
        nu_tensor: torch.Tensor,
        forcing_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """What: 用連續時間 trunk + AD 時間導數計算 Kolmogorov PDE residual。"""

        device = branch_inputs.device
        dtype = branch_inputs.dtype
        self._ensure_spectral_cache(device, dtype)

        grid_size = self.physics_grid_shape[0]
        num_samples = int(branch_inputs.shape[0])
        t_max = delta_t_tensor * float(self.horizon_steps)
        if self.physics_time_samples == 1:
            times_matrix = t_max.unsqueeze(1)
        else:
            times_matrix = torch.rand((num_samples, self.physics_time_samples), dtype=dtype, device=device)
            times_matrix = times_matrix * t_max.unsqueeze(1)

        loss_sum = torch.zeros((), dtype=dtype, device=device)
        loss_count = 0
        for batch_idx in range(num_samples):
            times = times_matrix[batch_idx]
            sample_branch = branch_inputs[batch_idx : batch_idx + 1]
            sample_nu = nu_tensor[batch_idx]
            sample_forcing = forcing_tensor[batch_idx]

            for t_scalar in times:
                t_col = torch.ones((physics_coords_xy.shape[0], 1), dtype=dtype, device=device) * t_scalar
                trunk_coords = torch.cat([physics_coords_xy, t_col], dim=1).requires_grad_(True)

                pred_norm = model.net((sample_branch, trunk_coords))[0]
                omega_flat = pred_norm * self.target_std + self.target_mean
                grads = torch.autograd.grad(
                    omega_flat.sum(),
                    trunk_coords,
                    create_graph=True,
                )[0]
                omega_x = grads[:, 0].reshape(grid_size, grid_size)
                omega_y = grads[:, 1].reshape(grid_size, grid_size)
                omega_t = grads[:, 2].reshape(grid_size, grid_size)
                omega = omega_flat.reshape(grid_size, grid_size)

                omega_hat = torch.fft.fft2(omega)
                if self._spectral_kx_complex is None or self._spectral_ky_complex is None or self._spectral_k2_complex is None:
                    raise RuntimeError("spectral cache 尚未初始化。")
                kx = self._spectral_kx_complex
                ky = self._spectral_ky_complex
                k2 = self._spectral_k2_complex

                psi_hat = torch.zeros_like(omega_hat)
                psi_hat[self._spectral_nonzero_mask] = (
                    omega_hat[self._spectral_nonzero_mask] / k2[self._spectral_nonzero_mask]
                )

                u = torch.fft.ifft2(1j * ky * psi_hat).real
                v = torch.fft.ifft2(-1j * kx * psi_hat).real
                laplace_omega = torch.fft.ifft2(-(kx**2 + ky**2) * omega_hat).real
                residual = omega_t + u * omega_x + v * omega_y - sample_nu * laplace_omega - sample_forcing
                loss_sum = loss_sum + torch.mean(residual**2)
                loss_count += 1
        return loss_sum / float(max(1, loss_count))

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        supervised_loss = loss_fn(targets, outputs)
        branch_inputs = inputs[0]
        indices = self._last_train_indices
        if self.physics_branch_batch_size is not None and len(indices) > self.physics_branch_batch_size:
            local_idx = self._physics_rng.choice(
                len(indices),
                size=self.physics_branch_batch_size,
                replace=False,
            )
            branch_inputs = branch_inputs[local_idx]
            indices = indices[local_idx]
        cached = self._ensure_tensor_cache(branch_inputs.device, branch_inputs.dtype)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=branch_inputs.device)
        physics_loss = self._physics_loss(
            model=model,
            branch_inputs=branch_inputs,
            physics_coords_xy=cached["physics_coords_xy"],
            delta_t_tensor=cached["train_dt"].index_select(0, index_tensor),
            nu_tensor=cached["train_nu"].index_select(0, index_tensor),
            forcing_tensor=cached["train_forcing"].index_select(0, index_tensor),
        )
        return [supervised_loss, physics_loss]

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        supervised_loss = loss_fn(targets, outputs)
        branch_inputs = inputs[0]
        cached = self._ensure_tensor_cache(branch_inputs.device, branch_inputs.dtype)
        physics_loss = self._physics_loss(
            model=model,
            branch_inputs=branch_inputs,
            physics_coords_xy=cached["physics_coords_xy"],
            delta_t_tensor=cached["test_dt"],
            nu_tensor=cached["test_nu"],
            forcing_tensor=cached["test_forcing"],
        )
        return [supervised_loss, physics_loss]

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            self._last_train_indices = np.arange(len(self.train_y), dtype=np.int64)
            return self.train_x, self.train_y
        if isinstance(batch_size, (tuple, list)):
            raise ValueError("目前只支援 branch batch size，不支援 `(branch, trunk)` 雙批次。")
        indices = self.branch_sampler.get_next(batch_size)
        self._last_train_indices = indices
        return (self.train_x[0][indices], self.train_x[1]), self.train_y[indices]

    def test(self):
        return self.test_x, self.test_y
