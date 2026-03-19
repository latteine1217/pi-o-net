"""DNS sensor loading and physics-informed data utilities.

What:
    從 Kolmogorov DNS 時間序列中抽取固定隨機 sensors，建立 DeepONet 的觀測資料，
    並同時準備在規則 collocation grid 上計算 Navier-Stokes + continuity residual 所需的物理量。
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

    field: str = "uvp"
    num_sensors: int = 1000
    history_steps: int = 1
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
    forcing_u: Array
    forcing_v: Array
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
    train_forcing_u: Array
    val_forcing_u: Array
    test_forcing_u: Array
    train_forcing_v: Array
    val_forcing_v: Array
    test_forcing_v: Array
    physics_coords: Array
    physics_grid_shape: tuple[int, int]
    physics_domain_length: float
    sensor_indices: Array
    sensor_coords: Array
    branch_mean: float
    branch_std: float
    reynolds_mean: float
    reynolds_std: float
    target_means: Array   # shape (3,): u / v / p 各自的 mean
    target_stds: Array    # shape (3,): u / v / p 各自的 std
    rollout_cases: list[dict[str, object]]
    metadata: dict[str, object]
    history_steps: int
    horizon_steps: int
    seed: int


def resolve_data_files(explicit_files: list[str] | None) -> list[Path]:
    """What: 解析要使用的 DNS 檔案清單。

    Why:
        physics loss 會直接用 DNS 的 `u,v,p` 計算 NS + continuity，
        上游資料就應明確限定為 DNS。
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

    if field != "uvp":
        raise ValueError("目前僅支援 `field=uvp`（直接輸出 u,v,p）。")
    required = {"x", "y", "time", "config", "u", "v", "p"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"{source_file} 缺少 DNS 欄位: {sorted(missing)}")


def _build_full_coords(x: Array, y: Array) -> Array:
    """What: 建立完整規則網格座標。"""

    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack([xx, yy], axis=-1)


def _build_component_trunk_coords(coords_xy: Array, time_value: float, num_components: int = 3) -> Array:
    """What: 將 `(x,y,t)` 擴成 component-aware trunk `(x,y,t,c)`。"""

    xy = np.asarray(coords_xy, dtype=np.float32)
    n_points = xy.shape[0]
    t_col = np.full((n_points, 1), np.float32(time_value), dtype=np.float32)
    base = np.concatenate([xy, t_col], axis=1)
    chunks: list[Array] = []
    for comp in range(num_components):
        comp_col = np.full((n_points, 1), np.float32(comp), dtype=np.float32)
        chunks.append(np.concatenate([base, comp_col], axis=1))
    return np.concatenate(chunks, axis=0).astype(np.float32)


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


def _forcing_velocity(config: dict, physics_coords_grid: Array) -> tuple[Array, Array]:
    """What: 建立 Navier-Stokes 動量方程用的體積力 `(f_x, f_y)`。"""

    amplitude = float(config.get("A", 0.1))
    forcing_wavenumber = float(config.get("k_f", 4.0))
    y = physics_coords_grid[..., 1]
    forcing_u = (amplitude * np.sin(forcing_wavenumber * y)).astype(np.float32)
    forcing_v = np.zeros_like(forcing_u, dtype=np.float32)
    return forcing_u, forcing_v


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
    u_field = np.asarray(payload["u"], dtype=np.float32)
    v_field = np.asarray(payload["v"], dtype=np.float32)
    p_field = np.asarray(payload["p"], dtype=np.float32)
    dns_config = dict(payload["config"])

    full_coords = _build_full_coords(x, y)
    flat_coords = full_coords.reshape(-1, 2)
    flat_u = u_field.reshape(u_field.shape[0], -1)
    flat_v = v_field.reshape(v_field.shape[0], -1)
    flat_p = p_field.reshape(p_field.shape[0], -1)

    if sensor_indices is None:
        sensor_indices = _sample_sensor_indices(
            num_points=flat_coords.shape[0],
            num_sensors=config.num_sensors,
            seed=config.seed,
        )
    sensor_coords = flat_coords[sensor_indices]
    sensor_u = flat_u[:, sensor_indices]
    sensor_v = flat_v[:, sensor_indices]
    sensor_p = flat_p[:, sensor_indices]
    sensor_values = np.concatenate([sensor_u, sensor_v, sensor_p], axis=1)

    physics_stride = max(1, int(config.physics_stride))
    physics_u = u_field[:, ::physics_stride, ::physics_stride]
    physics_v = v_field[:, ::physics_stride, ::physics_stride]
    physics_p = p_field[:, ::physics_stride, ::physics_stride]
    physics_coords_grid = full_coords[::physics_stride, ::physics_stride]
    forcing_u, forcing_v = _forcing_velocity(dns_config, physics_coords_grid)

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
        physics_values=np.concatenate(
            [
                physics_u.reshape(physics_u.shape[0], -1),
                physics_v.reshape(physics_v.shape[0], -1),
                physics_p.reshape(physics_p.shape[0], -1),
            ],
            axis=1,
        ).astype(np.float32),
        forcing_u=forcing_u.reshape(-1).astype(np.float32),
        forcing_v=forcing_v.reshape(-1).astype(np.float32),
        physics_grid_shape=(physics_u.shape[1], physics_u.shape[2]),
        domain_length=float(dns_config.get("L", float(x[-1] - x[0] + (x[1] - x[0])))),
    )
    return trajectory, sensor_indices


def _split_indices(
    num_pairs: int,
    train_ratio: float,
    val_ratio: float,
    temporal_stride: int,
    burn_in_steps: int,
    history_steps: int,
) -> tuple[Array, Array, Array]:
    """What: 建立 train/val/test 的時間索引。"""

    if burn_in_steps < 0:
        raise ValueError("burn_in_steps 不可為負數。")
    if history_steps <= 0:
        raise ValueError("history_steps 必須大於 0。")
    start_index = min(max(int(burn_in_steps), int(history_steps) - 1), num_pairs)
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
    history_steps: int,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """What: 將時間配對轉成觀測資料與 physics loss 所需陣列。"""

    token_dim = trajectory.sensor_values.shape[1] + 1
    branch = np.zeros((len(pair_indices), history_steps, token_dim), dtype=np.float32)
    target = np.zeros((len(pair_indices), trajectory.sensor_values.shape[1]), dtype=np.float32)
    current_physics = np.zeros((len(pair_indices), trajectory.physics_values.shape[1]), dtype=np.float32)
    dt = np.zeros((len(pair_indices), 1), dtype=np.float32)
    nu = np.full((len(pair_indices), 1), trajectory.nu, dtype=np.float32)

    for row, start in enumerate(pair_indices):
        history_start = start - history_steps + 1
        history_end = start + 1
        if history_start < 0:
            raise ValueError("history_steps 超過可用歷史長度，請提高 burn_in_steps 或降低 history_steps。")
        branch[row, :, :-1] = trajectory.sensor_values[history_start:history_end]
        branch[row, :, -1] = np.float32(trajectory.reynolds)
        target[row] = trajectory.sensor_values[start + target_offset]
        current_physics[row] = trajectory.physics_values[start]
        dt[row, 0] = trajectory.dt[start]
    forcing_u = np.repeat(trajectory.forcing_u[None, :], len(pair_indices), axis=0)
    forcing_v = np.repeat(trajectory.forcing_v[None, :], len(pair_indices), axis=0)
    return (
        branch,
        target,
        current_physics,
        dt,
        nu,
        forcing_u.astype(np.float32),
        forcing_v.astype(np.float32),
    )


def _normalize_dataset(
    train_branch: Array,
    val_branch: Array,
    test_branch: Array,
    train_target: Array,
    val_target: Array,
    test_target: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, dict[str, object]]:
    """What: 用訓練集統計量正規化觀測資料。

    Why:
        target 改為 per-component 正規化（u/v/p 各自獨立）。
        NS 流場中各分量的量級可能不同（例如壓力量級不同於速度），
        共用同一組 mean/std 會扭曲 loss landscape 並導致 physics residual 計算失真。
    """

    branch_mean = float(np.mean(train_branch[:, :, :-1]))
    branch_std = float(np.std(train_branch[:, :, :-1]) + 1e-6)
    reynolds_mean = float(np.mean(train_branch[:, :, -1]))
    reynolds_std = float(np.std(train_branch[:, :, -1]) + 1e-6)

    n_pts = train_target.shape[1] // 3  # 每個分量的 sensor 數
    target_means = np.array(
        [float(np.mean(train_target[:, c * n_pts : (c + 1) * n_pts])) for c in range(3)],
        dtype=np.float64,
    )
    target_stds = np.array(
        [float(np.std(train_target[:, c * n_pts : (c + 1) * n_pts]) + 1e-6) for c in range(3)],
        dtype=np.float64,
    )

    train_branch = train_branch.copy()
    val_branch = val_branch.copy()
    test_branch = test_branch.copy()
    train_target = train_target.copy()
    val_target = val_target.copy()
    test_target = test_target.copy()

    train_branch[:, :, :-1] = (train_branch[:, :, :-1] - branch_mean) / branch_std
    val_branch[:, :, :-1] = (val_branch[:, :, :-1] - branch_mean) / branch_std
    test_branch[:, :, :-1] = (test_branch[:, :, :-1] - branch_mean) / branch_std
    train_branch[:, :, -1] = (train_branch[:, :, -1] - reynolds_mean) / reynolds_std
    val_branch[:, :, -1] = (val_branch[:, :, -1] - reynolds_mean) / reynolds_std
    test_branch[:, :, -1] = (test_branch[:, :, -1] - reynolds_mean) / reynolds_std

    for c, (m, s) in enumerate(zip(target_means, target_stds)):
        sl = slice(c * n_pts, (c + 1) * n_pts)
        train_target[:, sl] = (train_target[:, sl] - m) / s
        val_target[:, sl] = (val_target[:, sl] - m) / s
        test_target[:, sl] = (test_target[:, sl] - m) / s

    stats: dict[str, object] = {
        "branch_mean": branch_mean,
        "branch_std": branch_std,
        "target_means": target_means,
        "target_stds": target_stds,
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
    train_forcing_u_parts: list[Array] = []
    train_forcing_v_parts: list[Array] = []

    val_branch_parts: list[Array] = []
    val_target_parts: list[Array] = []
    val_current_parts: list[Array] = []
    val_dt_parts: list[Array] = []
    val_nu_parts: list[Array] = []
    val_forcing_u_parts: list[Array] = []
    val_forcing_v_parts: list[Array] = []

    test_branch_parts: list[Array] = []
    test_target_parts: list[Array] = []
    test_current_parts: list[Array] = []
    test_dt_parts: list[Array] = []
    test_nu_parts: list[Array] = []
    test_forcing_u_parts: list[Array] = []
    test_forcing_v_parts: list[Array] = []

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
            config.history_steps,
        )

        (
            train_branch,
            train_target,
            train_current,
            train_dt,
            train_nu,
            train_forcing_u,
            train_forcing_v,
        ) = _build_sample_arrays(
            trajectory, train_indices, target_offset=0, history_steps=config.history_steps
        )
        (
            val_branch,
            val_target,
            val_current,
            val_dt,
            val_nu,
            val_forcing_u,
            val_forcing_v,
        ) = _build_sample_arrays(
            trajectory, val_indices, target_offset=config.horizon_steps, history_steps=config.history_steps
        )
        (
            test_branch,
            test_target,
            test_current,
            test_dt,
            test_nu,
            test_forcing_u,
            test_forcing_v,
        ) = _build_sample_arrays(
            trajectory, test_indices, target_offset=config.horizon_steps, history_steps=config.history_steps
        )

        train_branch_parts.append(train_branch)
        train_target_parts.append(train_target)
        train_current_parts.append(train_current)
        train_dt_parts.append(train_dt)
        train_nu_parts.append(train_nu)
        train_forcing_u_parts.append(train_forcing_u)
        train_forcing_v_parts.append(train_forcing_v)

        val_branch_parts.append(val_branch)
        val_target_parts.append(val_target)
        val_current_parts.append(val_current)
        val_dt_parts.append(val_dt)
        val_nu_parts.append(val_nu)
        val_forcing_u_parts.append(val_forcing_u)
        val_forcing_v_parts.append(val_forcing_v)

        test_branch_parts.append(test_branch)
        test_target_parts.append(test_target)
        test_current_parts.append(test_current)
        test_dt_parts.append(test_dt)
        test_nu_parts.append(test_nu)
        test_forcing_u_parts.append(test_forcing_u)
        test_forcing_v_parts.append(test_forcing_v)

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
                "initial_history": trajectory.sensor_values[
                    first_test_index - config.history_steps + 1 : first_test_index + 1
                ].astype(np.float32),
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
    train_forcing_u = np.concatenate(train_forcing_u_parts, axis=0)
    val_forcing_u = np.concatenate(val_forcing_u_parts, axis=0)
    test_forcing_u = np.concatenate(test_forcing_u_parts, axis=0)
    train_forcing_v = np.concatenate(train_forcing_v_parts, axis=0)
    val_forcing_v = np.concatenate(val_forcing_v_parts, axis=0)
    test_forcing_v = np.concatenate(test_forcing_v_parts, axis=0)

    train_branch, val_branch, test_branch, train_target, val_target, test_target, stats = _normalize_dataset(
        train_branch, val_branch, test_branch, train_target, val_target, test_target
    )

    supervised_eval_time = float(dt_reference * config.horizon_steps)
    sensor_coords_t0 = _build_component_trunk_coords(sensor_coords, time_value=0.0, num_components=3)
    sensor_coords_teval = _build_component_trunk_coords(
        sensor_coords, time_value=supervised_eval_time, num_components=3
    )

    metadata = {
        "config": json.dumps(asdict(config), ensure_ascii=False),
        "files": json.dumps([str(path) for path in data_files], ensure_ascii=False),
        "per_file": json.dumps(per_file_metadata, ensure_ascii=False),
        "train_samples": int(train_branch.shape[0]),
        "val_samples": int(val_branch.shape[0]),
        "test_samples": int(test_branch.shape[0]),
        "branch_dim": int(train_branch.shape[-1]),
        "branch_shape": np.asarray(train_branch.shape[1:], dtype=np.int64),
        "trunk_points": int(sensor_coords_t0.shape[0]),
        "target_dim": int(train_target.shape[1]),
        "physics_points": int(physics_coords.shape[0]),
        "physics_components": 3,
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
        train_forcing_u=train_forcing_u.astype(np.float32),
        val_forcing_u=val_forcing_u.astype(np.float32),
        test_forcing_u=test_forcing_u.astype(np.float32),
        train_forcing_v=train_forcing_v.astype(np.float32),
        val_forcing_v=val_forcing_v.astype(np.float32),
        test_forcing_v=test_forcing_v.astype(np.float32),
        physics_coords=physics_coords.astype(np.float32),
        physics_grid_shape=physics_grid_shape,
        physics_domain_length=float(domain_length),
        sensor_indices=sensor_indices.astype(np.int64),
        sensor_coords=sensor_coords.astype(np.float32),
        branch_mean=stats["branch_mean"],
        branch_std=stats["branch_std"],
        reynolds_mean=stats["reynolds_mean"],
        reynolds_std=stats["reynolds_std"],
        target_means=np.asarray(stats["target_means"], dtype=np.float64),
        target_stds=np.asarray(stats["target_stds"], dtype=np.float64),
        rollout_cases=rollout_cases,
        metadata=metadata,
        history_steps=int(config.history_steps),
        horizon_steps=int(config.horizon_steps),
        seed=int(config.seed),
    )


class PhysicsInformedTripleCartesianProd(Data):
    """What: 結合 sparse supervised loss 與 NS+continuity residual 的 DeepONet 資料類。

    Why:
        `DeepXDE` 已經負責 optimizer、checkpoint 與 metrics；
        只要把 physics residual 包進 `Data.losses()`，就能沿用既有訓練流程。
    """

    def __init__(
        self,
        dataset: PreparedDataset,
        physics_time_samples: int = 4,
        physics_branch_batch_size: int | None = None,
        disable_physics_loss: bool = False,
        physics_continuity_weight: float = 10.0,
        physics_causal_epsilon: float = 1.0,
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
        self.train_forcing_u = dataset.train_forcing_u
        self.test_forcing_u = dataset.val_forcing_u
        self.train_forcing_v = dataset.train_forcing_v
        self.test_forcing_v = dataset.val_forcing_v
        self.physics_coords = dataset.physics_coords
        self.physics_grid_shape = dataset.physics_grid_shape
        self.physics_domain_length = dataset.physics_domain_length
        self.target_means = np.asarray(dataset.target_means, dtype=np.float64)  # (3,): u/v/p
        self.target_stds = np.asarray(dataset.target_stds, dtype=np.float64)   # (3,): u/v/p
        self.horizon_steps = int(dataset.horizon_steps)
        self.seed = int(dataset.seed)
        self.physics_time_samples = int(physics_time_samples)
        self.disable_physics_loss = bool(disable_physics_loss)
        self.physics_continuity_weight = float(physics_continuity_weight)
        self.physics_causal_epsilon = float(physics_causal_epsilon)
        if self.physics_time_samples <= 0:
            raise ValueError("physics_time_samples 必須大於 0。")
        if self.physics_continuity_weight <= 0.0:
            raise ValueError("physics_continuity_weight 必須大於 0。")
        if self.physics_causal_epsilon < 0.0:
            raise ValueError("physics_causal_epsilon 不可小於 0。")
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

        self._tensor_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

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
            "train_forcing_u": torch.as_tensor(
                self.train_forcing_u.reshape(-1, grid_size, grid_size), dtype=dtype, device=device
            ),
            "test_forcing_u": torch.as_tensor(
                self.test_forcing_u.reshape(-1, grid_size, grid_size), dtype=dtype, device=device
            ),
            "train_forcing_v": torch.as_tensor(
                self.train_forcing_v.reshape(-1, grid_size, grid_size), dtype=dtype, device=device
            ),
            "test_forcing_v": torch.as_tensor(
                self.test_forcing_v.reshape(-1, grid_size, grid_size), dtype=dtype, device=device
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
        forcing_u_tensor: torch.Tensor,
        forcing_v_tensor: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """What: 計算 NS x/y 動量方程 + continuity 三項 residual MSE 並疊加。

        Why:
            DeepONet 的 Cartesian product 結構保證 pred[i] 僅依賴 trunk[i]，
            因此可將 T 個時間點 × 3 個分量打包成一個 trunk tensor，做單次 forward pass。
            舊做法每個 (sample, time) 做 3 次 forward + 6 次 backward；
            新做法每個 sample 做 1 次 forward + 5 次 backward，T 個時間點平行處理。
            Causal weighting 改用向量化 cumsum，消除 Python 內層 for-loop。
        """

        dtype = branch_inputs.dtype
        device = branch_inputs.device
        num_samples = int(branch_inputs.shape[0])
        n_pts = int(physics_coords_xy.shape[0])
        T = self.physics_time_samples

        # 取樣時間矩陣 (B, T)，排序以符合 causal weighting 的時序要求
        t_max = delta_t_tensor * float(self.horizon_steps)
        if T == 1:
            times_matrix = t_max.unsqueeze(1)
        else:
            times_matrix = torch.rand((num_samples, T), dtype=dtype, device=device) * t_max.unsqueeze(1)
        times_matrix = torch.sort(times_matrix, dim=1)[0]

        # 預先建立 per-component 統計量 tensor，避免在迴圈中重複建立
        comp_vals = torch.tensor([0.0, 1.0, 2.0], dtype=dtype, device=device)
        t_means = torch.as_tensor(self.target_means, dtype=dtype, device=device).reshape(1, 3, 1)
        t_stds = torch.as_tensor(self.target_stds, dtype=dtype, device=device).reshape(1, 3, 1)

        total_loss_x = branch_inputs.new_zeros(())
        total_loss_y = branch_inputs.new_zeros(())
        total_loss_c = branch_inputs.new_zeros(())

        for batch_idx in range(num_samples):
            times = times_matrix[batch_idx]  # (T,)
            sample_branch = branch_inputs[batch_idx : batch_idx + 1]
            sample_nu = nu_tensor[batch_idx]
            forcing_u_flat = forcing_u_tensor[batch_idx].reshape(-1)  # (n_pts,)
            forcing_v_flat = forcing_v_tensor[batch_idx].reshape(-1)  # (n_pts,)

            # 建立合併 trunk：(T, 3, n_pts, 4) → (T*3*n_pts, 4)
            # 每個位置 [t, c, j] 對應時間 t、分量 c、空間點 j
            xy = physics_coords_xy.unsqueeze(0).unsqueeze(0).expand(T, 3, n_pts, 2)
            t_col = times.reshape(T, 1, 1, 1).expand(T, 3, n_pts, 1)
            c_col = comp_vals.reshape(1, 3, 1, 1).expand(T, 3, n_pts, 1)
            trunk = torch.cat([xy, t_col, c_col], dim=-1).reshape(T * 3 * n_pts, 4).requires_grad_(True)

            # 單次 forward pass：branch (1,...) × trunk (T*3*n_pts, 4) → (T*3*n_pts,)
            pred_norm = model.net((sample_branch, trunk))[0]

            # Per-component 反正規化：pred_3d shape (T, 3, n_pts)
            pred_3d = pred_norm.reshape(T, 3, n_pts) * t_stds + t_means

            # 一次 backward 取所有一階導數 (T*3*n_pts, 4)
            first_grads = torch.autograd.grad(pred_3d.sum(), trunk, create_graph=True)[0]

            grads_4d = first_grads.reshape(T, 3, n_pts, 4)

            u_flat = pred_3d[:, 0, :]   # (T, n_pts)
            v_flat = pred_3d[:, 1, :]
            u_x = grads_4d[:, 0, :, 0]  # du/dx
            u_y = grads_4d[:, 0, :, 1]  # du/dy
            u_t = grads_4d[:, 0, :, 2]  # du/dt
            v_x = grads_4d[:, 1, :, 0]
            v_y = grads_4d[:, 1, :, 1]
            v_t = grads_4d[:, 1, :, 2]
            p_x = grads_4d[:, 2, :, 0]
            p_y = grads_4d[:, 2, :, 1]

            # 四次 backward 取二階導數（各覆蓋所有 T 個時間點，vs 舊做法 T×4 次）
            u_xx = torch.autograd.grad(u_x.sum(), trunk, create_graph=True)[0].reshape(T, 3, n_pts, 4)[:, 0, :, 0]
            u_yy = torch.autograd.grad(u_y.sum(), trunk, create_graph=True)[0].reshape(T, 3, n_pts, 4)[:, 0, :, 1]
            v_xx = torch.autograd.grad(v_x.sum(), trunk, create_graph=True)[0].reshape(T, 3, n_pts, 4)[:, 1, :, 0]
            v_yy = torch.autograd.grad(v_y.sum(), trunk, create_graph=True)[0].reshape(T, 3, n_pts, 4)[:, 1, :, 1]

            # NS residuals：(T, n_pts)
            residual_x = u_t + u_flat * u_x + v_flat * u_y + p_x - sample_nu * (u_xx + u_yy) - forcing_u_flat
            residual_y = v_t + u_flat * v_x + v_flat * v_y + p_y - sample_nu * (v_xx + v_yy) - forcing_v_flat
            residual_c = u_x + v_y

            # Per-time-step MSE：(T,)
            loss_x_t = residual_x.pow(2).mean(dim=1)
            loss_y_t = residual_y.pow(2).mean(dim=1)
            loss_c_t = residual_c.pow(2).mean(dim=1)
            total_t = loss_x_t + loss_y_t + self.physics_continuity_weight * loss_c_t

            # Causal weighting：w_t = exp(-eps * Σ_{s<t} total_s)，向量化取代 Python loop
            cumulative = torch.cat([total_t.new_zeros(1), torch.cumsum(total_t[:-1].detach(), dim=0)])
            weights = torch.exp(-self.physics_causal_epsilon * cumulative)  # (T,)

            # Per-sample 歸一化後累加（確保每個 sample 貢獻相同）
            safe_w = weights.sum().clamp(min=torch.finfo(dtype).eps)
            total_loss_x = total_loss_x + (weights * loss_x_t).sum() / safe_w
            total_loss_y = total_loss_y + (weights * loss_y_t).sum() / safe_w
            total_loss_c = total_loss_c + (weights * loss_c_t).sum() / safe_w

        loss_x = total_loss_x / num_samples
        loss_y = total_loss_y / num_samples
        loss_c = total_loss_c / num_samples
        total = loss_x + loss_y + self.physics_continuity_weight * loss_c
        if return_components:
            return total, {
                "ns_x_mse": float(loss_x.detach().cpu().item()),
                "ns_y_mse": float(loss_y.detach().cpu().item()),
                "continuity_mse": float(loss_c.detach().cpu().item()),
            }
        return total

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        supervised_loss = loss_fn(targets, outputs)
        if self.disable_physics_loss:
            return [supervised_loss, supervised_loss.new_zeros(())]
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
            forcing_u_tensor=cached["train_forcing_u"].index_select(0, index_tensor),
            forcing_v_tensor=cached["train_forcing_v"].index_select(0, index_tensor),
        )
        return [supervised_loss, physics_loss]

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        supervised_loss = loss_fn(targets, outputs)
        if self.disable_physics_loss:
            return [supervised_loss, supervised_loss.new_zeros(())]
        branch_inputs = inputs[0]
        cached = self._ensure_tensor_cache(branch_inputs.device, branch_inputs.dtype)
        physics_loss = self._physics_loss(
            model=model,
            branch_inputs=branch_inputs,
            physics_coords_xy=cached["physics_coords_xy"],
            delta_t_tensor=cached["test_dt"],
            nu_tensor=cached["test_nu"],
            forcing_u_tensor=cached["test_forcing_u"],
            forcing_v_tensor=cached["test_forcing_v"],
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
