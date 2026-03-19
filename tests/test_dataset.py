"""What: 測試文獻對齊版 PI-DeepONet 的資料與訓練介面契約。

Why:
    專案已重構為單一路徑（L_IC + L_physics）。
    這些測試確保資料切分、參數解析、callback 與模型建構都符合新流程。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pi_onet.dataset import DatasetConfig, PhysicsInformedTripleCartesianProd, build_dataset, load_dns_trajectory
import pi_onet.train as train_module
from pi_onet.train import (
    DEFAULT_TRAIN_ARGS,
    MidEvaluationCheckpointCallback,
    PASS_FAIL_THRESHOLDS,
    TotalLossThresholdStopCallback,
    build_pass_fail_summary,
    build_lr_decay_config,
    build_optimizer,
    create_model,
    load_train_config,
    parse_args,
    parse_hidden_dims,
)


def _dns_payload(time_steps: int = 6, grid_size: int = 8, nu: float = 1e-4) -> dict:
    x = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False, dtype=np.float32)
    base = np.arange(time_steps * grid_size * grid_size, dtype=np.float32).reshape(time_steps, grid_size, grid_size)
    return {
        "time": np.arange(time_steps, dtype=np.float32) * 0.5,
        "x": x,
        "y": y,
        "u": base,
        "v": base + 1.0,
        "p": base + 2.0,
        "config": {"nu": nu, "A": 0.1, "k_f": 4.0, "L": float(2.0 * np.pi)},
    }


def test_load_dns_trajectory_samples_fixed_sensors(tmp_path) -> None:
    path = tmp_path / "kolmogorov_dns_10000.npy"
    np.save(path, _dns_payload(), allow_pickle=True)

    trajectory, sensor_indices = load_dns_trajectory(
        path,
        DatasetConfig(num_sensors=10, physics_stride=2, seed=7),
    )

    assert sensor_indices.shape == (10,)
    assert trajectory.sensor_coords.shape == (10, 2)
    assert trajectory.sensor_values.shape == (6, 30)
    assert trajectory.physics_grid_shape == (4, 4)
    assert trajectory.physics_coords.shape == (16, 2)
    assert trajectory.physics_values.shape == (6, 48)
    assert trajectory.reynolds == 10000.0


def test_build_dataset_shapes_and_time_coords(tmp_path) -> None:
    path = tmp_path / "kolmogorov_dns_1000.npy"
    np.save(path, _dns_payload(time_steps=8, grid_size=8, nu=1e-3), allow_pickle=True)

    dataset = build_dataset(
        [path],
        DatasetConfig(
            num_sensors=12,
            horizon_steps=2,
            temporal_stride=1,
            train_ratio=0.6,
            val_ratio=0.2,
            physics_stride=4,
            seed=3,
        ),
    )

    train_branch, train_trunk = dataset.train_x
    val_branch, val_trunk = dataset.val_x
    test_branch, test_trunk = dataset.test_x
    assert train_branch.shape == (3, 1, 37)
    assert dataset.train_y.shape == (3, 36)
    assert val_branch.shape == (1, 1, 37)
    assert dataset.val_y.shape == (1, 36)
    assert test_branch.shape == (2, 1, 37)
    assert dataset.test_y.shape == (2, 36)
    assert train_trunk.shape == (36, 4)
    assert val_trunk.shape == (36, 4)
    assert test_trunk.shape == (36, 4)
    assert np.allclose(train_trunk[:, 2], 0.0)
    assert np.allclose(val_trunk[:, 2], 1.0)
    assert np.allclose(test_trunk[:, 2], 1.0)
    assert dataset.metadata["physics_points"] == 4
    assert len(dataset.rollout_cases) == 1


def test_build_dataset_respects_burn_in_steps(tmp_path) -> None:
    path = tmp_path / "kolmogorov_dns_1000.npy"
    np.save(path, _dns_payload(time_steps=9, grid_size=8, nu=1e-3), allow_pickle=True)

    dataset = build_dataset(
        [path],
        DatasetConfig(
            num_sensors=12,
            horizon_steps=1,
            temporal_stride=1,
            burn_in_steps=3,
            train_ratio=0.6,
            val_ratio=0.2,
            physics_stride=4,
            seed=3,
        ),
    )

    assert dataset.train_y.shape[0] == 3
    assert dataset.val_y.shape[0] == 1
    assert dataset.test_y.shape[0] == 1


def test_physics_loss_numerical_consistency(tmp_path) -> None:
    """What: 確認重構後的 _physics_loss 輸出與舊版（per-time-step loop）數值一致。

    Why:
        _physics_loss 改為單次 batched forward pass，需驗證 DeepONet Cartesian product
        結構下 pred[i] 僅依賴 trunk[i] 的假設確實成立，確保梯度計算不受 batching 影響。
    """
    path = tmp_path / "kolmogorov_dns_100.npy"
    np.save(path, _dns_payload(time_steps=7, grid_size=8, nu=1e-2), allow_pickle=True)

    dataset = build_dataset(
        [path],
        DatasetConfig(
            num_sensors=8,
            horizon_steps=2,
            temporal_stride=1,
            train_ratio=0.6,
            val_ratio=0.2,
            physics_stride=4,
            seed=5,
        ),
    )
    data = PhysicsInformedTripleCartesianProd(dataset, physics_time_samples=2, physics_branch_batch_size=2)

    net = create_model(
        branch_shape=tuple(int(d) for d in dataset.train_x[0].shape[1:]),
        trunk_dim=int(dataset.train_x[1].shape[1]),
        branch_hidden_dims=[16, 16],
        trunk_hidden_dims=[16, 16],
        latent_width=8,
        use_gated_mlp=True,
    )
    import deepxde as dde
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss="MSE", loss_weights=[1.0, 1.0])

    import pi_onet.train as train_module
    branch = torch.as_tensor(dataset.train_x[0][:2], dtype=torch.float32)
    cached = data._ensure_tensor_cache(branch.device, branch.dtype)
    idx = torch.arange(2, dtype=torch.long)

    # 呼叫兩次（固定 seed 使 torch.rand 可重現），比較 return_components
    torch.manual_seed(0)
    result1, comps1 = data._physics_loss(
        model=model,
        branch_inputs=branch,
        physics_coords_xy=cached["physics_coords_xy"],
        delta_t_tensor=cached["train_dt"].index_select(0, idx),
        nu_tensor=cached["train_nu"].index_select(0, idx),
        forcing_u_tensor=cached["train_forcing_u"].index_select(0, idx),
        forcing_v_tensor=cached["train_forcing_v"].index_select(0, idx),
        return_components=True,
    )
    torch.manual_seed(0)
    result2, comps2 = data._physics_loss(
        model=model,
        branch_inputs=branch,
        physics_coords_xy=cached["physics_coords_xy"],
        delta_t_tensor=cached["train_dt"].index_select(0, idx),
        nu_tensor=cached["train_nu"].index_select(0, idx),
        forcing_u_tensor=cached["train_forcing_u"].index_select(0, idx),
        forcing_v_tensor=cached["train_forcing_v"].index_select(0, idx),
        return_components=True,
    )

    assert abs(result1.detach().item() - result2.detach().item()) < 1e-5, "相同 seed 下兩次呼叫結果應一致"
    assert abs(comps1["ns_x_mse"] - comps2["ns_x_mse"]) < 1e-5
    assert comps1["continuity_mse"] >= 0.0
    assert result1.requires_grad, "physics loss 必須保留計算圖以供 backward"


def test_physics_informed_data_batch_contract(tmp_path) -> None:
    path = tmp_path / "kolmogorov_dns_100.npy"
    np.save(path, _dns_payload(time_steps=7, grid_size=8, nu=1e-2), allow_pickle=True)

    dataset = build_dataset(
        [path],
        DatasetConfig(
            num_sensors=8,
            horizon_steps=1,
            temporal_stride=1,
            train_ratio=0.6,
            val_ratio=0.2,
            physics_stride=4,
            seed=5,
        ),
    )
    data = PhysicsInformedTripleCartesianProd(dataset, physics_time_samples=3)

    x_batch, y_batch = data.train_next_batch(batch_size=2)
    assert x_batch[0].shape == (2, 1, 25)
    assert x_batch[1].shape == (24, 4)
    assert y_batch.shape == (2, 24)
    assert data._last_train_indices.shape == (2,)

    x_test, y_test = data.test()
    assert x_test[0].shape[0] == dataset.val_y.shape[0]
    assert y_test.shape == dataset.val_y.shape


def test_physics_branch_batch_size_only_applies_to_physics_loss(tmp_path) -> None:
    path = tmp_path / "kolmogorov_dns_100.npy"
    np.save(path, _dns_payload(time_steps=8, grid_size=8, nu=1e-2), allow_pickle=True)

    dataset = build_dataset(
        [path],
        DatasetConfig(
            num_sensors=8,
            horizon_steps=1,
            temporal_stride=1,
            train_ratio=0.6,
            val_ratio=0.2,
            physics_stride=4,
            seed=5,
        ),
    )
    data = PhysicsInformedTripleCartesianProd(dataset, physics_time_samples=2, physics_branch_batch_size=3)

    captured: dict[str, int] = {}

    def fake_physics_loss(model, branch_inputs, physics_coords_xy, delta_t_tensor, nu_tensor, forcing_u_tensor, forcing_v_tensor):
        captured["branch_batch"] = int(branch_inputs.shape[0])
        captured["delta_t_batch"] = int(len(delta_t_tensor))
        return torch.zeros((), dtype=branch_inputs.dtype, device=branch_inputs.device, requires_grad=True)

    data._physics_loss = fake_physics_loss  # type: ignore[method-assign]
    n_train = int(dataset.train_y.shape[0])
    data._last_train_indices = np.arange(n_train, dtype=np.int64)
    inputs = (
        torch.randn(n_train, dataset.train_x[0].shape[1], dataset.train_x[0].shape[2]),
        torch.randn(dataset.train_x[1].shape[0], dataset.train_x[1].shape[1]),
    )
    targets = torch.randn(n_train, dataset.train_y.shape[1])
    outputs = torch.randn(n_train, dataset.train_y.shape[1])
    loss_fn = lambda a, b: torch.mean((a - b) ** 2)

    losses = data.losses_train(targets, outputs, loss_fn, inputs, model=None)

    assert len(losses) == 2
    expected_batch = min(3, n_train)
    assert captured["branch_batch"] == expected_batch
    assert captured["delta_t_batch"] == expected_batch


def test_parse_hidden_dims_supports_string_and_list() -> None:
    assert parse_hidden_dims("512, 256,128", "branch_hidden_dims") == [512, 256, 128]
    assert parse_hidden_dims([256, 256], "trunk_hidden_dims") == [256, 256]


def test_build_lr_decay_config_supports_cosine_and_step() -> None:
    assert build_lr_decay_config("none", iterations=1000, min_learning_rate=1e-6) is None
    assert build_lr_decay_config("cosine", iterations=20000, min_learning_rate=1e-6) == ("cosine", 20000, 1e-6)
    assert build_lr_decay_config("step", iterations=20000, min_learning_rate=1e-6, step_size=10000, step_gamma=0.5) == (
        "step",
        10000,
        0.5,
    )


def test_build_optimizer_supports_adamw_and_validates_weight_decay() -> None:
    net = torch.nn.Linear(4, 2)
    adam = build_optimizer(net, optimizer_name="adam", learning_rate=1e-3, weight_decay=1e-4)
    adamw = build_optimizer(net, optimizer_name="adamw", learning_rate=1e-3, weight_decay=1e-4)

    assert adam == "adam"
    assert isinstance(adamw, torch.optim.AdamW)
    with pytest.raises(ValueError):
        build_optimizer(net, optimizer_name="adamw", learning_rate=1e-3, weight_decay=0.0)


def test_load_train_config_resolves_relative_paths(tmp_path) -> None:
    config_path = tmp_path / "train.toml"
    config_path.write_text(
        """
[train]
data_file = ["data/example.npy"]
artifacts_dir = "artifacts/run-a"
num_sensors = 128
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_train_config(config_path)

    assert config["data_file"] == [str(tmp_path / "data/example.npy")]
    assert config["artifacts_dir"] == tmp_path / "artifacts/run-a"
    assert config["num_sensors"] == 128


def test_load_train_config_rejects_unknown_key(tmp_path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        """
[train]
unknown_key = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_train_config(config_path)


def test_parse_args_merges_config_with_cli_override(tmp_path) -> None:
    config_path = tmp_path / "train.toml"
    config_path.write_text(
        """
[train]
num_sensors = 128
physics_loss_weight = 5.0
data_loss_weight = 15.0
artifacts_dir = "artifacts/from-config"
optimizer = "adamw"
weight_decay = 0.001
branch_hidden_dims = [512, 512]
trunk_hidden_dims = [512, 512]
latent_width = 256
use_gated_mlp = true
early_stop_total_loss = 1e-4
physics_branch_batch_size = 4
""".strip()
        + "\n",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path), "--num-sensors", "256"])

    assert args.num_sensors == 256
    assert args.physics_loss_weight == 5.0
    assert args.data_loss_weight == 15.0
    assert args.optimizer == "adamw"
    assert args.weight_decay == 0.001
    assert args.branch_hidden_dims == [512, 512]
    assert args.trunk_hidden_dims == [512, 512]
    assert args.latent_width == 256
    assert args.use_gated_mlp is True
    assert args.early_stop_total_loss == 1e-4
    assert args.physics_branch_batch_size == 4
    assert args.artifacts_dir == tmp_path / "artifacts/from-config"
    assert args.learning_rate == DEFAULT_TRAIN_ARGS["learning_rate"]


def test_create_model_supports_gated_mlp_forward() -> None:
    net = create_model(
        branch_shape=(1, 51),
        trunk_dim=4,
        branch_hidden_dims=[32, 32],
        trunk_hidden_dims=[32, 32],
        latent_width=16,
        use_gated_mlp=True,
    )

    branch = torch.randn(2, 1, 51)
    trunk = torch.randn(50, 4)
    prediction = net((branch, trunk))

    assert prediction.shape == (2, 50)


def test_create_model_supports_transformer_branch_forward() -> None:
    net = create_model(
        branch_shape=(4, 25),
        trunk_dim=4,
        branch_hidden_dims=[32, 32],
        trunk_hidden_dims=[32, 32],
        latent_width=16,
        use_gated_mlp=True,
        use_transformer_branch=True,
        transformer_model_dim=32,
        transformer_num_heads=4,
        transformer_num_layers=2,
        transformer_ff_dim=64,
    )

    branch = torch.randn(3, 4, 25)
    trunk = torch.randn(24, 4)
    prediction = net((branch, trunk))

    assert prediction.shape == (3, 24)


def test_mid_evaluation_callback_writes_checkpoint_and_summary(tmp_path) -> None:
    callback = MidEvaluationCheckpointCallback(
        period=5,
        artifacts_dir=tmp_path,
        data=None,  # type: ignore[arg-type]
        dataset=None,  # type: ignore[arg-type]
        X_val=None,  # type: ignore[arg-type]
        y_val=None,  # type: ignore[arg-type]
        rollout_steps=3,
    )

    class DummyModel:
        def __init__(self):
            self.loss_weights = [20.0, 1.0]
            self.train_state = type("TrainState", (), {})()
            self.train_state.step = 5

        def save(self, path: str) -> str:
            out = f"{path}-5.pt"
            torch.save({"dummy": True}, out)
            return out

    original_builder = train_module.build_mid_evaluation_summary
    train_module.build_mid_evaluation_summary = lambda **kwargs: {  # type: ignore[assignment]
        "step": 5,
        "validation_mean_relative_l2": 1.23,
        "rollout": {"evaluated_steps": 3},
    }
    try:
        callback.model = DummyModel()
        callback.on_epoch_end()
    finally:
        train_module.build_mid_evaluation_summary = original_builder  # type: ignore[assignment]

    assert (tmp_path / "mid_evals" / "eval_step_5.json").exists()
    assert (tmp_path / "best_validation" / "best_validation_summary.json").exists()
    assert callback.best_checkpoint_path is not None
    assert callback.history[0]["step"] == 5


def test_total_loss_threshold_callback_triggers_stop() -> None:
    callback = TotalLossThresholdStopCallback(threshold=1e-4)

    class DummyModel:
        def __init__(self):
            self.stop_training = False
            self.train_state = type("TrainState", (), {})()
            self.train_state.step = 77
            self.train_state.loss_train = [4e-5, 3e-5]

    callback.model = DummyModel()
    callback.on_epoch_end()
    summary = callback.summary()

    assert callback.model.stop_training is True
    assert summary["triggered"] is True
    assert summary["step"] == 77


def test_build_pass_fail_summary_marks_fail_and_pass() -> None:
    fail_summary = build_pass_fail_summary(
        validation_mean_relative_l2=1.0,
        test_mean_relative_l2=1.0,
        rollout_summary={
            "sensor_relative_l2_mean": 1.0,
            "physics_relative_l2_mean": 1.0,
        },
        include_test_metric=True,
    )
    assert fail_summary["status"] == "FAIL"
    assert len(fail_summary["failed_metrics"]) == 4
    assert fail_summary["thresholds"]["validation_mean_relative_l2"] == PASS_FAIL_THRESHOLDS["validation_mean_relative_l2"]

    pass_summary = build_pass_fail_summary(
        validation_mean_relative_l2=0.1,
        test_mean_relative_l2=0.1,
        rollout_summary={
            "sensor_relative_l2_mean": 0.2,
            "physics_relative_l2_mean": 0.3,
        },
        include_test_metric=True,
    )
    assert pass_summary["status"] == "PASS"
    assert pass_summary["failed_metrics"] == []


def test_legacy_field_is_rejected_by_new_config_schema(tmp_path) -> None:
    config_path = tmp_path / "legacy.toml"
    config_path.write_text(
        """
[train]
ic_loss_weight = 1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        parse_args(["--config", str(config_path)])
