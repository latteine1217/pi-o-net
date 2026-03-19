"""Evaluate a saved literature-aligned PI-DeepONet checkpoint.

What:
    針對文獻對齊版 PI-DeepONet（L_data + L_physics）重跑 validation/test/rollout 評估。
Why:
    最佳模型常出現在中途 checkpoint，需要獨立重評估以維持可重現比較。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde

from pi_onet.dataset import DatasetConfig, PhysicsInformedTripleCartesianProd, build_dataset, resolve_data_files
from pi_onet.train import (
    DEFAULT_TRAIN_ARGS,
    build_full_evaluation_summary,
    configure_torch_runtime,
    create_model,
    parse_hidden_dims,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """What: 解析 checkpoint 評估 CLI 參數。"""

    parser = argparse.ArgumentParser(description="Evaluate a saved DeepONet checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="要重評估的 checkpoint `.pt` 路徑。")
    parser.add_argument("--manifest", type=Path, default=None, help="實驗 manifest；預設從 checkpoint 所在 artifacts 目錄推斷。")
    parser.add_argument("--data-file", action="append", default=None, help="覆寫 manifest 的資料檔。")
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
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--physics-time-samples", type=int, default=None)
    parser.add_argument("--physics-branch-batch-size", type=int, default=None)
    parser.add_argument("--physics-continuity-weight", type=float, default=None)
    parser.add_argument("--physics-causal-epsilon", type=float, default=None)
    parser.add_argument("--time-fourier-modes", type=int, default=None)
    parser.add_argument("--branch-hidden-dims", type=str, default=None)
    parser.add_argument("--trunk-hidden-dims", type=str, default=None)
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    parser.add_argument("--output", type=Path, default=None, help="評估結果 JSON 輸出路徑。")
    return parser.parse_args()


def infer_manifest_path(checkpoint: Path, manifest: Path | None) -> Path:
    """What: 推斷 checkpoint 對應的 experiment manifest 路徑。"""

    if manifest is not None:
        return manifest
    artifacts_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    return artifacts_dir / "experiment_manifest.json"


def load_manifest_defaults(manifest_path: Path) -> dict[str, Any]:
    """What: 載入 manifest 裡的 configuration 區塊作為 CLI 預設值。"""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dict(payload.get("configuration", {}))


def merge_with_defaults(manifest_config: dict[str, Any]) -> dict[str, Any]:
    """What: 以 train 預設值補齊 manifest 欄位。"""

    merged = dict(DEFAULT_TRAIN_ARGS)
    merged.update(manifest_config)
    return merged


def get_config_value(args: argparse.Namespace, defaults: dict[str, Any], name: str) -> Any:
    """What: 以 CLI override 優先，否則退回 manifest 預設。"""

    value = getattr(args, name)
    return defaults.get(name) if value is None else value


def print_section(title: str) -> None:
    """What: 輸出結構化段落標題。"""

    print(f"=== {title} ===")


def main() -> None:
    """What: 載入指定 checkpoint 並輸出完整 evaluation。"""

    args = parse_args()
    requested_device = args.device if args.device is not None else "auto"
    runtime_device = configure_torch_runtime(requested_device)
    manifest_path = infer_manifest_path(args.checkpoint, args.manifest)
    defaults = merge_with_defaults(load_manifest_defaults(manifest_path))

    # "data_file" 是目前 manifest 的 key；"data_files" 相容舊版 manifest
    data_files = resolve_data_files(
        args.data_file if args.data_file is not None
        else defaults.get("data_file") or defaults.get("data_files")
    )
    config = DatasetConfig(
        field=get_config_value(args, defaults, "field") or "uvp",
        num_sensors=int(get_config_value(args, defaults, "num_sensors")),
        history_steps=int(get_config_value(args, defaults, "history_steps")),
        horizon_steps=int(get_config_value(args, defaults, "horizon_steps")),
        temporal_stride=int(get_config_value(args, defaults, "temporal_stride")),
        burn_in_steps=int(get_config_value(args, defaults, "burn_in_steps")),
        train_ratio=float(get_config_value(args, defaults, "train_ratio")),
        val_ratio=float(get_config_value(args, defaults, "val_ratio")),
        physics_stride=int(get_config_value(args, defaults, "physics_stride")),
        seed=int(get_config_value(args, defaults, "seed")),
    )
    rollout_steps = int(get_config_value(args, defaults, "rollout_steps"))
    physics_time_samples_value = get_config_value(args, defaults, "physics_time_samples")
    physics_time_samples = 4 if physics_time_samples_value is None else int(physics_time_samples_value)
    physics_branch_batch_size_value = get_config_value(args, defaults, "physics_branch_batch_size")
    physics_branch_batch_size = (
        None if physics_branch_batch_size_value in (None, "None") else int(physics_branch_batch_size_value)
    )
    physics_continuity_weight = float(get_config_value(args, defaults, "physics_continuity_weight"))
    physics_causal_epsilon = float(get_config_value(args, defaults, "physics_causal_epsilon"))
    time_fourier_modes = int(get_config_value(args, defaults, "time_fourier_modes"))
    data_loss_weight = float(get_config_value(args, defaults, "data_loss_weight"))
    physics_loss_weight = float(get_config_value(args, defaults, "physics_loss_weight"))
    branch_hidden_dims = parse_hidden_dims(
        get_config_value(args, defaults, "branch_hidden_dims"),
        "branch_hidden_dims",
    )
    trunk_hidden_dims = parse_hidden_dims(
        get_config_value(args, defaults, "trunk_hidden_dims"),
        "trunk_hidden_dims",
    )
    latent_width = int(get_config_value(args, defaults, "latent_width"))
    use_gated_mlp = bool(get_config_value(args, defaults, "use_gated_mlp"))
    use_transformer_branch = bool(get_config_value(args, defaults, "use_transformer_branch"))
    transformer_model_dim = int(get_config_value(args, defaults, "transformer_model_dim"))
    transformer_num_heads = int(get_config_value(args, defaults, "transformer_num_heads"))
    transformer_num_layers = int(get_config_value(args, defaults, "transformer_num_layers"))
    transformer_ff_dim = int(get_config_value(args, defaults, "transformer_ff_dim"))
    transformer_dropout = float(get_config_value(args, defaults, "transformer_dropout"))
    use_resnet_branch = bool(get_config_value(args, defaults, "use_resnet_branch"))
    trunk_rff_features_value = get_config_value(args, defaults, "trunk_rff_features")
    trunk_rff_features = 0 if trunk_rff_features_value is None else int(trunk_rff_features_value)
    trunk_rff_sigma_value = get_config_value(args, defaults, "trunk_rff_sigma")
    trunk_rff_sigma = 1.0 if trunk_rff_sigma_value is None else float(trunk_rff_sigma_value)

    print_section("Checkpoint Evaluation Configuration")
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "manifest": str(manifest_path),
                "data_files": [str(path) for path in data_files],
                "num_sensors": config.num_sensors,
                "history_steps": config.history_steps,
                "physics_stride": config.physics_stride,
                "rollout_steps": rollout_steps,
                "data_loss_weight": data_loss_weight,
                "physics_loss_weight": physics_loss_weight,
                "physics_time_samples": physics_time_samples,
                "physics_branch_batch_size": physics_branch_batch_size,
                "physics_continuity_weight": physics_continuity_weight,
                "physics_causal_epsilon": physics_causal_epsilon,
                "time_fourier_modes": time_fourier_modes,
                "branch_hidden_dims": branch_hidden_dims,
                "trunk_hidden_dims": trunk_hidden_dims,
                "latent_width": latent_width,
                "use_gated_mlp": use_gated_mlp,
                "use_transformer_branch": use_transformer_branch,
                "transformer_model_dim": transformer_model_dim,
                "transformer_num_heads": transformer_num_heads,
                "transformer_num_layers": transformer_num_layers,
                "transformer_ff_dim": transformer_ff_dim,
                "transformer_dropout": transformer_dropout,
                "use_resnet_branch": use_resnet_branch,
                "trunk_rff_features": trunk_rff_features,
                "trunk_rff_sigma": trunk_rff_sigma,
                "device": requested_device,
                "resolved_device": str(runtime_device),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    dataset = build_dataset(data_files=data_files, config=config)
    data = PhysicsInformedTripleCartesianProd(
        dataset,
        physics_time_samples=physics_time_samples,
        physics_branch_batch_size=physics_branch_batch_size,
        disable_physics_loss=(physics_loss_weight == 0.0),
        physics_continuity_weight=physics_continuity_weight,
        physics_causal_epsilon=physics_causal_epsilon,
    )
    net = create_model(
        branch_shape=tuple(int(dim) for dim in dataset.train_x[0].shape[1:]),
        trunk_dim=dataset.train_x[1].shape[1],
        branch_hidden_dims=branch_hidden_dims,
        trunk_hidden_dims=trunk_hidden_dims,
        latent_width=latent_width,
        use_gated_mlp=use_gated_mlp,
        time_fourier_modes=time_fourier_modes,
        use_transformer_branch=use_transformer_branch,
        transformer_model_dim=transformer_model_dim,
        transformer_num_heads=transformer_num_heads,
        transformer_num_layers=transformer_num_layers,
        transformer_ff_dim=transformer_ff_dim,
        transformer_dropout=transformer_dropout,
        use_resnet_branch=use_resnet_branch,
        trunk_rff_features=trunk_rff_features,
        trunk_rff_sigma=trunk_rff_sigma,
    )
    net = net.to(runtime_device)
    model = dde.Model(
        data,
        net,
    )
    model.compile(
        "adam",
        lr=1e-3,
        loss="MSE",
        loss_weights=[data_loss_weight, physics_loss_weight],
        metrics=["mean l2 relative error"],
    )
    model.restore(str(args.checkpoint), verbose=1)

    summary = build_full_evaluation_summary(
        model,
        data,
        dataset,
        dataset.val_x,
        dataset.val_y,
        dataset.test_x,
        dataset.test_y,
        rollout_steps,
    )
    summary["checkpoint"] = str(args.checkpoint)
    summary["manifest"] = str(manifest_path)

    print_section("Checkpoint Evaluation")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    output_path = args.output
    if output_path is None:
        output_dir = manifest_path.parent / "checkpoint_evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.checkpoint.stem}_evaluation.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, summary)


if __name__ == "__main__":
    main()
