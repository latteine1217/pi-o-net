"""Evaluate a saved literature-aligned PI-DeepONet checkpoint.

What:
    針對文獻對齊版 PI-DeepONet（L_IC + L_physics）重跑 validation/test/rollout 評估。
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
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--physics-time-samples", type=int, default=None)
    parser.add_argument("--physics-branch-batch-size", type=int, default=None)
    parser.add_argument("--branch-hidden-dims", type=str, default=None)
    parser.add_argument("--trunk-hidden-dims", type=str, default=None)
    parser.add_argument("--latent-width", type=int, default=None)
    parser.add_argument("--use-gated-mlp", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
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
    manifest_path = infer_manifest_path(args.checkpoint, args.manifest)
    defaults = merge_with_defaults(load_manifest_defaults(manifest_path))

    data_files = resolve_data_files(args.data_file if args.data_file is not None else defaults.get("data_files"))
    config = DatasetConfig(
        field=get_config_value(args, defaults, "field") or "omega",
        num_sensors=int(get_config_value(args, defaults, "num_sensors")),
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
    ic_loss_weight = float(get_config_value(args, defaults, "ic_loss_weight"))
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

    print_section("Checkpoint Evaluation Configuration")
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "manifest": str(manifest_path),
                "data_files": [str(path) for path in data_files],
                "num_sensors": config.num_sensors,
                "physics_stride": config.physics_stride,
                "rollout_steps": rollout_steps,
                "ic_loss_weight": ic_loss_weight,
                "physics_loss_weight": physics_loss_weight,
                "physics_time_samples": physics_time_samples,
                "physics_branch_batch_size": physics_branch_batch_size,
                "branch_hidden_dims": branch_hidden_dims,
                "trunk_hidden_dims": trunk_hidden_dims,
                "latent_width": latent_width,
                "use_gated_mlp": use_gated_mlp,
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
    )
    model = dde.Model(
        data,
        create_model(
            branch_dim=dataset.train_x[0].shape[1],
            trunk_dim=dataset.train_x[1].shape[1],
            branch_hidden_dims=branch_hidden_dims,
            trunk_hidden_dims=trunk_hidden_dims,
            latent_width=latent_width,
            use_gated_mlp=use_gated_mlp,
        ),
    )
    model.compile(
        "adam",
        lr=1e-3,
        loss="MSE",
        loss_weights=[ic_loss_weight, physics_loss_weight],
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
