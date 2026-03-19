# pi-o-net

文獻對齊版 PI-DeepONet（對齊 arXiv:2103.10974）訓練專案。  
目前流程固定為：

- Branch：`DNS sensors at t=0 + Re`
- Trunk：`(x, y, t)` 原始可微座標
- Loss：`L_total = λ_ic * L_IC + λ_phys * L_physics`
- Physics：Kolmogorov vorticity equation residual（spectral 空間導數 + AD 時間導數）
- 記憶體修復：支援 `physics_branch_batch_size`，只對 `L_physics` 做 branch micro-batch

不保留舊版流程（Fourier trunk、GradNorm、自回歸單步回歸、Adam→L-BFGS 雙階段）。

## 安裝與測試

```bash
uv sync --python 3.11
uv run pytest
```

## 3090 伺服器部署

此 repo 不包含 `data/` 與 `artifacts/`（已加入 `.gitignore`），請在伺服器自行放置 DNS 資料到 `data/kolmogorov_dns/`。

```bash
chmod +x scripts/setup_3090.sh
./scripts/setup_3090.sh
```

正式訓練：

```bash
uv run train-kolmogorov-deeponet --config configs/paper_aligned_step.toml
```

## 訓練

預設從 `data/kolmogorov_dns/*.npy` 讀資料；也可用 `--data-file` 指定。

```bash
uv run train-kolmogorov-deeponet --config configs/paper_aligned_constant.toml
uv run train-kolmogorov-deeponet --config configs/paper_aligned_step.toml
uv run train-kolmogorov-deeponet --config configs/paper_aligned_cosine.toml
uv run train-kolmogorov-deeponet --config configs/local_fast.toml
```

三份 config 僅差在學習率調度：

- `paper_aligned_constant.toml`: `lr_schedule = "none"`
- `paper_aligned_step.toml`: `lr_schedule = "step"`
- `paper_aligned_cosine.toml`: `lr_schedule = "cosine"`

## Checkpoint 重評估

```bash
uv run eval-kolmogorov-checkpoint \
  --checkpoint artifacts/paper-aligned-step/checkpoints/kolmogorov_deeponet_step_5000-5000.pt
```

預設會自動讀取同一實驗目錄下的 `experiment_manifest.json`。

## 重要輸出

- `artifacts/*/training_summary.json`
- `artifacts/*/evaluation_summary.json`
- `artifacts/*/best_checkpoint_evaluation.json`
- `artifacts/*/mid_eval_history.json`
- `artifacts/*/best_checkpoint.pt`

上述 `evaluation_summary.json` / `best_checkpoint_evaluation.json` / `mid_eval_history.json` 皆包含 `pass_fail` 欄位，規則為：
- `validation_mean_relative_l2 < 0.2`
- `test_mean_relative_l2 < 0.2`（僅 full evaluation）
- `rollout.sensor_relative_l2_mean < 0.3`
- `rollout.physics_relative_l2_mean < 0.4`

## 目前模型結構（預設）

- Branch MLP: `[num_sensors + 1, 512, 512, 256]`
- Trunk MLP: `[3, 512, 512, 256]`
- Activation: `tanh`
- Gating: 啟用（modified gated MLP）
- Optimizer: `AdamW`
- Loss function: `MSE`（`L_IC` 與 `L_physics` 都是 MSE）
- Physics micro-batch: `physics_branch_batch_size = 4`（預設 config）
