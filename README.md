# pi-o-net — PiT LDC

Physics-informed Transformer (PiT) 用於多 Re 穩態 Lid-Driven Cavity (LDC) 流場預測。

## 架構：PiT CrossAttentionOperator

以 Cross-Attention 取代 DeepONet 的 branch-trunk 點積，讓每個查詢點只關注相關感測器。

```
Sensors [N_s × 5]          Query points [N_q × 2]
(x, y, u, v, p)            (x, y) + component c ∈ {0,1,2}
       │                            │
  SensorEncoder               QueryEncoder
  ─────────────               ────────────
  RFF(x,y) → concat          RFF(x,y) shared B
  Linear → [N_s, d_model]    Embedding(c) → concat
  prepend re_token            Linear → Q [N_q, d_model]
  TransformerEncoder × 2
  → K, V [N_s+1, d_model]
       │                            │
       └──────── CrossAttention ────┘
                 MHA(Q, K, V)
                 LayerNorm
                 Linear(d_model → 1)
                 ComponentScaler
                      │
                 output [N_q, 1]
```

### 關鍵設計選擇

| 元件 | 說明 |
|------|------|
| **RFF** | Random Fourier Features 編碼空間座標，提供豐富頻率基底 |
| **re_token** | 可學習 Re 標記，prepend 至感測器序列，讓 attention 獲得全局 Re 資訊 |
| **ComponentScaler** | 獨立縮放 u/v/p 輸出，解決各分量量級差異 |
| **Shared B** | SensorEncoder 與 QueryEncoder 共享 RFF 隨機矩陣，確保座標空間一致 |

### 超參數（Run 4）

```toml
d_model = 128
nhead = 4
num_encoder_layers = 2
dim_feedforward = 256
rff_features = 64
rff_sigma = 5.0
num_interior_sensors = 80
num_boundary_sensors = 20
num_query_points = 2048
num_physics_points = 1024
```

## 訓練設定

```toml
optimizer = "adamw"
learning_rate = 0.001
weight_decay = 0.0001
lr_schedule = "cosine"        # CosineAnnealingLR
min_learning_rate = 1e-5
iterations = 10000
batch_size = 3                # Re cases per step
max_grad_norm = 1.0
```

**Loss 函數：**

```
L_total = L_data + 0.1 × (L_NS_x + L_NS_y) + L_cont + L_BC + L_gauge
```

**資料：** LDC steady-state，Re ∈ {3000, 4000, 5000}，mat 格式 256×256 均勻網格。

## 結果

| Run | LR Schedule | Best val rel_L2 | Step |
|-----|-------------|-----------------|------|
| Run 3 | StepLR (×0.9 / 1k) | 0.0845 | 10k |
| **Run 4** | **Cosine 1e-3 → 1e-5** | **0.0757** | **10k** |

Run 4 相較 Run 3 改善 **10.4%**，cosine annealing 消除了 StepLR 在 step 7k 的 loss bump。

### Run 4 Checkpoint 進程

| Step | val rel_L2 | 備註 |
|------|-----------|------|
| 1,000 | 0.9496 | 初期訓練 |
| 2,000 | 0.6897 | — |
| 3,000 | 0.3294 | 急速下降 |
| 4,000 | 0.2029 | — |
| 5,000 | 0.1163 | — |
| 6,000 | 0.1018 | — |
| 7,000 | 0.0900 | 平滑（無 bump） |
| 8,000 | 0.0813 | — |
| 9,000 | 0.0772 | — |
| 10,000 | **0.0757** | **最佳** |

## 安裝

```bash
uv sync --python 3.11
uv run pytest
```

## 訓練

```bash
uv run pit-ldc-train --config configs/pit_ldc_run4_cosine.toml
```

可用 configs：

| 檔案 | 說明 |
|------|------|
| `pit_ldc.toml` | 基礎設定 |
| `pit_ldc_run3.toml` | Run 3（StepLR） |
| `pit_ldc_run4_cosine.toml` | Run 4（Cosine，目前最佳） |
| `transformer_re1000.toml` | Re=1000 實驗 |

## 專案結構

```
src/pi_onet/
  pit_ldc.py        # PiT 模型、訓練迴圈、physics loss
  ldc_dataset.py    # LDC .mat 載入、感測器採樣
configs/            # TOML 訓練設定
tests/
  test_pit_ldc.py   # 單元測試（13 個）
docs/
  pit_architecture.html  # 互動式架構說明文件
```

`data/` 與 `artifacts/` 不納入版本控制（`.gitignore`）。
