# Architecture Design: PiT Cross-Attention Operator for LDC

**Date**: 2026-03-22
**Scope**: `src/pi_onet/pit_ldc.py`, `configs/pit_ldc.toml`, `tests/test_pit_ldc.py`
**Status**: Pending Review

---

## Motivation

DeepONet 與 PIRATENet 同屬 branch-trunk 架構，研究價值高度重疊。本設計以 Cross-Attention Operator 取代 branch-trunk，使輸出在每個 query point 成為「對所有感測器的加權聚合」，而非獨立 branch/trunk 的內積。這在物理上的意義更直接：每個預測點從感測器學到的訊息量由 attention weight 顯式決定。

問題目標：LDC steady-state cavity flow，多 Re（3000、4000、5000），operator learning 設定（一次訓練，zero-shot 推論任意 Re）。

---

## New Files

| File | Responsibility |
|------|---------------|
| `src/pi_onet/pit_ldc.py` | 模型定義、訓練邏輯、CLI entry point |
| `configs/pit_ldc.toml` | 超參數設定 |
| `tests/test_pit_ldc.py` | 單元測試與煙霧測試 |

**Reused without modification:**
- `src/pi_onet/ldc_dataset.py` — `LDCDataset`、`load_ldc_mat`
- `src/pi_onet/ldc_train.py` — `steady_ns_residuals`、`compute_bc_loss`、`compute_gauge_loss`

**Not reused** (PiT 有不同 config keys，`load_ldc_config` 會拒絕 PiT 專用欄位):
- `src/pi_onet/ldc_train.py:load_ldc_config` — `pit_ldc.py` 自行實作 `load_pit_config`

---

## Architecture

### Overview

```
感測器讀值 [N_s, 5]              Query points [N_q, 3]
[x_s, y_s, u_s, v_s, p_s]       [x, y, c]
         │                              │
   SensorEncoder                  QueryEncoder
   (RFF + Linear +                (RFF + Embedding
    Self-Attn × L)                 + Linear)
         │                              │
   Keys, Values [N_s+1, d]       Queries [N_q, d]
              └──────────────────────┘
                    CrossAttention
                         │
                  [N_q, d_model]
                         │
                   LayerNorm
                         │
                   Linear → 1
                         │
                 ComponentScaler
                         │
                  field value [N_q, 1]
```

### Shared RFF Matrix B

`B: [2, rff_features]`，entries i.i.d. sampled as `torch.randn(2, rff_features) * rff_sigma`，registered as a **non-trainable buffer** on `CrossAttentionOperator`。

`SensorEncoder` 與 `QueryEncoder` 共用同一個 `B`（由 `CrossAttentionOperator` 傳入），確保感測器座標與 query 座標的空間頻率對齊。

RFF 編碼：`γ(z) = [sin(2π·z·B), cos(2π·z·B)]`，其中 `z: [N, 2]`，輸出 `[N, 2*rff_features]`。

---

### Sensor Tensor Assembly

`SensorEncoder` 接收 `sensors: [N_s, 5]`，其中 `N_s = num_interior_sensors + num_boundary_sensors = 100`（預設）。

感測器場值是靜態的（steady-state），因此在訓練迴圈**開始前**組裝一次並快取為 device tensor。同時預先計算正規化 Re 值：

```python
from pi_onet.ldc_dataset import RE_MEAN, RE_STD

# 訓練迴圈前，組裝並快取
idx = np.concatenate([dataset.sensor_interior, dataset.sensor_boundary])  # [N_s]
sensors_list = []
re_norm_list = []   # 正規化 Re 值（Python float），全文統一使用此名稱
for i, g in enumerate(dataset.grids):
    s = np.stack([g["x"][idx], g["y"][idx], g["u"][idx], g["v"][idx], g["p"][idx]], axis=1)
    sensors_list.append(torch.tensor(s, dtype=torch.float32, device=device))
    re_norm_list.append(float((dataset.re_values[i] - RE_MEAN) / RE_STD))
# sensors_list[i]: [N_s, 5] Tensor on device
# re_norm_list[i]: Python float（正規化 Re），傳入 net.forward 後在 forward 內部轉為 tensor

# 訓練迴圈內直接取用
model_fn_i = make_pit_model_fn(net, sensors_list[i], re_norm_list[i], device)
```

`make_pit_model_fn` 的第三個參數 `re_norm` 為 **Python float**，在 `CrossAttentionOperator.forward` 內部統一轉為 `Tensor[1,1]`。全文中不出現 `re_values`、`re_norm_t` 等其他名稱，一律用 `re_norm_list[i]`（訓練迴圈）或 `re_norm_list[i]`（validation loop）。

sensor index 安全性：`dataset.sensor_interior` 與 `dataset.sensor_boundary` 的最大 index 受限於 `grid_size = int(round(min_n_pts ** 0.5))`（所有 Re case 中最小格點的 grid_size），對所有 Re case 均有效。

---

### SensorEncoder

**輸入**：
- `sensors: [N_s, 5]` — `[x_s, y_s, u_s, v_s, p_s]`（原始量，不含 Re）
- `re_norm: Tensor[1, 1]` — `(Re - 4000) / 816.5`，reshape 為 `[1, 1]` 後傳入
- `B: [2, rff_features]` — 共用 RFF 矩陣

**步驟**：

1. `rff_s = γ(sensors[:, :2], B)` → `[N_s, 2*rff_features]`
2. `token_input = cat([rff_s, sensors[:, 2:]], dim=1)` → `[N_s, 2*rff_features + 3]`
3. `tokens = sensor_proj(token_input)` → `[N_s, d_model]`（Linear）
4. `re_token = re_proj(re_norm)` → `[1, d_model]`（Linear(1 → d_model)，輸入為 `[1, 1]`）
5. `tokens = cat([re_token, tokens], dim=0)` → `[N_s+1, d_model]`
6. Self-Attention × `num_encoder_layers`（`TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=attn_dropout)`）

**輸出**：`[N_s+1, d_model]`（作為 K, V）

**`re_norm` reshape 責任**：`re_norm` 以 Python float 形式傳入 `CrossAttentionOperator.forward`，由 `CrossAttentionOperator.forward` 在呼叫 `SensorEncoder` 之前執行 reshape：

```python
# CrossAttentionOperator.forward 內
re_norm_t = torch.tensor([[re_norm]], dtype=torch.float32, device=sensors.device)
kv = sensor_encoder(sensors, re_norm_t, B)
```

`SensorEncoder.forward` 的 `re_norm` 參數型別為 `Tensor[1, 1]`，不負責 reshape。

---

### QueryEncoder

**輸入**：
- `xy: [N_q, 2]`
- `c: [N_q]`（int long tensor，0=u, 1=v, 2=p）
- `B: [2, rff_features]` — 共用 RFF 矩陣

**步驟**：

1. `rff_q = γ(xy, B)` → `[N_q, 2*rff_features]`
2. `emb_c = component_emb(c)` → `[N_q, 8]`（`Embedding(3, 8)`，`init: normal(0, 0.1)`）
3. `q_input = cat([rff_q, emb_c], dim=1)` → `[N_q, 2*rff_features + 8]`
4. `queries = query_proj(q_input)` → `[N_q, d_model]`（Linear）

**輸出**：`[N_q, d_model]`（作為 Q）

---

### CrossAttentionOperator（主模型）

```python
class CrossAttentionOperator(nn.Module):
    # Buffers
    B: [2, rff_features]  # non-trainable

    # Sub-modules
    sensor_encoder: SensorEncoder
    query_encoder: QueryEncoder
    cross_attn: nn.MultiheadAttention(d_model, nhead, batch_first=True)
    norm: nn.LayerNorm(d_model)
    output_head: nn.Linear(d_model, 1, bias=True)
    component_scale: nn.Parameter(ones(3))   # 可學習量級，初始 identity
    component_bias:  nn.Parameter(zeros(3))  # 可學習偏移，初始 0
```

**forward(sensors, re_norm, xy, c) → [N_q, 1]**：

```python
kv = sensor_encoder(sensors, re_norm, B)   # [N_s+1, d]
q  = query_encoder(xy, c, B)               # [N_q, d]

# MultiheadAttention 需要 [batch, seq, d]
kv = kv.unsqueeze(0)   # [1, N_s+1, d]
q  = q.unsqueeze(0)    # [1, N_q, d]

attn_out, _ = cross_attn(q, kv, kv)       # [1, N_q, d]
attn_out = attn_out.squeeze(0)            # [N_q, d]

feat = norm(attn_out)
out  = output_head(feat)                  # [N_q, 1]
out  = out * component_scale[c].unsqueeze(1) + component_bias[c].unsqueeze(1)
```

**`create_pit_model(cfg: dict) -> CrossAttentionOperator`**：工廠函式，讀取 config dict 建立模型。

---

## Physics Loss

`steady_ns_residuals`、`compute_bc_loss`、`compute_gauge_loss` 直接 import：

```python
from pi_onet.ldc_train import steady_ns_residuals, compute_bc_loss, compute_gauge_loss
```

**`make_pit_model_fn(net, sensors_t, re_norm_t, device)`**：

```python
def model_fn(xy, c):
    # c 是 int（與 compute_bc_loss / compute_gauge_loss 的 keyword 呼叫介面一致）
    return net(sensors_t, re_norm_t, xy,
               torch.full((xy.shape[0],), c, dtype=torch.long, device=device))
```

**注意**：參數名稱必須是 `c`（不是 `c_idx`），因為 `compute_bc_loss` 與 `compute_gauge_loss` 以 keyword 呼叫 `model_fn(xy, c=0)`。

**u_fn / v_fn / p_fn 包裝**（與 ldc_train.py 相同模式）：

```python
model_fn_i = make_pit_model_fn(net, sensors_list[i], re_norm_list[i], device)
u_fn = lambda xy, fn=model_fn_i: fn(xy, c=0)
v_fn = lambda xy, fn=model_fn_i: fn(xy, c=1)
p_fn = lambda xy, fn=model_fn_i: fn(xy, c=2)
ns_x, ns_y, cont = steady_ns_residuals(u_fn, v_fn, p_fn, xy_phys, re=re_values[i])
```

**eval/train 模式切換**（僅在 `attn_dropout > 0` 時有實質效果，預設 `attn_dropout=0.0` 時為 no-op，仍建議保留以示意圖）：

```python
net.eval()
# 物理損失計算（autograd 仍正常，eval 只關閉 dropout 噪聲）
ns_x, ns_y, cont = steady_ns_residuals(...)
l_bc   = compute_bc_loss(...)
l_gauge = compute_gauge_loss(...)
net.train()
```

**Validation Loop**

每隔 `checkpoint_period` 步執行一次 validation，使用 `dataset.sample_val_trunk_all()` 取得完整 val pool。

`sample_val_trunk_all()` 回傳 `trunk_pts: [n_val*3, 3]`（column 2 為 float c ∈ {0.0, 1.0, 2.0}）與 `ref_vals: [num_re, n_val*3]`。PiT 使用整數 c，轉換如下：

```python
trunk_pts, ref_vals = dataset.sample_val_trunk_all()
xy_val = torch.tensor(trunk_pts[:, :2], dtype=torch.float32, device=device)
c_val  = torch.tensor(trunk_pts[:, 2], dtype=torch.long, device=device)  # float → long

with torch.no_grad():
    net.eval()
    total_rel_l2 = 0.0
    for i in range(num_re):
        # re_norm_list[i] 為 Python float，與訓練迴圈使用的同一個列表
        pred = net(sensors_list[i], re_norm_list[i], xy_val, c_val).squeeze(1)
        ref  = torch.tensor(ref_vals[i], dtype=torch.float32, device=device)
        rel_l2 = (torch.norm(pred - ref) / (torch.norm(ref) + 1e-8)).item()
        total_rel_l2 += rel_l2
    mean_rel_l2 = total_rel_l2 / num_re
    net.train()
```

Best checkpoint 以 `mean_rel_l2` 最小為準。

**Total Loss**：

```
L_total = data_loss_weight    × L_data
        + physics_loss_weight × (L_NS + physics_continuity_weight × L_continuity)
        + bc_loss_weight      × L_bc
        + gauge_loss_weight   × L_gauge
```

**Gradient Clipping**：Transformer + 二階物理梯度容易發生梯度 spike，訓練迴圈加入：

```python
torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_grad_norm)
```

`max_grad_norm` 預設 1.0，可由 config 設定。

---

## Config Loading (`load_pit_config`)

`pit_ldc.py` 實作自己的 `load_pit_config(path: Path) -> dict`，不使用 `ldc_train.py` 的 `load_ldc_config`（後者會拒絕 PiT 專用欄位）。

實作模式與 `load_ldc_config` 相同：

```python
DEFAULT_PIT_ARGS = { "d_model": 128, "nhead": 4, ... }

def load_pit_config(path: Path) -> dict:
    with open(path, "rb") as f:
        payload = tomllib.load(f)
    raw = payload.get("train", payload)  # 處理 [train] section header
    merged = {**DEFAULT_PIT_ARGS, **raw}
    unknown = set(merged) - set(DEFAULT_PIT_ARGS)
    if unknown:
        raise ValueError(f"PiT config 含有不支援的欄位：{unknown}")
    return merged
```

---

## Config (`configs/pit_ldc.toml`)

```toml
[train]
data_files = [
  "../data/ldc/cavity_Re3000_256_Uniform.mat",
  "../data/ldc/cavity_Re4000_256_Uniform.mat",
  "../data/ldc/cavity_Re5000_256_Uniform.mat",
]
num_interior_sensors = 80
num_boundary_sensors = 20
# num_query_points: 每步從 train pool 採樣的 query points 數（原 num_trunk_points）
num_query_points = 2048
num_physics_points = 1024
num_bc_points = 100

# PiT 架構參數
d_model = 128
nhead = 4
num_encoder_layers = 2
dim_feedforward = 256
attn_dropout = 0.0
rff_features = 64
rff_sigma = 5.0

# Loss weights
data_loss_weight = 1.0
physics_loss_weight = 0.1
physics_continuity_weight = 1.0
bc_loss_weight = 1.0
gauge_loss_weight = 1.0

# Training
iterations = 10000
batch_size = 3
optimizer = "adamw"
learning_rate = 0.001
weight_decay = 0.0001
lr_schedule = "step"
lr_step_size = 5000
lr_step_gamma = 0.5
min_learning_rate = 1e-6
max_grad_norm = 1.0
checkpoint_period = 2000
seed = 42
device = "auto"
artifacts_dir = "../artifacts/pit-ldc"
```

**注意**：`num_query_points`（非 `num_trunk_points`）——PiT 沒有 trunk network，此參數代表每步採樣的 query points 數量。

---

## CLI & Entry Point

```
python -m pi_onet.pit_ldc --config configs/pit_ldc.toml [--device cuda]
```

`pyproject.toml` 新增：

```toml
pi_onet_pit_ldc_train = "pi_onet.pit_ldc:main"
```

---

## Output Structure

```
artifacts/pit-ldc/
  experiment_manifest.json
  checkpoints/
    pit_ldc_step_<N>.pt
  best_validation/
    pit_ldc_best.pt
    best_validation_summary.json
  final_evaluation.json
```

---

## Testing (`tests/test_pit_ldc.py`)

| 測試 | 驗證內容 |
|------|---------|
| `test_rff_shape` | γ 輸出 shape = `[N, 2*rff_features]` |
| `test_sensor_encoder_output_shape` | SensorEncoder 輸出 `[N_s+1, d_model]` |
| `test_query_encoder_output_shape` | QueryEncoder 輸出 `[N_q, d_model]` |
| `test_b_not_trainable` | B buffer 不在 `parameters()` 中 |
| `test_component_scale_trainable` | component_scale / component_bias 在 `parameters()` 中 |
| `test_forward_shape` | forward 輸出 `[N_q, 1]` |
| `test_forward_backward` | loss.backward() 不 crash，梯度非 None |
| `test_create_pit_model` | 工廠函式建立正確 d_model |
| `test_component_scale_differentiates` | u/v/p 三個 channel 輸出值不同（scaling 有效） |
| `test_physics_loss_zero_for_linear` | 已知線性解的 NS 殘差接近零 |
| `test_bc_loss_zero` | 正確 BC 時 bc_loss ≈ 0 |
| `test_pit_model_fn_bc_gauge_compat` | `make_pit_model_fn` 的 closure 能正確傳入 `compute_bc_loss(model_fn, ...)` 與 `compute_gauge_loss(model_fn, ...)` |
| `test_smoke_train` | subprocess 跑 3 steps，checkpoint 存在，loss 非 NaN |

---

## What Is Not Changed

- `ldc_dataset.py`、`ldc_train.py`、`train.py`、`dataset.py` — 完全不動
- 現有 LDC DeepONet 訓練流程保持可運行
