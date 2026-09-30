# GenTFM 第一版：無條件表格 flow matching

## 目錄

```text
GenTFM2.2/
├── model/          # 神經網路架構
│   ├── GenTFM.py
│   ├── ColEmb.py
│   └── RowInteract.py
├── data/           # 資料編碼、合成 prior、真實資料
│   ├── encoding.py
│   ├── prior.py
│   └── real_data.py
├── training/       # flow matching 目標、loss、checkpoint
│   ├── flow_matching.py
│   ├── checkpoint.py
│   └── plotting.py      # 訓練結束時保存 loss 曲線
├── inference/      # ODE 採樣與生成入口
│   ├── sampling.py
│   └── generation.py
├── evaluation/     # 基準方法和品質評估
│   ├── baselines.py
│   └── metrics.py
├── script/
│   └── train.py    # 訓練命令入口
├── tests/
│   └── test_flow_matching.py
├── requirements.txt
└── README.md
```

各 Python 套件另有 `__init__.py`。`model/` 只包含神經網路架構；
loss 定義在 `training/flow_matching.py`，Euler／Heun 採樣定義在 `inference/sampling.py`。

模型由 ColEmbedding、時間 embedding、RowInteraction 和逐 cell 速度頭組成。

```python
from model import GenTFM
```

模型接口：

```python
velocity = model(x_t, t, feature_mask)
# x_t / velocity: (B, K, D)
# t: (B,)，每張表一個時間，範圍 [0, 1]
# feature_mask: (B, D)，bool，True 表示有效編碼維度
```

這裡的 D 是 encoding.py 的編碼維度：連續值、類別 one-hot 和觀測標記。
類別欄不是單一維度；所有有效編碼維度的 loss 權重相同。

## Loss

取標準高斯噪聲 x0、prior 樣本 x1 和 t ~ Uniform(0, 1)：

```text
xt = (1-t) * x0 + t * x1
目標速度 = x1 - x0
loss = 有效 cell 上的速度平方誤差總和 / 有效 cell 數量
```

先清除 padding，再計算誤差。全 padding batch 的 loss 是可反向傳播的零。

## GPU 伺服器安裝

使用 Python 3.11 環境。在伺服器上先安裝與其 CUDA 環境相容的 PyTorch，
再從 `GenTFM2.2` 根目錄安裝專案依賴：

```sh
python -m pip install -r requirements.txt
python -c "import torch; from tabicl.prior import PriorDataset; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`requirements.txt` 不綁定特定 CUDA 版本；已安裝且符合要求的 PyTorch 會被保留。
TabICL 固定為 `2.1.1`，對應目前使用的 PriorDataset API；另列出 prior 所需的 xgboost。
這份檔案不是所有間接依賴的完整版本鎖定檔，也不會下載 TabICL 預訓練模型權重。
GPU 訓練前請確認上述檢查的 `CUDA available` 為 `True`。

你指定的 100,000 張訓練表格，每張 512 行、batch size 8，共 12,500 步：

```sh
python -m script.train \
  --steps 12500 \
  --batch-size 8 \
  --num-rows 512 \
  --log-every 100 \
  --val-every 100 \
  --device cuda:0 \
  --output-dir runs/v2.2_100k_tables
```

表格數量不含固定驗證表格。模型與訓練 tensor 放在 GPU；目前 TabICL SCM prior
仍以 CPU 生成表格，再傳到 GPU。訓練結束後會在指定目錄產生 best.pt、latest.pt
和 loss_curve.png。

## 執行

以下命令從 `GenTFM2.2` 目錄執行，使用已安裝 PyTorch 的 Python 環境。
訓練需要 numpy、torch、matplotlib、TabICL 及其 prior 的依賴（包含 xgboost）。

```sh
python -m unittest discover -s tests -v
python -m script.train --steps 1000 --batch-size 4 --num-rows 128
```

小型訓練：

```sh
python -m script.train --steps 2 --batch-size 2 --num-rows 16 \
  --embed-dim 8 --num-col-blocks 1 --num-row-blocks 1 \
  --nhead 2 --dim-feedforward 16 --num-inds 3 \
  --max-cont 4 --max-cat 2 --cat-cardinality 3 \
  --prior-type mlp_scm --val-batches 1 --val-every 1
```

訓練預設使用 CPU，可指定 `--device`。第一版固定每張表的行數，沒有 row padding。
驗證表格、噪聲和時間在啟動時固定，之後驗證使用相同批次。
checkpoint 存到工作目錄的 `checkpoints/best.pt` 和 `checkpoints/latest.pt`；
可用 `--output-dir` 改變位置。此入口尚未提供中斷後續訓。
訓練正常結束後，會在相同目錄保存 `loss_curve.png`，同時顯示 train loss 和
validation loss。Train loss 記錄每一個訓練 step，validation loss 標在實際驗證的
step；橫軸是 training step，縱軸是有效 cell 上的 velocity MSE。
曲線只在訓練結束時繪製，不需要新增命令列參數。

也可以從專案根目錄直接執行 `python script/train.py --help`。

## 生成

```python
import torch
from training.checkpoint import load_pretrained
from data.encoding import Schema, batch_feature_mask, sanitize_mixed_encoded
from inference.sampling import sample_table

model, checkpoint = load_pretrained("checkpoints/best.pt")
cfg = checkpoint["train_config"]
schema = Schema(cfg["max_cont"], cfg["max_cat"], cfg["cat_cardinality"])

# metadata 描述要生成的表格，例如有效連續欄數、類別欄數和各欄基數。
metadata = {"n_cont": 4, "n_cat": 2, "cat_cardinalities": [3, 3],
            "no_missingness": True}
mask = batch_feature_mask([metadata], *schema.as_tuple(), device="cpu")
raw = sample_table(model, mask, num_rows=128, n_steps=60, method="heun")
encoded = sanitize_mixed_encoded(raw[0].numpy(), metadata, *schema.as_tuple())
```

sample_table 回傳尚未投影的連續編碼結果；sanitize_mixed_encoded 將類別轉為
one-hot、觀測標記轉為 0/1、padding 清零。這一步不在 ODE 的中間步驟執行。

## 目前範圍

這一版按 feature_mask 生成整張表，沒有 context conditioning。
inference/generation.py 的 generate_in_context / generate_zero_context 依賴尚未實作的
條件生成接口，不能用來呼叫此版本；目前請使用 sample_table。
舊架構 checkpoint 不保證與這個新模型相容。

model/ColEmb.py 和 model/RowInteract.py 的既有行為保留。
data/prior.py 的資料生成邏輯保留。
