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

模型由 ColEmbedding、時間 embedding、RowInteraction、逐 cell 速度頭和類別分類頭組成。
訓練入口會配置混合 schema，分類頭以各類別欄的有效 one-hot 座標表示平均池化，
再用共用 MLP 輸出該欄的類別 logits。沒有缺失 mask 頭或 feature_mask 預測頭。

```python
from model import GenTFM
from data.encoding import Schema
```

模型接口：

```python
schema = Schema(max_cont=32, max_cat=8, cat_cardinality=12)
model = GenTFM(max_features=schema.encoded_dim, max_cont=schema.max_cont,
               max_cat=schema.max_cat, cat_cardinality=schema.cat_cardinality)
velocity = model(x_t, t, feature_mask)
outputs = model(x_t, t, feature_mask, return_aux=True)
# outputs["velocity"]: (B, K, D)
# outputs["categorical_logits"]: (B, K, max_cat, cat_cardinality)
# x_t / velocity: (B, K, D)
# t: (B,)，每張表一個時間，範圍 [0, 1]
# feature_mask: (B, D)，bool，True 表示有效編碼維度
```

這裡的 D 是 encoding.py 的編碼維度：連續值、類別 one-hot 和觀測標記。
類別欄不是單一維度；數值與有效 one-hot 座標的速度 MSE 權重相同。
有效觀測標記固定為 1，速度為零，不參與 MSE。feature_mask 仍由外部提供，
表示完整編碼中的有效維度；它也決定各類別欄的有效類別。

## Loss

取標準高斯噪聲 x0、prior 樣本 x1 和 t ~ Uniform(0, 1)：

```text
xt = (1-t) * x0 + t * x1       # 僅數值與類別 one-hot 座標
目標速度 = x1 - x0
velocity_mse = 有效數值/one-hot 座標的速度平方誤差平均
categorical_ce = 有效原始類別 cell 的交叉熵平均，標籤取自乾淨 x1
loss = velocity_mse + categorical_loss_weight * categorical_ce
```

先清除 padding，再計算誤差；CE 的 softmax 排除該欄不存在的類別，
完全 padding 的欄位與表格不參與 CE。全 padding batch 的 loss 是可反向傳播的零。
訓練資料必須完整觀測，遇到有效觀測標記不是 1 會報錯。編碼寬度與 prior 行為保留。

`--categorical-loss-weight` 預設為 `1.0`，是可調起點，尚未透過實驗確定最佳權重。
設為 `0` 可做速度 MSE 消融，此時分類頭未受訓練，採樣需使用
`categorical_method="none"` 再以 one-hot 座標 argmax 解碼。
驗證 MSE 與 CE 各自按有效座標數／類別 cell 數聚合，再相加；best.pt 依總驗證 loss 選取。

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
  --categorical-loss-weight 1.0 \
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
可用 `--output-dir` 改變位置；支援以 `--resume` 接續完整訓練 checkpoint。
訓練正常結束後，會在相同目錄保存 `loss_curve.png`，同時顯示 train loss 和
validation loss。Train loss 記錄每一個訓練 step，validation loss 標在實際驗證的
step；橫軸是 training step，縱軸是 MSE 加權 CE 的總 loss。
訓練與驗證日誌分別顯示總 loss、velocity_mse 和 categorical_ce。
曲線只在訓練結束時繪製，不需要新增命令列參數。

也可以從專案根目錄直接執行 `python script/train.py --help`。

## 中斷後續訓

從 `GenTFM2.2` 目錄執行，指定實際的 latest.pt 路徑。例如輸出放在版本目錄內：

```sh
python -m script.train --resume runs/v2.2_100k_tables/latest.pt --device cuda:0
```

若輸出放在 GenTFM_series 根目錄的 runs，則改用
`--resume ../runs/v2.2_100k_tables/latest.pt`。

- 自動沿用原本的模型、schema、batch size、每表行數、loss 權重與原定總步數。
- 恢復模型權重、AdamW 狀態、cosine scheduler、最佳驗證 loss、固定驗證批次、
  Python／NumPy／torch／已初始化 CUDA 的隨機狀態，以及 prior 的生成計數和 cache 建構順序。
- 從保存的 step + 1 接續；`--steps` 是原定總步數，包含已完成步數。
  為保留原 cosine 計畫，不允許在 resume 時變更總步數或其他訓練設定。
  這個入口用於接續中斷的訓練，不用於延長已完成的訓練。
- 可改 `--device`、`--output-dir`、`--log-every`、`--val-every` 和 `--save-every`。
  未指定 output-dir 時會繼續寫入 checkpoint 所在目錄，方便搬移到 GPU 伺服器後續訓。
  指定新目錄時會攜帶原本可取得的 best.pt；找不到對應最佳權重時，重新比較最佳 loss。
- `--save-every 100` 是預設保存間隔。第 1 步、每次驗證與最後一步也會保存 latest.pt，
  訓練開始前保存 step=0。best.pt 只在驗證總 loss 改善時更新。
- checkpoint 先寫入同目錄的暫存檔，完成後才替換正式檔案；保存失敗會保留上一份。
- Ctrl+C 時不保存可能只完成一部分的 optimizer 更新；可從最後成功寫入的 latest.pt 恢復。
  突然斷電／強制終止也同樣處理，尚未保存的步數會重新執行。
- 新 checkpoint 保留 loss 歷史，最終 loss_curve.png 會接續中斷前後的記錄。
  舊版完整 checkpoint 可恢復權重、optimizer、scheduler 與步數，但缺少隨機狀態、
  原驗證資料和歷史，會重新生成固定驗證批次並重設最佳 loss，打印說明。
  只有權重的 slim checkpoint 不能用於 resume。

CPU 的隨機 prior 測試確認中斷／續訓與連續訓練的權重、optimizer、scheduler、
驗證資料和 loss 歷史一致；正式 TabICL／GPU 續訓尚未實測，跨硬體或非確定性
GPU 運算不保證逐位一致。

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

sample_table 用速度頭完成 ODE 後，在 t=1 呼叫分類頭；預設從有效類別機率
抽樣並輸出精確 one-hot，觀測標記固定為 1，padding 保持為零。
可用 `categorical_method="argmax"` 改成最大機率類別，或 `"none"` 保留連續類別座標。
分類投影只在 ODE 結束後執行；sanitize_mixed_encoded 可繼續用來整理編碼。
使用 generator 可重現噪聲及類別抽樣；輸出不代表已驗證的生成品質改善。

## 目前範圍

這一版按 feature_mask 生成整張表，沒有 context conditioning。
inference/generation.py 的 generate_in_context / generate_zero_context 依賴尚未實作的
條件生成接口，不能用來呼叫此版本；目前請使用 sample_table。
原本 v2 的速度模型 checkpoint 未存混合 schema 時，仍以速度模型載入；
不能直接視為已訓練的分類模型。新增 schema 的 checkpoint 會記錄分類頭設定。
其他舊架構 checkpoint 不保證相容。

model/ColEmb.py 和 model/RowInteract.py 的既有行為保留。
data/prior.py 的資料生成邏輯保留。
