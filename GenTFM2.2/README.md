# GenTFM v2.2：條件式表格生成

本版本保留 v2.2 的列注意力與行注意力，恢復 v1 的條件生成流程。
每次輸入一份乾淨、有標籤的 context，生成同一 schema 下的新資料行。
不提供無條件模式；舊無條件 v2.2 checkpoint 無法載入或續訓，需重新訓練。

## 架構

```text
乾淨 context [B,K,D]
  → context ColEmbedding → context RowInteraction → RowProject [B,K,E]
                                                        ↓ Key / Value
加噪 target [B,N,D]                                      ↓
  → target ColEmbedding → 加時間 embedding               ↓
  → target RowInteraction → RowProject [B,N,E] → CrossAttention
                                              Query      ↓
                                        velocity / category heads
```

- `ColEmbedding`：每個有效編碼欄位沿資料行做 induced attention。
- `RowInteraction`：每行內，不同編碼欄位做 attention，保留 feature embedding。
- context 和 target 使用獨立參數。context 不加噪聲、不加入時間 embedding。
- RowProject 將每行 `[D,E]` 展平並投影為 `[E]`，保留固定編碼欄位的位置。
- 每個交叉注意力 block 使用 target 作 Query，context 作 Key / Value。
- 輸出頭從條件化 row tokens 預測 velocity `[B,N,D]` 和 category logits
  `[B,N,max_cat,cat_cardinality]`，只輸出 target 行。
- context 行不使用位置 embedding；排列順序不影響生成速度，target 行排列
  對輸出保持 equivariance。欄位位置仍有意義，不能隨意重排編碼欄位。
- target 的列注意力會讓待生成行彼此交換資訊；整批生成與分批生成不等價。
  實驗中需固定每批生成行數。

編碼沿用固定寬度 `[continuous | categorical one-hots | observed bits]`。
預設 `Schema(32,8,12)`，D=160。所有有效資料必須完整觀測；觀測位固定為 1，
padding 固定為 0，不訓練 mask 頭。類別 softmax 排除無效類別。

```python
from model import GenTFM
from data.encoding import Schema

schema = Schema(32, 8, 12)
model = GenTFM(max_features=schema.encoded_dim,
               max_cont=32, max_cat=8, cat_cardinality=12)
outputs = model(x_t, t, feature_mask, context, return_aux=True)
# x_t: [B,N,D]；context: [B,K,D]；t: [B]；feature_mask: bool [B,D]
# context 和 x_t 必須使用同一個 context 標準化尺度。
```

## 訓練目標

每張 prior 表先打亂行順序，再切成 K 行 context 和其餘 target。
一次 batch 共用 K，各表獨立打亂；兩組資料行互不重疊。

連續欄位均值／標準差只從 context 計算，套用於兩條路徑。使用 population std，
非恆定欄位以 `std + 1e-6` 縮放；context 中 std ≤ 1e-6 的欄位使用 scale=1，
避免極小 context 或恆定欄位造成巨大 target 值。原始尺度生成使用相同規則。
此處與舊 v1 訓練時先按整張表正規化的做法不同。

```text
x0 ~ N(0,I)，t ~ Uniform(0,1)
x_t = (1-t) * x0 + t * clean_target
目標 velocity = clean_target - x0
loss = weighted_target_velocity_mse + categorical_loss_weight * target_ce
```

只有數值和 one-hot 座標走 flow path；observed bits 固定。
context 不計算重建 loss，但其編碼器透過 target loss 更新參數。

預設採用 v1 風格的 loss 權重：數值 velocity 權重 1、one-hot velocity 權重
`--discrete-flow-weight 0.05`，CE 權重 `--categorical-loss-weight 0.8`。
CE 排除無效類別，按有效 target 類別 cell 平均；這不是舊 v1 逐欄平均 CE 的逐字移植。
驗證按相應加權 velocity 座標數／類別 cell 數聚合，best.pt 依總驗證 loss 選取。

## 安裝與訓練

使用 Python 3.11；先安裝與 GPU/CUDA 相容的 PyTorch，再執行：

```sh
python -m pip install -r requirements.txt
python -c "import torch; from tabicl.prior import PriorDataset; print(torch.__version__, torch.cuda.is_available())"
```

TabICL 固定為 `2.1.1`，用於產生訓練 prior，不需要下載 TabICL predictor 權重。
以下命令從 `GenTFM2.2` 目錄執行。

### 100,000 張 prior 表，涵蓋 500 行 context

```sh
python -m script.train \
  --steps 12500 --batch-size 8 --num-rows 1024 \
  --min-context 5 --max-context 500 --min-target 512 \
  --context-sizes 20 50 100 200 500 \
  --num-cross-blocks 2 \
  --categorical-loss-weight 0.8 --discrete-flow-weight 0.05 \
  --device cuda:0 --log-every 100 --val-every 100 \
  --output-dir runs/v2.2_conditional_100k
```

100,000 是訓練表數，不含固定驗證表；batch size × steps 決定訓練表數。
`--num-rows` 是 context + target 的總行數；使用 500 行 context 時，1024 行表
剩下 524 行 target。參數必須滿足 `num_rows >= max_context + min_target`。
未指定 context-sizes 時，K 在 min-context 到 max-context 之間均勻抽樣。
prior 實際回傳形狀不符合請求會報錯，不會靜默縮短 context。

條件模型有兩條編碼路徑；上述 batch size 的 GPU 記憶體需求尚未實測。
可降低 batch size 並相應提高 steps，保持訓練表數。TabICL prior 仍在 CPU
產生，再傳至指定 GPU。

### 小型訓練

```sh
python -m script.train --steps 2 --batch-size 2 --num-rows 16 \
  --min-context 3 --max-context 8 --min-target 8 \
  --embed-dim 8 --num-col-blocks 1 --num-row-blocks 1 --num-cross-blocks 1 \
  --nhead 2 --dim-feedforward 16 --num-inds 3 \
  --max-cont 4 --max-cat 2 --cat-cardinality 3 \
  --prior-type mlp_scm --val-batches 1 --val-every 1 \
  --output-dir runs/conditional_smoke
```

輸出包括 `best.pt`、`latest.pt`、`loss_curve.png`。
`--help` 亦支援 `python script/train.py --help`。

## 條件生成

```python
import numpy as np
import torch
from training.checkpoint import load_pretrained
from inference import generate_in_context

model, checkpoint = load_pretrained("runs/v2.2_conditional_100k/best.pt", device="cuda:0")
context = np.load("context_encoded.npy")   # [500,160]，含 target，模型編碼中的原始尺度
# metadata 需與 context 一致：n_cont、n_cat、cat_cardinalities。
synthetic = generate_in_context(
    model, context, metadata, num_gen=500, n_steps=60, method="euler",
    generator=torch.Generator(device="cuda:0").manual_seed(42),
)
np.save("synthetic.npy", synthetic)
```

wrapper 使用 context 的均值與尺度，context encoder 每次生成只執行一次。
每一步 ODE 都用相同 context tokens。最後以分類頭抽樣合法類別，還原連續值尺度。
`categorical_method="argmax"` 改用最大機率類別；`"none"` 不使用分類頭抽樣，
raw sampling 保留 flow one-hot 座標，wrapper 最後仍會用 argmax 整理成合法資料。
目前沒有額外的類別頻率 calibration。

低階接口（資料已按 context 正規化）：

```python
from inference import sample_table
raw = sample_table(model, context_tensor, feature_mask, num_rows=500,
                   n_steps=60, method="heun")  # [B,500,D]
# 或 model.generate(context_tensor, feature_mask, num_gen=500)
```

context 必須非空、完整觀測，類別是合法 one-hot。
採樣暫時切換 eval 模式，結束或出錯後恢復各模組原先模式。

## 中斷續訓與 checkpoint

```sh
python -m script.train --resume runs/v2.2_conditional_100k/latest.pt --device cuda:0
```

- 新模型 config 記錄 `architecture="conditional_v1"`；舊無條件權重明確拒絕載入。
- 完整 checkpoint 保存 model、optimizer、scheduler、訓練配置、固定驗證
  context/target/noise/t、RNG、prior cache 請求順序和 loss 歷史。
- 續訓沿用 context 範圍、loss 權重、架構與原定總步數；不允許改變原訓練計畫。
- 可修改 device、output-dir、log-every、val-every、save-every。
- slim checkpoint 可供生成，不能續訓。新架構 checkpoint 缺少 training_state
  時可恢復權重／optimizer／scheduler，但重建驗證並重設 best loss。
- 寫入使用暫存檔原子替換。Ctrl+C 不保存可能只完成一部分的 optimizer 更新；
  使用最後完成的 latest.pt 續訓。
- CPU 的本機 prior 測試涵蓋中斷續訓與連續訓練逐位一致。跨硬體或非確定性
  GPU 運算不保證逐位一致。

## 驗證

```sh
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

測試覆蓋 context 改變輸出、context 排列不變性、target 排列 equivariance、
兩條注意力路徑與 cross-attention 梯度、context-only 縮放、padding、類別限制、
Euler/Heun、context cache、原始尺度還原、500→500 形狀與 checkpoint 續訓。
測試不下載 TabICL 模型；通過不等於已驗證真實資料的合成品質，品質需在重新訓練後評估。

`data/real_data.py` 的通用 encode_dataframe 仍可能將連續 target 分箱。
回歸 R² 實驗必須使用把 target 放入連續座標、只在 context 擬合的專用編碼器；
不要直接使用 target 分箱作為回歸標籤。

本次本機驗證：條件模型單元測試及預設寬度 160 的 500→500 採樣檢查。
TabICL prior 的實際小型訓練因本機 Python 缺少 xgboost 而未開始；
需安裝 requirements.txt 中的依賴後，在訓練環境執行上述小型訓練命令。
