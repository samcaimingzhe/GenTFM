# GenTFMv1.1 → v1.2：AI 修改指南

日期：2026-10-04  
基準程式：`/Users/caimingzhe/Desktop/new/GenTFMv1.1`  
後續實作目標目錄：`/Users/caimingzhe/Desktop/new/GenTFMv1.2`  
閱讀背景：[論文筆記與版本規劃](./GenTFM_論文閱讀筆記與版本規劃_2026-10-04.md)。

> 這份文件交給後續 coding AI 使用。本次只產生指南，沒有建立 v1.2 程式目錄、修改模型或啟動訓練。以下新增參數、函數與命令均為**待實作規格**，不能假定目前已存在。根據使用者最新要求，既有實驗資料的複用與適配不在本指南範圍內。

## 1. 實作目標與範圍

### 最終目標

給定帶標籤 context，生成新的 `(x, y)` 行，作為 TabICL 的 context，改善真實測試資料的 label 預測。以固定下游效用作為主要結果。

### 兩項主改動

1. **A：分類項從 one-hot 改為二進位類別碼。** 數值欄保持連續值；分類 y 的 embedding 仍使用整數 ID。
2. **B：明確控制 clean-context／noisy-query 拆分，並提供 noisy-query memory 候選模式。** 首輪配置為 K=200、Q=200，模式可切換，以便單獨驗證。

### 「模型框架不改」的具體實作界線

保留現有 `ContextEncoder → MixedFlowNet → velocity／cat／mask heads`、行級 hidden representation、CFM 路徑與 ODE 求解方式。二進位模式只改編碼引起的輸入／輸出尺寸；query-memory 使用現有 `MultiheadAttention` 改變 K/V 內容，不增加圖網路、欄位 Transformer 或 VAE。

query-memory 會改變 attention 的可見範圍和生成行之間的依賴，應稱為**既有框架內的 attention 行為變體**，不能說數學上的條件依賴完全沒變。提供 `context_only` 保留現有行為；若需完全維持現有 attention 語義，採用 A + 明確 K/Q 拆分即可，B 的互動模式只作候選實驗。

VGM／MSN、數值 float32 bits、TabSyn encoder、對比損失、y 採樣新策略放在後續版本研究。

## 2. 先理解 v1.1 的現有流程

主要實作位於 [model.py](/Users/caimingzhe/Desktop/new/GenTFMv1.1/gen_tfm/model.py)。

```text
整張 synthetic training table
    → 目前的數值正規化
    → 隨機打亂行
    → clean X_ctx 與 clean X_target/query

clean X_ctx → ContextEncoder → context memory
clean query → 提取 y → 移除待生成表示中的 y 座標
noise x0、clean target x1、時間 t → x_t
(x_t, t, y, context memory) → 預測 velocity、category logits、mask logits
```

目前 `compute_loss()` 已有：

```python
x_t = (1 - t) * x0 + t * x1
target_v = x1 - x0
```

`TabularFlowBlock.forward()` 現在以 query hidden 作 Q，clean context hidden 作 K/V。context 不加噪；query 已加噪；query 之間不互相注意。

所以單純改成「200 context + 200 query，query 加噪」是採樣配置修改，不是新增條件生成原理。DiffICL 原文同樣使用 clean context／noisy query，且明確禁止 query 間 attention。[DiffICL 第 3.1 節](https://arxiv.org/html/2605.04911v1)。

## 3. A：二進位 codec 的資料契約

### 3.1 固定 schema

保留每欄最大類別數 `C_max=12`。增加：

```text
cat_encoding: "onehot" | "binary"
binary_bit_order: "msb_first"
schema_version: "mixed_onehot_v1" | "mixed_binary_v1"
```

每個分類槽寬度：

```text
onehot：cat_width = C_max
binary：cat_width = ceil(log2(C_max))
```

固定行布局為：

```text
[ max_cont 連續值 | max_cat × cat_width 分類表示 | max_cont observed bits ]
```

預設 `max_cont=32、max_cat=8、C_max=12`：

| 模式 | 分類槽寬 | 行維度 D |
|---|---:|---:|
| onehot | 12 | 32 + 8×12 + 32 = 160 |
| binary | 4 | 32 + 8×4 + 32 = 96 |

**分類 logits 的類別數仍為 12。** 表示寬度 `cat_width` 和語義類別上限 `cat_cardinality` 必須分開使用；不能因為輸入有 4 bits 就把分類 head 改成 4 類。

### 3.2 類別 ID ↔ bits

- 保存／沿用現有 vocabulary 與類別 ID 對應，不重新排序類別。
- 高位在前；例如 `0→0000、3→0011、11→1011`。
- 每欄有效類別範圍為 `[0, C_j)`；metadata 明確保存 `C_j`。
- 固定 4 位槽內，較小 cardinality 右對齊，未使用的高位為 0。
- feature mask 由 schema + metadata 產生。建議僅開啟該欄需要的低位，其餘高位作固定 padding；有效欄寬用 `max(1, ceil(log2(C_j)))` 計算。
- 類別 0 的全零 bits 是有效資料，不能判成欄位不存在。padding 欄位由 `n_cat` 和 mask 區分。
- 超出目前最大類別容量的處理延續既有 vocabulary／other-bucket 協議；不要順便改動類別選取規則。

建議建立共用的 NumPy 與 Torch codec，避免 prior、model、資料載入端各自實作不同 bit order。新增函數可命名：

```text
category_ids_to_bits(ids, width)
category_bits_to_ids(bits, width)
category_block_slice(schema, field_index)
validate_category_ids(ids, cardinality)
```

`category_bits_to_ids()` 的嚴格模式只處理合法離散碼；生成途中 `x_t` 是浮點數，不能假定它已經是合法 bits。

### 3.3 分類生成的推薦做法

保留現有每欄 categorical head 和 cross-entropy：

```text
hidden → C_max 個 logits
       → 按 C_j 屏蔽無效類別
       → 抽樣整數類別 ID
       → 將 ID 編碼成 binary block
```

不將最後一步改成逐 bit 獨立閾值。否則例如 12 類的 `1100` 至 `1111` 都是無效碼，且逐 bit 抽樣會改變模型的輸出假設。

`sanitize_mixed_encoded()` 若必須處理任意浮點分類塊，可採對合法碼字的最近距離投影，平手規則固定；主生成路徑仍以 cat head 抽樣的 ID 為準。資料載入端使用嚴格驗證，發現原始資料不合法時應報錯，不使用取模或裁剪來掩蓋問題。

二進位 codec 借鑑 [Binary Diffusion 的表示方式](https://arxiv.org/html/2409.13882v1)，但仍使用 GenTFM 的連續 flow，不移植該文的 XOR corruption 或 binary diffusion loss。

### 3.4 y 的處理

| 場景 | 訓練／生成處理 |
|---|---|
| context 裡的分類 y | 與其他分類欄一起表示為 bits |
| query 分類 y 條件 | 從 clean 行解碼為整數 ID，交給既有 `y_cat_embed` |
| query 中的分類 y 座標 | 整個 y 分類槽從 flow mask 移除，不加入噪聲、不計入分類 CE |
| 最終分類行 | 把採樣／指定的 y 編碼回原 label 槽 |
| 數值 y | 維持既有 MLP 條件，移除數值座標及其 observed bit；生成後逆正規化並填回 |

`generate()` 的 y 只在每行開始時抽樣一次，整個 ODE 軌跡保持相同 y。分類按 context 類別頻率、數值按 context y bootstrap，沿用 v1.1 策略，避免增加第三個主變量。

## 4. B：context/query 拆分與 noisy-query memory

### 4.1 顯式 K/Q 拆分

為 `compute_loss()` 增加可選 query 數量或 query 數量配置，例如 `num_query`／`query_sizes`；保留舊模式 `Q=N−K` 供歷史配置重現。

第一組實驗：

```text
同一個 TabICL prior synthetic table
    → 隨機排列行
    → 前 K=200 行 clean context
    → 接下來 Q=200 行 query
    → 若 N>K+Q，剩餘行按明示策略不參與本步訓練
```

要求：

- K 和 Q 的行索引互不重疊；不能重複取 context 行充當 query。
- `N >= K+Q`；不滿足時給清楚的錯誤或按事先設定的可變 K/Q 政策處理。
- 修正目前 `min_target=512` 和 K=200、Q=200 的衝突；新模式不能讓 `min_target` 悄悄把 Q 擴大。
- 首輪可讓 prior 仍生成 768 行，只取明確的 400 行參與 loss。若改成生成 400 行，還需檢查 `prior_train_min_seq_len` 等約束，並在所有消融組使用相同設定。
- 記錄實際 K、Q、參與 loss 的行數、CFM／CE／mask loss 分量與訓練 token 預算。
- context 行保持乾淨；query 仍沿用原 CFM 路徑，沒有第二套加噪流程。
- 200 不寫死在模型中；推論接受可變 K、Q，小資料集沿用實際 context 行數。

### 4.2 兩個可切換模式

新增構造／checkpoint 配置：

```text
query_conditioning = "context_only" | "context_plus_noisy_query"
```

**`context_only`：**

```text
Q = query hidden
K/V = encoded clean context
```

這是 v1.1 的既有 attention 行為，也是明確 K/Q 拆分的對照組。

**`context_plus_noisy_query`：**

```text
Q = query hidden
K/V = concat(encoded clean context, current noisy-query hidden)
```

它實現使用者「query 也參與上下文」的候選想法，允許 query 行之間交換資訊。它不是 DiffICL 的 attention 復現。

### 4.3 在既有 block 裡實作

優先只改 `TabularFlowBlock.forward()` 的 memory 組裝，保留現有 attention、norm、MLP 和殘差模組。示意如下，實際參數命名可依程式風格調整：

```python
query = self.cross_norm(x)
if query_conditioning == "context_only":
    memory = context
elif query_conditioning == "context_plus_noisy_query":
    memory = torch.cat([context, query], dim=1)
else:
    raise ValueError(...)

cross_out, _ = self.cross_attn(
    query=query, key=memory, value=memory, need_weights=False
)
# 後續沿用既有殘差與 MLP。
```

每層使用該層**當前** noisy-query hidden；每次 ODE velocity 評估都重建 query memory。clean context encoding 可快取。Heun 的第二次 velocity 評估也必須使用對應中間狀態，不能沿用上一步 query memory。

不要把 clean `X_query` 再送入 `ContextEncoder` 當條件。模型 forward 可見的 query 特徵只能來自 `x_t`；clean query 的特徵用於構造訓練路徑和 loss，y 則是明確允許的生成條件。

### 4.4 生成階段的含義

```text
真實帶標籤 context → 固定 clean context memory
為 Q 個新行採樣 y → 初始化 Q 個 Gaussian noise 行
聯合更新這 Q 個 query flow 狀態 → 最後分類取樣／y 回填
```

生成 query 是準備產生的新行，不是真實測試集裡的行。不要額外引入 test 的 x 或 y；否則已改變原本只給 context 的生成任務。

### 4.5 需要單獨評估的影響

- query-memory 模式會破壞「給定 context，各生成行條件獨立」的原有性質；是否有益由實驗決定。
- 同批 query 數量 Q／生成 chunk 大小可能影響分佈，必須保存到實驗配置。
- context-only 模式可檢查分批一致性；query-memory 模式不能要求不同 chunk 切分必然得到相同結果。
- 保留行排列等變性：不新增依賴 query 行序的 positional embedding。
- attention 的 query 部分計算量由約 `O(QK)` 增為 `O(Q(K+Q))`，實際時間與記憶體要量測。
- 不宣稱 query 加噪本身是新的貢獻，不預設 query-memory 一定提高下游效用。

## 5. 逐檔案／函數修改清單

以下是基於目前 v1.1 實際函數的工作清單。先建立 v1.2 程式副本，再修改副本；不複製 checkpoint／大型 outputs 作為新版本訓練結果，不干擾正在訓練的 v1.1。

### 5.1 `gen_tfm/encoding.py`

| 現有類／函數 | 必要修改 |
|---|---|
| `Schema` | 加入 codec 配置、`cat_width`，按模式計算 `encoded_dim`／`mask_start` |
| `Schema.as_tuple()` | 不再依賴三參數 tuple 傳遞全部 schema；保留 legacy API 時必須明確傳 codec，避免 binary 默認成 onehot |
| `encoded_dim()`、`slices()` | 根據 schema／cat_width 計算，禁止散落的 `j * cat_cardinality` 位址計算 |
| `cat_cardinalities()` | 保留語義類別上限；校驗各欄真實 cardinality，與 bits 寬度分離 |
| `mixed_feature_mask()`、`batch_feature_mask()` | 依 metadata 建立 binary 位元／padding mask，不能從資料值猜欄位是否有效 |
| `encode_components()` | 整數 categories 編碼成 onehot 或 bits；若要完整支援 missingness round-trip，讓 observed bits 可明確傳入 |
| `decode_components()` | 用 codec 解回 ID，維持 `cont`／`obs`／`cats` 回傳契約 |
| `sanitize_mixed_encoded()` | 對合法碼字作明確投影，保留 observed bits 與 padding 處理 |
| 新增共用 codec | NumPy／Torch 使用相同 bit order、合法碼與欄位切片 |

### 5.2 `gen_tfm/prior.py`

| 現有函數 | 必要修改 |
|---|---|
| `TabICLPriorEngine.__init__()` | `max_features` 和切片由完整 schema 決定 |
| `_encode_one_dataset()` | 各分類 feature／label 先得到 ID，再用共用 codec；不要直接往 `start + ID` 寫 1 |
| `sample_batch()`／`sample_dataset()` | metadata 補 schema version、codec、bit order；輸出符合所選模式 |

保留分類 label 位於最後一個有效分類欄、回歸 target 位於第一數值槽的約定。回歸 prior 目前用 SCM 第一 feature 作 target 的代理設定應在 README 明記，不在 v1.2 順便換 prior。

### 5.3 `gen_tfm/model.py`

| 現有函數／模組 | 必要修改 |
|---|---|
| `infer_feature_mask()` | binary 模式不得用非零值推斷有效欄位；缺 metadata 時明確拒絕或要求顯式 mask |
| `metadata_from_feature_mask()` | 舊的 onehot mask 求和不能推出 binary cardinality；binary 模式要求真實 cardinality metadata |
| `normalize_mixed_batch()` | 只處理連續值；分類 bits 與 observed bits 不做數值標準化，切片依 schema |
| `GenTFM.__init__()`、`config()` | 傳遞並保存 codec、schema version、query_conditioning；調整 D，保留其餘框架配置 |
| `ContextEncoder`、`MixedFlowNet` | 使用 D=96 或 160 配置行投影／velocity head；cat head 仍輸出 `max_cat × C_max` logits |
| `MixedFlowNet.forward()` | 保留 `x_proj + t_emb + y_emb`；向既有 blocks 傳遞所選 conditioning 模式 |
| `TabularFlowBlock.forward()` | 支援 context-only 與 context-plus-noisy-query 的 K/V 組裝 |
| `_label_info()` | 統一 task/label metadata，修復 legacy regression 中 `label_cat_index=None` 被誤認分類的問題 |
| `_extract_y()` | 用 codec 讀取分類 ID；按新切片移除 label 座標；數值 y 規則保持 |
| `_type_weights()` | 按 schema 切片加權，避免把 bits 當數值欄；保留 loss 超參數並記錄各分量 |
| `_apply_categorical_context_calibration()` | 先把 context bits 解碼成 ID，再統計 histogram；不能把各 bit 的和當類別頻率 |
| `compute_loss()` | 明確 K/Q 拆分；分類 target ID 透過 codec 讀取，CE 前屏蔽無效類別；去除 label loss；沿用 CFM |
| `generate()` | 從 context 解碼取得 y 分佈；flow 全程固定 y；ODE 按所選模式重算 query memory；cat ID → bits；最後回填 y |

**Legacy metadata 正規化規則：** 優先讀明確 `label_type`，其次讀 `target_task`／`task_type`／`task`，最後根據非 `None` 的 label index 推斷。不可只因為字典含 `label_cat_index` 就判定是分類。檢查 index 是否落在有效欄位範圍；資訊矛盾時報錯。

分類 mask／loss 必須按每欄實際 `C_j` 處理。當 C_j 小於 C_max 時，無效類別 logits 在 CE／sampling／校準中都不可被當成真實類別。這項 metadata 一致性處理應套用所有對照模式，避免把工程修正誤算作二進位編碼收益。

### 5.4 `gen_tfm/generation.py`

| 現有函數 | 必要修改 |
|---|---|
| `context_stats()` | 延續只用 context 的連續統計，不碰 bits；常數欄沿用明確 epsilon 策略 |
| `generate_in_context()` | 從 model 完整配置建 schema；檢查輸入 codec、label 位置與維度；傳遞新模式 |
| `generate_zero_context()` | binary mask 仍須由 metadata 建立；無 context 時的 y 條件需有明確既有／指定政策，避免對全零類別頻率取樣 |
| `Calibration` | 所有消融組採用相同校準設定並明確記錄，不因入口不同而默默改變預設值 |

### 5.5 訓練入口與 checkpoint

| 檔案／函數 | 必要修改 |
|---|---|
| `scripts/train.py::parse_args()` | 增加 codec、query_conditioning 和顯式 Q 參數，檢查 K/Q 與 prior 行數約束 |
| `build_prior()`／`build_model()` | 使用同一完整 schema，不能一邊 binary、一邊 onehot |
| `train()` | 傳入 Q；metadata 和 mask 明確提供；記錄 schema、K/Q、loss 分量與實驗模式 |
| `synthetic_eval()` | 解碼／評估同時支援兩 codec，使用完整 schema |
| `checkpoint.py::save_training_checkpoint()` | 保存新增 model/schema 配置以及 train K/Q、種子等設定 |
| `export_slim_checkpoint()` | slim checkpoint 保留足夠還原模型的 codec 與 conditioning 配置 |
| `load_pretrained()` | 缺新增配置的 legacy checkpoint 按 onehot/context_only 讀取；拒絕不相容的強制配置 |

舊 onehot checkpoint 和新 binary checkpoint 的投影／velocity 尺寸不同。不得用 `strict=False` 假裝完成 resume，也不得直接復用尺寸不符的 optimizer state。正式消融優先從相同初始化政策重新訓練；若做部分 warm-start，需另列實驗並輸出載入／未載入參數清單。

### 5.6 其他呼叫端：避免只改主模型

| 檔案／函數 | 要審查的內容 |
|---|---|
| `real_data.py::build_encoded_table()`、`encode_dataframe()`、`decode_to_dataframe()` | 全部分類寫入／讀取走 codec；保留 category vocab 和 label 語義 |
| `baselines.py::mixed_*_baseline()` | 使用解碼 ID 與共用編碼，替換直接 onehot 寫入；不能在 binary 模式輸出 160 維 |
| `metrics.py::_active_encoded_view()`、`evaluate_mixed()` | 由完整 schema 得 mask；分類分佈用解碼 ID 計算 |
| `metrics.py::feature_matrix_without_label()`、`downstream_accuracy()` | 現有函數偏分類；回歸明確分支或沿用固定外部 TabICL 評估，不可 argmax 數值 target |
| `scripts/sample_prior.py`、`scripts/evaluate_real.py` | schema 構造、decode、label 著色、評估呼叫均傳新配置 |
| README／範例命令 | 標明新參數、兼容規則和回歸 prior 限制 |

建議全域搜尋 `cat_cardinality`、`as_tuple`、`argmax`、`label_cat_index`、`start +`、`reshape`，逐一確認分類槽寬與語義類別數沒有混用。

## 6. 消融與比較協議

| 組別 | 分類表示 | attention memory | K/Q |
|---|---|---|---|
| 對照 00 | onehot | context_only | 200/200 |
| A | binary | context_only | 200/200 |
| B | onehot | context_plus_noisy_query | 200/200 |
| A+B | binary | context_plus_noisy_query | 200/200 |

優先先完成 A 與其對照，再完成 B；沒有結果前不把 A+B 定為必然最好的版本。

固定 prior、數值處理、y 採樣、loss 超參數、優化器、batch、訓練步數／參與 loss 行數、種子政策，以及生成行數、ODE／校準、TabICL 配置。binary 會改變 discrete flow 維度與加權分母，應記錄 loss 分量；若再調權重，另列敏感度實驗，不混入主比較。

K=200、Q=200 是預訓練試驗配置。真實評估時，context 行數及合成行數按共同協議設定；模型接受可變大小，不為了湊 200 而改動資料切分。

分類與回歸分開匯總；採用既有協議的任務指標，跨資料集不直接混合未標準化誤差。現有 v1.1 分數可作歷史參考，主效果歸因用以上匹配配置的四組。

每個 run 至少記錄 checkpoint、codec、query 模式、K/Q、generation chunk、ODE solver／步數、校準、y 採樣政策、TabICL 設定、種子和評估格式。不同 run 的輸出分開保存。

## 7. 已有數值正規化差異：如何保持歸因清楚

現有 `compute_loss()` 在 context/query 拆分前使用整張 synthetic table 的統計；`generate_in_context()` 使用真實 context 的統計。這是目前既有差異。

主消融先沿用相同數值政策並記錄限制，不在 binary 或 query-memory 組單獨改正規化。若選擇把訓練統一成 context-only 統計，需在四組中同步修改，另列為共同工程變更；與原 v1.1 的比較不能只歸因 A/B。v1.3 的分佈轉換再另做實驗。

## 8. 建議實作順序

1. 閱讀目標目錄的 README／AGENTS 規則，建立 v1.2 程式副本，記錄基準來源。
2. 統一 schema／metadata 契約；先讓 legacy onehot 路徑能按明確配置工作。
3. 完成 NumPy／Torch binary codec、mask、合法碼與 round-trip。
4. 接通 prior → model → generate；分類 head／y embedding 延續既有語義。
5. 接通 checkpoint，明確拒絕跨 codec 的 resume。
6. 加入顯式 K/Q；接通 context-only 對照模式。
7. 在既有 flow blocks 中加入可切換 noisy-query memory，生成 ODE 同步支援。
8. 審查 utility／metrics 呼叫端，準備四組共用的評估入口與配置。
9. 按下列驗收項目核對，再安排四組訓練及評估；記錄實際執行的範圍與結果。

### 待實作後的命令形態

下列是新增介面示例，參數目前尚不存在；coding AI 應在實作完成後以實際 `--help` 更新：

```bash
python scripts/train.py \
  --output_dir outputs/v1_2_bin_qmem_cls \
  --target_task classification \
  --cat_encoding binary \
  --binary_bit_order msb_first \
  --query_conditioning context_plus_noisy_query \
  --rows_per_dataset 768 \
  --min_ctx 200 --max_ctx 200 \
  --num_query 200 --min_target 200
```

回歸使用獨立 run 配置 `--target_task regression`，沿用目前 prior 的代理 target 設定；不暗中把兩任務改為混合訓練。對照組只切換 codec／query_conditioning，其他配置一致。

## 9. 驗收條件

以下是後續實作應核對的項目；本次文件整理沒有執行模型測試或訓練。

### 編碼與 metadata

- [ ] 2–12 類所有有效 ID 的 `ID → bits → ID` 一致；單值欄／constant bits 邊界按明確策略處理。
- [ ] NumPy、Torch 和資料載入端的 bit order 相同；類別 0 不被當成 padding。
- [ ] `encode → decode` 保留連續值、observed bits、類別 ID 和欄位語義；padding 保持 0。
- [ ] D=160／96、cat_start、mask_start、feature mask 與 model 配置一致。
- [ ] 無效碼、矛盾 metadata、錯誤欄位索引與 codec 不匹配能給出明確錯誤。
- [ ] legacy regression 的 `label_cat_index=None` 能正確判任務。

### 訓練與 attention

- [ ] K/Q 不重疊、數量精確，query loss 分母使用實際 Q；兩任務 loss／gradient 有限。
- [ ] y 不在 query flow 座標中，分類 y 不計入 cat CE，回歸 y 的 observed bit 不計入 mask loss。
- [ ] binary 與 onehot 使用相同分類 ID target；每欄無效 logits 被屏蔽。
- [ ] `context_only` 保持原 attention 可見範圍；`context_plus_noisy_query` 的額外 K/V 只來自當前 x_t hidden。
- [ ] 固定 context、x_t、t、y 後，forward 不再讀 clean query 特徵；clean query 只用於路徑／loss。
- [ ] query 行置換後，對應輸出隨同置換；不同 K/Q 形狀可執行。

### 生成與 checkpoint

- [ ] y 在整條 ODE 軌跡固定，生成後正確回填；指定分類／數值 y 能保持其語義與尺度。
- [ ] 類別均位於 `[0, C_j)`，最終 binary blocks 是合法碼字。
- [ ] query-memory 在每層、每步及 Heun 子步重建；context encoding 快取不含 query 特徵。
- [ ] 存檔／載入能還原 codec、query 模式、張量尺寸；舊 checkpoint 默認 onehot/context_only。
- [ ] 不相容 codec 的 resume 被拒絕；不靜默丟棄權重或 optimizer state。

### 下游效用

- [ ] 分類與回歸各核對生成／預測／評分流程，合成行的 label 語義正確。
- [ ] 四組採用相同真實 context／test 切分、生成行數與 TabICL 配置。
- [ ] 每組以自身生成的 context 取得預測，不混用其他組的結果。
- [ ] 結果標明四組差異、訓練量、Q／chunk、成本與限制；不以降維圖代替效用分數。

## 10. 後續版本備忘

v1.3 研究 per-column 數值分佈診斷、可逆預處理、VGM／MSN、長尾與簡單分佈的回退政策。GT 思路參見 [CTAB-GAN+ 第 4.5 節](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2023.1296508/full)。轉換只在 context 擬合，保存參數與逆轉換；VGM mode indicators 需新的 schema 設計。

TabSyn 欄位 encoder、CTVAE 對比損失、平衡 y 採樣另立實驗。本文不提供未驗證的效用提升數字。

## 11. 可直接交給 coding AI 的任務摘要

> 請閱讀本文件與論文筆記，以 `/Users/caimingzhe/Desktop/new/GenTFMv1.1` 為基準建立獨立的 `GenTFMv1.2` 程式目錄。保留行級 context encoder、條件 flow matching、y embedding、head 與 ODE 框架。實作可切換 onehot／binary 分類 codec；binary 僅改分類輸入表示，分類 logits 仍是合法類別分佈。實作顯式 K/Q 拆分及 `context_only`／`context_plus_noisy_query` attention 模式，首輪用 K=Q=200。noisy query 只能來自當前 flow 狀態，不能把 clean target 特徵作額外 context。這個互動模式是我們自己的變體，不能聲稱直接復現 DiffICL。統一 schema、metadata、mask、codec 和 checkpoint，處理 legacy regression metadata，審查所有 onehot 假設的呼叫端。依本文件準備四組匹配配置消融與驗收項目；完整訓練配置、實際執行結果及待執行項目都需清楚標明。保護正在訓練的 v1.1；既有實驗資料的適配與複用不在本次任務範圍內。
