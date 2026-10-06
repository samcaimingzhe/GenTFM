# GenTFMv1.2

以 GenTFMv1.1 為基準建立的獨立程式版本。保留行級 ContextEncoder、y embedding、條件 flow matching、分類／mask heads，以及 Euler／Heun ODE；新增分類二進位編碼與可切換 noisy-query memory。

## 主要改動

- `cat_encoding=binary`：分類 ID → 高位在前的 bits；預設 schema 從 160 維變為 96 維。數值仍是連續值。
- 分類 head 仍輸出每欄 12 類 logits，按該欄真實 cardinality 取樣，再編碼回合法 bits。分類 y 的 embedding 仍接收整數 ID。
- `num_query=200`：精確取 200 context + 200 query；未選中的 prior 行不計入 loss。設為 `0` 可回到 `Q=N−K`。
- `query_conditioning=context_plus_noisy_query`：在原 attention 的 K/V 中加入當層 noisy-query hidden；每個 query 只能注意 context 和自己的 noisy hidden，不能讀取其他 query。`context_only` 保留原 attention 行為。
- schema、metadata、校準、生成、baseline、metrics 和 checkpoint 均傳遞 codec 配置。

query 加噪在 v1.1 已存在。現在的 noisy-query memory 僅加入每行自己的 hidden，不會產生 query 間互動；先前未加 mask 的實驗與目前行為不同。

## 環境

Python 3.10+。依現有 `requirements.txt` 安裝依賴；GPU 伺服器先安裝對應 CUDA／ROCm 的 PyTorch。TabICL prior 使用固定的 `tabicl==2.1.1` 介面。

```bash
pip install -r requirements.txt
```

## 開始訓練

在本目錄下執行，四組 JSON 使用共同訓練配方。命令列參數會覆蓋 JSON 的設定：

```bash
python scripts/train.py --config configs/bin_qmem.json
```

預設：binary、noisy-query memory、K=Q=200、每張 prior table 最多 768 行、batch 8、100000 steps；JSON 配方關閉訓練後自動評估。實際 prior table 可以比 768 短，但必須有至少 K+Q 行。

| 配置 | 分類表示 | Query memory |
|---|---|---|
| `configs/oh_ctx.json` | onehot | context_only |
| `configs/bin_ctx.json` | binary | context_only |
| `configs/oh_qmem.json` | onehot | context + noisy query |
| `configs/bin_qmem.json` | binary | context + noisy query |

回歸使用獨立 run：

```bash
python scripts/train.py --config configs/bin_qmem.json \
  --target_task regression --output_dir outputs/v1_2_bin_qmem_reg
```

目前回歸 prior 沿用 v1.1：把 SCM 第一個來源 feature 作數值 y，是代理回歸任務。分類 y 在最後一個有效分類槽，數值 y 在第 0 數值槽；實際位置以 metadata 為準。

若要用舊訓練配方，需明確給 `--cat_encoding onehot --query_conditioning context_only --num_query 0 --min_ctx 5 --max_ctx 200 --min_target 512`。這不等於逐步完全復現歷史訓練：合法 cardinality 屏蔽與 observed-mask 處理已統一修正。

## 從 context 生成新行

輸入 `.npy` 必須是 checkpoint 對應的 codec；metadata 必須包含 `n_cont`、`n_cat`、`cat_cardinalities`、任務／label index，以及 codec 欄位。binary 的類別 0 是全零碼，不能靠非零資料猜有效欄位。

```bash
python scripts/generate.py \
  --checkpoint outputs/v1_2_bin_qmem_cls/checkpoints/latest.pt \
  --context /path/to/context_binary.npy \
  --metadata /path/to/context_metadata.json \
  --output /path/to/generated_binary.npy \
  --num_gen 200 --n_steps 60 --method heun
```

輸出為模型 codec 的 encoded rows，旁邊的 `.json` 記錄模型配置、生成設定和 metadata。程式拒絕覆寫既有輸出。可用 `--target_y` 指定固定 y：分類為類別 ID，回歸為輸入 context 的數值尺度；未指定則沿用 context 的 empirical y 採樣。

Python API：

```python
from gen_tfm import load_pretrained, generate_in_context
from gen_tfm.encoding import decode_components

model, checkpoint = load_pretrained("checkpoint.pt", device="cpu")
generated = generate_in_context(model, context_encoded, metadata,
                                num_gen=200, calibration=None)
components = decode_components(generated, metadata, *model.schema.as_tuple(),
                               **model.schema.codec_kwargs())
# components: cont、obs、cats（整數類別 ID）、encoded
```

編碼時使用 `schema=model.schema`，不要用預設的 legacy `Schema()` 來建立 binary 資料。`Schema.as_tuple()` 僅保留舊三個尺寸，所有 codec helper 呼叫還需傳 `**schema.codec_kwargs()`。

## Prior 與評估入口

```bash
python scripts/sample_prior.py --cat_encoding binary \
  --target_task classification --output_dir results/prior_binary
```

除 CSV／圖外，保存 `table_<i>_encoded.npy` 和對應 metadata，方便查看資料契約。

`train.py --eval_tabicl` 可在訓練後 synthetic evaluation 中加入 TabICL TSTR。`evaluate_real.py` 預設使用 TabICL；可明確選 `--downstream_predictor proxy` 使用 LogReg／Ridge 的輕量指標。分類報 accuracy，回歸報 R²。

所有 codec 的分佈距離與下游特徵先解碼成共同的 onehot 評估表示，以避免僅改 bit 距離就造成指標變化。這只影響評估表示，生成模型的輸入仍為所選 codec。既有真實資料 loader 仍保留原來整表預處理的限制；嚴格 context-only 擬合的真實資料評估需要另行安排，不能將該入口當作無洩漏的完整效用證據。

## Checkpoint 與限制

- 新 checkpoint 保存 codec、schema version 和 query-conditioning；train config 保存 K/Q、種子及其他訓練設定。舊的 `context_plus_noisy_query` checkpoint 是在 query 間可互看的注意力下訓練的；雖然參數形狀相同，應重新訓練再評估新 mask 的效果。
- 舊 checkpoint 缺新增欄位時按 onehot／context_only 載入。binary 的投影／velocity 尺寸不同，需新訓練；不使用 `strict=False` 續訓。
- resume 會拒絕不相容的模型配置或 K/Q／任務配置。
- 數值訓練正規化仍使用整張 prior table；生成 wrapper 使用 context 統計。四組沿用相同政策，這個已有差異另作後續研究。
- zero-context ablation 明確使用均勻分類 y 或模型空間數值 y=0；不是一般 context-conditioned 生成。
- 新版本尚無訓練 checkpoint 或下游效用結果，不能據此宣稱效果提升。

## 參考文件

- [AI 修改指南](docs/GenTFMv1.1_to_v1.2_AI修改指南.md)
- [論文閱讀筆記](docs/GenTFM_論文閱讀筆記與版本規劃_2026-10-04.md)
- [實作記錄](IMPLEMENTATION.md)
