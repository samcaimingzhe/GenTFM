# v1.2 實作記錄

基準：`/Users/caimingzhe/Desktop/new/GenTFMv1.1`。本目錄是獨立副本，沒有複製 checkpoint 或訓練 outputs。

## 已接通

| 部分 | 實作 |
|---|---|
| Encoding | onehot/binary schema，NumPy/Torch codec，合法碼投影，explicit cardinalities／mask |
| Prior／real-data encoder | 分類 ID 透過共用 codec 寫入，metadata 保存 codec 和 version |
| Model | y codec 提取／回填，合法類別 CE／sampling，ID histogram 校準 |
| Attention | context-only／當層 noisy-query K/V，沿用既有 MHA、MLP 和殘差 |
| Training | 精確 Q、K/Q 約束、loss 分量與實際 loss 行數，JSON 配置和 CLI 覆蓋 |
| Generation | Euler／Heun 當前狀態更新，指定 y 的尺度轉換，encoded context 生成 CLI |
| Checkpoint | strict weights，legacy onehot/context-only defaults，配置衝突／resume 拒絕 |
| Utilities | codec-aware baselines、共同 onehot 評估幾何、分類／回歸 proxy、TabICL TSTR |
| Ablations | 四組匹配 K/Q 的 JSON 訓練配置 |

## 工程修正（套用所有消融組）

- label/task aliases 統一，`label_cat_index=None` 不再被當分類。
- 各欄 CE、sampling、校準依真實 cardinality 處理。
- no-missingness 生成不讓隨機 observed bits 抹去數值。
- zero-context 的 y 政策明確化；無分類欄的 conditional baseline 回退 independent baseline。
- 分類與回歸分別報 accuracy／R²；不把數值 y 作 argmax 分類。

## 檢查與尚未執行的項目

已做 Python 原始碼語法解析與呼叫端靜態檢查。未執行模型訓練、runtime smoke tests、codec／attention／checkpoint 測試或資料集評估；目前沒有實測改善數字。

後續驗收項目詳見 docs 中的指南，包括 codec round-trip、兩任務 loss／gradient、可變 K/Q、y 保持、attention 可見範圍、Heun 中間狀態、strict checkpoint 和四組 TabICL 效用消融。
