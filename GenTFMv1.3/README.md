# GenTFMv1.3

从 **v1.1** 升级，参考 [Tabular Data Generation using Binary Diffusion，第 3–4 节](https://arxiv.org/html/2409.13882v1#S4) 的全表二进制表示与 XOR 加噪，同时保留 **flow matching**。

## 数据表示

- 数值列：`(x - min) / (max - min)`，转为 `float32`，再逐位保存 IEEE-754 的 **32 位**，不是整数分箱或 32 位定点数。常数列映射到 0，反解码恢复常数。
- 分类列：类别 ID 从 0 开始，使用 MSB-first binary encoding。每列预留 `ceil(log2(cat_cardinality))` 位，具体表未使用的高位补零并屏蔽。
- 沿用 v1.1 的补零、observed mask 和监督目标约定；编码张量用 `float32` 存储，元素全部是 0/1。
- metadata 保存 `cont_min`、`cont_max`、分类映射和编码版本。`decode_to_dataframe` 恢复原列名、类别和数值范围。

默认 schema 的行宽为 `32*32 + 8*4 + 32 = 1088`。

## Binary diffusion 与 flow matching

全部生成维度（包括数值的 32 位）使用二进制状态。以 `t=0` 表示噪声、`t=1` 表示数据：

```text
x_t = x_1 XOR Bernoulli((1-t)/2)
p_t(bit | x_1) = (1-t)*Bernoulli(0.5) + t*delta(x_1)
```

训练采用 [Discrete Flow Matching](https://arxiv.org/abs/2407.15595) 的 endpoint prediction 参数化：现有 velocity head 输出干净位的 logits，以 BCE 学习条件端点分布。该分布定义离散概率速度；从当前位 `b` 跳到 `1-b` 的速率为：

```text
u_t(1-b, b) = P_theta(x_1=1-b | x_t, t, context, y) / (1-t)
```

采样从 Bernoulli(0.5) 开始，通过上述概率速度逐步 XOR 翻转位。每一步的状态始终为 0/1，最后一步不在 `t=1` 处计算除法。原分类 CE、分类 context calibration 和目标条件嵌入继续保留。生成结果会修复非法类别码，以及 NaN/Inf 或超出 `[0,1]` 的浮点位串。

这是将参考论文的编码和 binary diffusion 加噪路径适配到 **离散 flow matching**，不是逐字复现论文的双输出去噪网络和采样算法。没有新增独立的 noise head。`normalize=True` 参数为兼容原训练入口保留，数值归一化在编码阶段完成，不会再次标准化二进制位。

## 模型与版本

`gen_tfm/model.py` 与 v1.1 **逐字一致**。新的 `gen_tfm/flow_matching.py` 继承原模型，仅适配训练、目标提取和采样；encoder、cross-attention、时间/目标 embedding、head 结构、默认超参数和 TabICL prior 规则保持原样。输入输出宽度按新 schema 自动调整，这是编码变化所需的尺寸调整。

请使用 `from gen_tfm import GenTFM`（或 `gen_tfm.flow_matching`）。`gen_tfm.model` 是保留的底层 v1.1 网络定义，不是 v1.3 训练入口。

**需要重新训练。** 新 checkpoint 标记 `representation_version=binary_float32_v1_3`；加载或续训时明确拒绝旧版本 checkpoint。v1.1 和 v1.2 目录保留。

## 使用

在本目录安装 `requirements.txt` 中的依赖，然后训练：

```bash
python scripts/train.py --output_dir runs/v1_3_cls --target_task classification
python scripts/train.py --output_dir runs/v1_3_reg --target_task regression
```

`--n_ode_steps`、`--ode_method` 和 `method=euler/heun` 保留旧命令接口；v1.3 的两个 method 名称均执行二进制概率转移，不执行连续 Heun ODE。

对自己的表生成：

```python
from gen_tfm import (Schema, load_pretrained, encode_dataframe,
                     generate_in_context, decode_to_dataframe)

model, _ = load_pretrained("runs/v1_3_cls/gen_tfm_best_slim.pt", device="cpu")
schema = Schema(model.max_cont, model.max_cat, model.cat_cardinality)
encoded, metadata = encode_dataframe(df, schema, target_col="label")
generated = generate_in_context(model, encoded, metadata, num_gen=512, schema=schema)
synthetic_df = decode_to_dataframe(generated, metadata, schema)
```

回归用 `target_task="regression"`。直接调用 `model.generate(..., target_y=...)` 时，分类目标传类别 ID，回归目标传已 min-max 归一化的 float32 数值。默认从 context 的目标中有放回采样。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖 IEEE-754 位精确往返、min-max/常数列、分类补零、非法生成码修复、XOR 概率路径、离散 flow 的概率速度、分类/回归反向传播与条件生成、baseline/metrics、checkpoint 加载，以及 `model.py` 与 v1.1 一致性。
