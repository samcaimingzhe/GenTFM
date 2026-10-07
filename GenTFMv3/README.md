# GenTFMv3

SCM 合成表格 → 冻结 pretrained TabICL encoder → 每行 hidden embedding。
先训练 decoder 验证重建，再训练 schema-conditioned latent flow。

输入为连续数值和类别 ID，没有 binary/onehot 展开或 mask 通道。`gen_tfm/table.py` 管理 schema、有效列和类别合法性；旧 `encoding.py` 已移除。默认张量宽度为 32 连续槽 + 8 类别槽 = 40，未使用槽位为 padding；送入 encoder 时去掉无效列。有效行由 row mask 标识。

```sh
pip install -r requirements.txt
OMP_NUM_THREADS=1 python scripts/train_decoder.py --output_dir runs/decoder --device cuda --steps 10000
OMP_NUM_THREADS=1 python scripts/train_latent.py --decoder_checkpoint runs/decoder/best.pt --output_dir runs/flow --device cuda --steps 10000
```

先检查 decoder 在全新 SCM 表格上的重建指标，满意后再开始第二阶段。CPU 可使用 `--device cpu --steps 2 --batch_size 1 --rows 16 --width 32 --save_every 1`，decoder 可额外使用 `--calibration_tables 1 --validation_batches 1`；flow 可额外使用 `--heads 4 --layers 1`。

表征仍为 `col_embedder + row_interactor` 的 512 维行向量。本次没有切换到 cell/feature 级 hidden。

```sh
python -m unittest discover -s tests
```

[Decoder 训练说明](DECODER.md) · [Latent flow 说明](LATENT_FLOW.md) · [验证记录](VALIDATION.md)

当前不支持缺失值、原始字符串、参考表格条件生成。连续值仍由 prior 标准化。schema 版本为 `raw_ids_v1`，旧 binary/onehot 表格及训练 checkpoint 不兼容，需要重新训练 decoder 和 flow；原始 TabICL 预训练权重可继续使用。

## 中断续训

```sh
OMP_NUM_THREADS=1 python scripts/train_decoder.py --resume runs/decoder/latest.pt --steps 20000
OMP_NUM_THREADS=1 python scripts/train_latent.py --resume runs/flow/latest.pt --steps 20000
```

`--steps` 是最终总步数，必须大于保存步数。其余训练参数自动读取 checkpoint。详见 [恢复训练说明](RESUME.md)。
