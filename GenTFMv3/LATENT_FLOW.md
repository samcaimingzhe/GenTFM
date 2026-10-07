# Raw-table embeddings and flow

Whole SCM tables contain standardized continuous values and scalar category IDs. `table.py` owns schema and validation only; no category expansion or observed-mask channels.

The frozen TabICL v1 `col_embedder + row_interactor` produces `[B,N,512]` hidden embeddings. All valid rows form the column-attention context. Target is an ordinary table column. Target-aware checkpoints are rejected rather than bypassing labels. Tables are encoded individually, then padded with explicit row masks.

Stage 1 trains reconstruction; stage 2 requires its decoder checkpoint:

```sh
OMP_NUM_THREADS=1 python scripts/train_latent.py --decoder_checkpoint runs/decoder/best.pt --output_dir runs/flow --device cuda
```

Fixed latent normalization is loaded from stage 1. With normalized data endpoint z1, Gaussian z0 and one uniform time per table:

```
zt = (1-t)*z0 + t*z1
velocity_target = z1-z0
loss = mean_tables(sum_valid_rows_dims((flow(zt,t,schema)-velocity_target)^2)
                   / (valid_rows * hidden_dim))
```

Only flow parameters update. The Transformer mixes rows without positional embeddings; schema conditions carry continuous-slot activity and categorical cardinalities. Sampling uses Heun integration t=0..1. This learns the mixture of synthetic tables for a schema, not generation conditioned on an observed reference table.

Outputs: `latest.pt`, `losses.json`, `config.json`, `sample_embeddings.pt`. Generated embeddings are restored to raw latent scale for storage. Decode them using `decode_hidden.py` (see DECODER.md). No decoded quality claim follows from flow loss alone.

To encode your own raw batch, save `{'tables': tensor, 'metadata': list, 'schema': dataclasses.asdict(schema)}` and run:

```sh
python scripts/extract_hidden.py --input table_batch.pt --encoder_checkpoint /path/to/encoder.ckpt --output hidden.pt
```

Use finite numeric columns and zero-based category IDs, the raw schema contract and explicit column/cardinality metadata. Original strings require a saved category mapping. Currently missing values are unsupported.

On this Mac use `OMP_NUM_THREADS=1` to avoid a native XGBoost threading crash in the SCM prior. `--prior_type mlp_scm` isolates the MLP prior. Existing codec decoder/flow checkpoints must be retrained for `raw_ids_v1`.
