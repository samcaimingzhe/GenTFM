# Decoder-first training with raw tables

`tables`: `[B,N,40]` by default. Continuous columns occupy slots 0..31, categorical IDs slots 32..39. Metadata carries `n_cont`, `n_cat`, `cat_cardinalities`, `tabicl_seq_len`, `schema_version=raw_ids_v1` and `no_missingness=True`. Inactive columns and padded rows are excluded from losses. Zero is a legal category ID; never infer validity from values.

## Stage 1

```sh
OMP_NUM_THREADS=1 python scripts/train_decoder.py --output_dir runs/decoder --device cuda --steps 10000
```

First use downloads official non-target-aware TabICL v1 weights. Offline: `--encoder_checkpoint /path/to/tabicl-classifier-v1-20250208.ckpt`.

The encoder sees active continuous columns plus scalar categorical IDs directly. It remains frozen. Latent normalization is calibrated once on independent draws. A row-wise MLP receives standardized latent plus schema conditions and predicts continuous values and per-column category logits.

`loss = continuous_MSE + cat_weight * categorical_CE` (default category weight 1).
Each component averages within each table, then over tables. Categorical logits are restricted to valid cardinalities. Missingness is not modeled.

Outputs: `latest.pt`, validation-selected `best.pt`, `losses.json`, `config.json`, and `reconstruction.pt` containing original/reconstructed raw tables and row masks. Checkpoints embed encoder weights/config, decoder weights/config, fixed latent statistics, schema and prior settings. Both stages support --resume; see RESUME.md.

Check continuous MSE and category accuracy on many unseen SCM tables before training flow. Two-step smoke runs verify execution only. Current prior does not preserve original SCM scales/category mappings; reconstruction targets the standardized/adapted table.

## Stage 2

```sh
OMP_NUM_THREADS=1 python scripts/train_latent.py --decoder_checkpoint runs/decoder/best.pt --output_dir runs/flow --device cuda --steps 10000
```

Flow reuses the exact encoder, schema and normalization from stage 1. Decoder is frozen and included in flow checkpoints. For nondefault prior types, pass the matching `--prior_type`.

## Decode hidden embeddings

```sh
python scripts/decode_hidden.py --decoder_checkpoint runs/decoder/best.pt --input runs/flow/sample_embeddings.pt --output runs/flow/generated_tables.pt --sample_categories
```

Input files contain raw-scale latent `embeddings`, `row_mask`, and `metadata`. The command standardizes embeddings with stage-1 statistics before decoding. Omit `--sample_categories` for reconstruction argmax. Use the same pretrained encoder; embedding files do not verify encoder identity cryptographically.

Output: raw standardized continuous columns and legal scalar category IDs. No bit/onehot codec is used. `row_mask` and metadata identify padding. Old codec checkpoints fail schema construction and must be retrained.
