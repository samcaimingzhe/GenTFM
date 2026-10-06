# GenTFMv1.1

GenTFMv1.1 adds a supervised target embedding to the v1 flow-matching generator. The target is encoded separately, excluded from the generated feature dimensions, and restored after conditional sampling.

## Target convention

- Classification: the TabICL prior's target is the final categorical field. Its index is stored as `label_cat_index` in table metadata.
- Regression: the prior's first source feature is used as a continuous target and encoded as the first continuous field. Its index is stored as `label_cont_index`.
- At generation time, target values are sampled with replacement from the context labels. The model then generates the remaining fields conditioned on those values and the full labeled context.

The embedding is fused additively with the existing noisy-row and time embeddings in `MixedFlowNet`; the encoder, cross-attention stack, flow-matching objective, and ODE solver remain in place. New v1.1 checkpoints are required because the model adds target-embedding parameters.

## Training

Install the dependencies in `requirements.txt` and make TabICL importable (installed package or `TABICL_SRC`). Then train with:

```bash
python scripts/train.py --output_dir runs/v1_1_cls --target_task classification
python scripts/train.py --output_dir runs/v1_1_reg --target_task regression
```

Use the resulting v1.1 checkpoint with this directory's generation and evaluation code. The regression prior uses an SCM feature as its numeric target because TabICL's built-in prior supplies classification labels.
