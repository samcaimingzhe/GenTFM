# GenTFMv2.1

This directory is a copy of GenTFMv2 with a different checkpoint-selection rule. The model architecture and loss formula are unchanged.

## Checkpoint selection

- `loss`: the loss on one newly sampled training batch, measured before its optimizer update.
- `best_loss`: the minimum finite single-step training loss observed so far. `best.pt` is saved whenever this improves.
- `best.pt` contains the **pre-update** model and optimizer state that produced `best_loss`. Its `step` is the number of completed optimizer updates; `best_loss_step` is the one-based number of the batch on which the score was observed. Thus `best_loss_step == step + 1` when a best checkpoint is written.
- `latest.pt` contains the latest post-update state, the historical `best_loss`, and the historical `best_validation_loss`.
- `best_validation_loss` inside a full `best.pt` is the validation history **at the moment that file was saved**. It is not the validation score of the saved weights and may be older than the value in `latest.pt`.
- `validation_loss` and `best_validation_loss` remain diagnostic metrics. They do not select `best.pt`.

The selection rule is stored as `train_config["checkpoint_selection"] == "single_step_train_loss_pre_update"`.

## Training and resuming

From this directory, run `python scripts/train.py --output_dir /path/to/new_run` with the desired training arguments. Use a **new output directory** when switching from a V2 checkpoint because its saved `best_loss` meant validation loss. You may pass the V2 `latest.pt` to `--resume`; V2.1 resets the training-best threshold and starts selecting from the newly resumed steps. The original `validation_set.pt` must be available in the resumed run directory or beside the source `checkpoints/` directory so that validation remains comparable.

When resuming a V2.1 run, the model configuration and training-loss selection settings must match. The code reconciles `best_loss` against the available `best.pt` file, so a more recent best checkpoint is not silently overwritten after a restart from an older `latest.pt`.

This change does **not** fix the V2 near-constant-column normalization issue. A training batch with a tiny context standard deviation can still produce a very large loss. See [CODE_REVIEW_ZH.md](CODE_REVIEW_ZH.md) for the diagnosis.

The minimum loss from one random training batch can favor an unusually easy batch. Keep the separate validation history when judging overall model quality.
