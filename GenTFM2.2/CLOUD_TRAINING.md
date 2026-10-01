# GenTFM v2.2 Cloud Training Runbook

This file is the canonical handoff for running GenTFM v2.2 on Google Colab or another cloud GPU. When asked to run or prepare a cloud training job, read this file first and use the standard recipe below unless the user explicitly gives a different target. Do not ask the user to restate the batch size, table count, context sizes, or output names already specified here.

## Standard production run

Train on 100,000 synthetic prior tables with batches of 8:

```text
batch size       = 8 tables per optimizer step
optimizer steps  = 12,500
training tables  = 8 × 12,500 = 100,000
rows per table   = 1,024 total context + target rows
context sizes    = randomly choose from 20, 50, 100, 200, 500 rows
target rows      = all remaining rows (minimum 512)
prior            = mix_scm
device           = cuda:0 when CUDA is available
```

The 100,000 count excludes the fixed validation batches. With 500 context rows, each 1,024-row table has 524 target rows. Do not describe this run as 100,000 optimizer steps.

Run all commands from the `GenTFM2.2` project directory. Use this command for the standard run:

```bash
python -m script.train \
  --steps 12500 --batch-size 8 --num-rows 1024 \
  --min-context 5 --max-context 500 --min-target 512 \
  --context-sizes 20 50 100 200 500 \
  --num-cross-blocks 2 \
  --categorical-loss-weight 0.8 --discrete-flow-weight 0.05 \
  --device cuda:0 --log-every 100 --val-every 100 --save-every 100 \
  --output-dir runs/v2.2_conditional_100k
```

The trainer samples new prior tables as it runs. TabICL prior generation currently runs on CPU; the model and training tensors use the selected device. GPU memory use for batch size 8 depends on the cloud GPU and has not been established for every provider.

## Colab and cloud setup

1. Start a GPU runtime and confirm that the runtime exposes a CUDA-enabled PyTorch build. On Colab, mount persistent Drive storage before copying the project or starting training. On another provider, choose a persistent volume for the project and run output.
2. Place or clone the complete project on persistent storage. The repository includes an `external/tabicl` submodule path; if the clone did not initialize submodules, run `git submodule update --init --recursive` from the repository root. The prior adapter can also use an installed TabICL package.
3. From `GenTFM2.2`, install dependencies. Install the provider-compatible PyTorch build first if needed, then install the project requirements:

   ```bash
   python -m pip install -r requirements.txt
   ```

   `requirements.txt` pins TabICL to 2.1.1 and includes XGBoost. Avoid replacing a working CUDA PyTorch installation with a CPU-only build.
4. Confirm the runtime and prior can be imported:

   ```bash
   python -c "import torch; from tabicl.prior import PriorDataset; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
   ```

   If the project uses its vendored submodule instead of an installed package, test the project adapter import from the project directory:

   ```bash
   python -c "import torch; from data.prior import TabICLPriorEngine; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
   ```

5. Keep `--output-dir` on persistent storage. Cloud notebook disks can be erased when a runtime disconnects. Do not start the full run with its only checkpoint on ephemeral local storage.
6. Launch the standard production command above. Preserve the full terminal log, including step, train loss, validation loss, and the final output path.

## Memory adjustment

If batch size 8 does not fit the selected GPU, lower the batch size and increase steps so that their product remains 100,000. For example:

| Batch size | Steps | Training tables |
|---:|---:|---:|
| 8 | 12,500 | 100,000 |
| 4 | 25,000 | 100,000 |
| 2 | 50,000 | 100,000 |
| 1 | 100,000 | 100,000 |

Keep the row count, context-size set, model settings, prior settings, and seed unchanged when only adapting for memory. Note the chosen batch size and step count in the run report. A smaller batch means more optimizer updates for the same number of training tables; it is not exactly the same optimization schedule.

If even batch size 1 does not fit, stop and report the GPU model, peak memory/error, and attempted settings. Do not silently reduce the row count, schema, model width, or requested table count.

## Checkpoints, interruption, and resume

The trainer writes:

- `latest.pt`: resumable model, optimizer, scheduler, RNG, prior state, fixed validation batches, and loss history.
- `best.pt`: checkpoint with the best validation loss so far.
- `loss_curve.png`: training and validation loss plot, written at normal completion.

`latest.pt` is saved at step 0, after step 1, every `--save-every` steps, and at validation steps. With the standard command, validation and saving occur every 100 steps. On an interruption, resume from the persistent `latest.pt`:

```bash
python -m script.train \
  --resume runs/v2.2_conditional_100k/latest.pt \
  --device cuda:0 \
  --log-every 100 --val-every 100 --save-every 100
```

When resuming, the checkpoint restores the original training configuration. `--steps` is the original total planned step count, not the number of additional steps; leave it out unless explicitly matching the original value. The trainer allows device, output directory, and logging/validation/save intervals to change. Do not resume from `best.pt` if the goal is to continue the latest training state; use `latest.pt`.

After normal completion, confirm that the last logged step is 12,500 (or the adjusted total), and that `best.pt`, `latest.pt`, and `loss_curve.png` exist in the persistent output directory. Copy/download those artifacts before deleting or shutting down the cloud runtime.

## Generation from a trained checkpoint

For generation, use `best.pt` by default. Provide clean context rows in the model's encoded schema plus matching metadata (`n_cont`, `n_cat`, and `cat_cardinalities`). The raw context must include the columns to be generated and be in the original encoded scale expected by `generate_in_context`; the wrapper normalizes using context statistics and restores the continuous scale afterward. No true target rows are required.

Example code is maintained in the `Conditional generation` section of `README.md`. Keep the generation batch size fixed when comparing runs because target rows interact within each generated batch. Save the checkpoint path, context source, context row count, seed, number of generated rows, sampler (`euler` or `heun`), and step count with the output.

## Short validation run

Before a production run on a new cloud environment, a two-step run can confirm imports, prior sampling, CUDA placement, checkpoint writing, and loss plotting. Use the small-training command in `README.md`, directing output to a disposable directory such as `runs/cloud_smoke`. This is an environment check only; it does not validate model quality or estimate production-run duration.

## Run report

At the end, report:

- cloud provider and GPU model;
- repository revision, if available;
- effective batch size, total steps, and table count;
- context-size set, rows per table, and prior type;
- whether the run completed or was interrupted;
- final train and validation losses when available;
- persistent paths for `best.pt`, `latest.pt`, `loss_curve.png`, and the log;
- any memory changes, warnings, or failures.

Do not claim training completed unless the final planned step completed and the output artifacts were confirmed on persistent storage.
