# Gen-TFM: in-context synthetic data generation using a tabular foundation model

One model, pre-trained once on millions of synthetic tables. At test time you hand it a
few rows (the *context*) of a brand-new real table and it samples more rows of that
table, without any training on the table itself.

```text
x_new  ~  p_theta( x_new | context rows C, schema S )
```

This is the bet TabPFN / TabICL made for *prediction*, applied to *generation*.

| Stage | What happens | Code |
|---|---|---|
| Synthetic data engine | sample a random structural causal model (TabICL prior), draw N rows, turn it into a mixed-type table | `gen_tfm/prior.py`, `scripts/sample_prior.py` |
| Pre-training | split each synthetic table into K context rows + target rows, train with conditional flow matching + categorical cross-entropy | `gen_tfm/model.py`, `scripts/train.py` |
| In-context generation | encode K real rows, integrate the learned velocity field from noise to rows, snap categoricals | `gen_tfm/generation.py`, `notebooks/quickstart_in_context_generation.ipynb` |
| Evaluation | fidelity (MMD, per-column distances, correlations), utility (TSTR), privacy (distance to closest record) | `gen_tfm/metrics.py`, `scripts/evaluate_real.py` |

A project outline with pseudocode and the research questions is in
[`docs/PROJECT_OUTLINE.md`](docs/PROJECT_OUTLINE.md).  Reference numbers of the
frozen checkpoint are in [`docs/REFERENCE_RESULTS.md`](docs/REFERENCE_RESULTS.md).

## Installation

```bash
git clone --recursive <this repo>        # --recursive pulls the TabICL prior (external/tabicl)
cd gen-tfm
pip install -r requirements.txt          # or: pip install -e ".[train,notebook]"
```

`external/tabicl` is the public TabICL repository (BSD-3), pinned to the commit used for
all experiments. It is only needed for *training* (the synthetic data engine).
Generation and evaluation only need the checkpoint.

On LUMI everything runs inside the container
`/appl/local/laifs/containers/lumi-multitorch-latest.sif`; the container lacks `xgboost`
(needed by TabICL's tree prior), see `slurm/train_lumi.sh` for how it is added.

## Two ways to start

### A. Use the pre-trained checkpoint (minutes)

Download `gen_tfm_target_rich_100k.pt` (73 MB) from the GitHub release page of this repo
(or copy it from `/projappl/project_465002610/yanli/mingzhe_project/checkpoints/` on LUMI)
into `checkpoints/`, then open the notebook:

```bash
jupyter notebook notebooks/quickstart_in_context_generation.ipynb
```

or in Python:

```python
from gen_tfm import load_pretrained, encode_dataframe, decode_to_dataframe, generate_in_context, DEFAULT_CALIBRATION

model, ckpt = load_pretrained("checkpoints/gen_tfm_target_rich_100k.pt", device="cuda")
table, meta = encode_dataframe(df, target_col="income")     # df: pandas DataFrame
context = table[:100]                                       # K = 100 rows the model is allowed to see
generated = generate_in_context(model, context, meta, num_gen=1000, calibration=DEFAULT_CALIBRATION)
df_synth = decode_to_dataframe(generated, meta)
```

### B. Re-train from scratch (about 15 GPU hours)

```bash
mkdir -p logs
sbatch slurm/train_lumi.sh 2000     # smoke test, ~20 min
sbatch slurm/train_lumi.sh 100000   # the frozen recipe, ~15 h on one MI250x GCD
```

`scripts/train.py --help` lists every knob; the defaults *are* the frozen recipe
(medium MixSCM prior, 100k steps, batch 8, 768 rows per table, at least 512 target rows,
K in [5, 200]). The run ends with a synthetic held-out evaluation and a real-data
evaluation on the sklearn tables, and exports `gen_tfm_best_slim.pt`.

## Evaluate on real tables

```bash
# offline sanity check (4 sklearn tables, ~15 min on GPU)
python scripts/evaluate_real.py --checkpoint_path checkpoints/gen_tfm_target_rich_100k.pt \
    --output_dir results/sklearn --data_source sklearn --calibration logk --also_uncalibrated

# the "all-real" slice of the reference results (sklearn + TabSyn npy + OpenML Adult)
sbatch slurm/evaluate_real_lumi.sh checkpoints/gen_tfm_target_rich_100k.pt results/all_real all \
    --tabsyn_data_root /project/project_465002610/yanli/tabsyn/data --also_uncalibrated

# 35 BeyondArena datasets from Hugging Face (needs network once; ~1 h on GPU)
sbatch slurm/evaluate_real_lumi.sh checkpoints/gen_tfm_target_rich_100k.pt results/beyondarena hf_tabarena \
    --hf_tabarena_set broad --n_repeats 4 --eval_test_rows 256 --n_gen_rows 256

python scripts/summarize_results.py results/sklearn --metric encoded_mmd --output results/sklearn_k_curve.png
```

Each run writes `eval_per_dataset.csv` with one row per (dataset, K, repeat, method) and
these methods:

| method | meaning |
|---|---|
| `gen_tfm_matched` | Gen-TFM conditioned on the K context rows (the main result; calibrated unless `--calibration none`) |
| `gen_tfm_uncalibrated` | same without the categorical context calibration (with `--also_uncalibrated`) |
| `gen_tfm_zero` | Gen-TFM with an empty context (ablation) |
| `mixed_bootstrap` / `mixed_independent` / `mixed_conditional` | trivial context-only baselines |
| `real_vs_real` | held-out real rows vs. other real rows: the noise floor of every metric |

## Frozen inference setting

```text
checkpoint   = gen_tfm_target_rich_100k.pt    (target-rich medium MixSCM, 100k steps)
ODE          = euler, 60 steps
calibration  = logk schedule, alpha = 3.5, tau = 0.25     (gen_tfm.generation.DEFAULT_CALIBRATION)
```

The calibration adds `alpha_eff * log(smoothed context category frequencies)` to the
categorical logits at generation time, with `alpha_eff = alpha * log(1+K) / log(1+200)`.
It removed a failure mode where categorical proportions drifted at large K
(broad BeyondArena K=200 encoded MMD 0.0766 -> 0.0385, see `docs/REFERENCE_RESULTS.md`).

## Repository layout

```text
gen_tfm/
  encoding.py     padded mixed-type row encoding, feature masks, sanitising
  prior.py        synthetic data engine: TabICL SCM prior -> encoded tables
  model.py        GenTFM: context encoder + cross-attention flow network, loss, sampler
  generation.py   normalise context -> model.generate -> valid rows (+ calibration)
  metrics.py      MMD, Wasserstein, categorical JS, coverage/density, DCR privacy, TSTR
  baselines.py    bootstrap / independent / conditional context-only baselines
  real_data.py    DataFrame <-> encoding, sklearn / OpenML / TabSyn / BeyondArena loaders
  checkpoint.py   save / load / slim-export helpers
scripts/
  train.py, evaluate_real.py, sample_prior.py, summarize_results.py
slurm/            LUMI job scripts
notebooks/        quickstart: load the checkpoint, generate, inspect
docs/             project outline, reference results, final report of the previous phase
external/tabicl   TabICL (submodule, BSD-3), provides the synthetic prior
```

## Limits of the frozen model (know them before choosing datasets)

* schema cap: 32 continuous + 8 categorical columns, at most 12 levels per categorical
  column (rarer levels are merged into "other"); extra columns are dropped
* no missing values are modelled (every value is treated as observed)
* the model was trained with K in [5, 200]; larger contexts are untested
* rows are generated independently given the context: the *set* of generated rows has no
  explicit constraint (e.g. exact class balance)
* continuous columns are standardised and modelled as unbounded real values: no clipping to the
  observed range, no point masses (a column that is 90 % zeros, like Adult's `capital-gain`, comes
  out as a smeared distribution with negative values); simple marginal baselines can beat the
  model on such columns
