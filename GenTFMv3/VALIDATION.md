# Raw-table pipeline validation

- Seven tests passed: raw IDs and legacy-contract rejection, real TabICL API/freeze/variable rows, velocity loss weighting/gradients, padding/permutation/sampling, normalization, decoder loss masks/legal IDs, paired-embedding overfit.
- Official TabICL v1 pretrained weights with Mix-SCM: two-step decoder training and independent validation completed (validation loss 2.53263 -> 2.53153).
- Two-step flow training from decoder checkpoint completed (fixed validation velocity loss 2.37076 -> 2.36583).
- Generated raw latent embeddings decoded into finite 40-column-slot tables with legal category IDs.
- Exact latent normalization reuse across stages verified.
- Classification and regression prior batch outputs checked: both [2,16,40], category IDs valid.

These smoke runs verify execution only, not convergence or reconstruction/generation quality. Encoder still produces 512-dimensional row embeddings; cell-level representations were not introduced. On this Mac OMP_NUM_THREADS=1 avoids a native XGBoost threading crash.

## Resume update

- Nine unit tests passed, including safe RNG checkpoint roundtrip, inherited configuration, incompatible configuration rejection, and incomplete legacy resume-state rejection.
- Official pretrained TabICL v1 + Mix-SCM CPU integration: continuous 4-step decoder training exactly matched 2 steps + resume to step 4 (weights, normalization, history, prior counter).
- The same continuous/split comparison passed exactly for flow training.
- Resume with a changed row count was rejected before training.

Reproduce integration checks with `python tests/check_resume_pipeline.py --encoder_checkpoint /path/to/official-v1.ckpt --work_dir /path/to/test-runs`. Exact equality is verified only for this CPU environment and short run, not across hardware/software changes.
