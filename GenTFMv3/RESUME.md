# Resume training

Both training entry points support `--resume`. Start a new run normally; checkpoints written by this update contain complete resume state.

```sh
OMP_NUM_THREADS=1 python scripts/train_decoder.py --resume runs/decoder/latest.pt --steps 20000
OMP_NUM_THREADS=1 python scripts/train_latent.py --resume runs/flow/latest.pt --steps 20000
```

`--steps` is the **total target step**, not additional steps. A checkpoint at 5000 with `--steps 20000` resumes at 5001 and ends at 20000.

Training arguments automatically inherit saved values; you do not need to repeat width, rows, batch size, prior, learning rate, or seed. Source encoder/decoder checkpoint files are not required: model representations are embedded.

Restored state includes model and optimizer, frozen encoder, fixed latent mean/std, step/history, decoder's best validation loss, validation data (and flow validation noise/times), SCM engine generated-batch count, Python/NumPy/PyTorch CPU and CUDA random states. The prior cache is rebuilt before restoring RNG.

Only `--steps`, `--output_dir`, `--device`, and `--save_every` may change. Incompatible training arguments fail before training. CPU/GPU changes are permitted for practicality but numerical or random-stream equality across devices is not guaranteed; CUDA device-count changes can be rejected when restoring CUDA RNG. Use the same dependencies/hardware for reproducibility.

Resume from `latest.pt` is recommended. Decoder `best.pt` also has resume state, but resumes from the selected best step rather than the latest step. Keep the original output directory if you want to retain the historical `best.pt`; a new output directory only gets a new best file when validation improves on the restored historical best score.

Checkpoint writes are atomic. Saving occurs at step 1, every `--save_every` steps (default 100), and the final step. If interrupted, work since the last completed checkpoint is lost. Reducing `--save_every` trades write overhead for less lost work. There is no signal handler or checkpoint-on-crash guarantee.

Checkpoints made before this update lack complete RNG/validation state and are explicitly rejected for resume. They still work for stage-2 initialization or decoding; do not treat loading those weights as exact continuation.
