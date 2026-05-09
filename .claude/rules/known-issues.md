## Known issues and environment gotchas

### 4 frames on disk vs num_frames=8

Every video folder contains only `frame_000.jpg … frame_003.jpg` (4 frames). The dataloader samples `num_frames` indices uniformly, so with the default `num_frames: 8` it samples with replacement, introducing duplicate frames. Consider overriding to `dataset.num_frames=4` until a fuller frame extraction is done.

### GPU requirement

Training requires a CUDA-capable GPU. CPU-only runs will OOM or be prohibitively slow on frame batches.

### Data location

Frames live at `/Data/vianney.gauthier/processed_data/val2/`. On a different machine, update `src/configs/data/vianney.yaml` only — do NOT modify `default.yaml`.

### Hydra outputs

Every `python src/train.py` run creates a timestamped folder under `src/outputs/`. These accumulate; clean periodically with `rm -rf src/outputs/`.

### Frame count

`num_frames` in the data config must match the model's expected T. Mismatches cause silent shape errors downstream; assert shapes in tests.

### uv environment

Always invoke via `uv run` or activate `.venv`. Do not use system Python — package versions will diverge.

### `src/outputs/` in git diffs

Hydra writes run logs to `src/outputs/`. The `.gitignore` covers `outputs/` at any depth, so these should not be staged. Verify with `git status` before committing.
