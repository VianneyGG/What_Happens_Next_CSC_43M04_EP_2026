## /train

Run a training experiment and report results.

Usage: `/train <experiment_name>`  (e.g. `/train baseline_pretrained`)

### Steps
1. `python src/train.py experiment=$ARGUMENTS +data=vianney | head -60`
2. `python src/evaluate.py training.checkpoint_path=src/best_model.pt +data=vianney | head -20`
3. Report:

| Metric | Value |
|---|---|
| Top-1 accuracy | ... |
| Top-5 accuracy | ... |

Flag Top-1 < 0.5 with ⚠. Write full output to `.claude/results/YYYYMMDD-<experiment>.md`.
