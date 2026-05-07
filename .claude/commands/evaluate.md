## /evaluate

Run evaluation on the current best checkpoint.

Usage: `/evaluate` or `/evaluate <checkpoint_path>`

### Steps
1. `python src/evaluate.py training.checkpoint_path=${ARGUMENTS:-src/best_model.pt} +data=vianney | head -30`
2. Report Top-1 and Top-5 in the standard table:

| Metric | Value |
|---|---|
| Top-1 accuracy | ... |
| Top-5 accuracy | ... |
