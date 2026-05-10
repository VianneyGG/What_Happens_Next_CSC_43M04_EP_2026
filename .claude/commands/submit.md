## /submit

Generate a submission CSV from the current best checkpoint.

### Steps
1. `python src/create_submission.py training.checkpoint_path=src/best_model.pt +data=vianney | head -20`
2. Confirm the output CSV path from the log.
3. Report: number of predictions written and output file path.
