## /debug

Diagnose a failing model, training loop, or data pipeline issue.

Usage: `/debug <brief description>` (e.g. `/debug val loss diverges after epoch 3`)

### Prompt template — fill this before submitting to Claude

SYMPTOM: <what you observe — loss values, error message, unexpected output>
CODE: <paste only the relevant function — train_one_epoch(), forward(), etc.>
CONFIG: <paste relevant YAML block or key hyperparameters>
ALREADY TRIED: <what you changed and what happened>

### Steps

1. Submit the filled template with: "Do not write any code yet. Using SYMPTOM → ROOT CAUSE → FIX
   format, diagnose the most likely root cause. Ask clarifying questions if needed."
2. If the diagnosis is wrong after 2 exchanges: `/clear` and restart with more context.
3. Once root cause is confirmed: "Propose the minimal fix — no refactoring, no new abstractions."
4. `git diff` before applying. Verify tensor contract: `(B, T, C, H, W) → (B, 33)`.
5. Run the relevant test: `uv run pytest tests/test_models.py` or `tests/test_training_loop.py`.
