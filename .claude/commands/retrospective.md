## /retrospective

Document the current session's findings to the experiment registry.

Usage: `/retrospective <experiment_name>` (e.g. `/retrospective cnn_lstm_lr1e-3`)

### Steps

1. Determine experiment name from `$ARGUMENTS` or the most recent `/train` call in this session.
2. `mkdir -p .claude/experiments/$ARGUMENTS`
3. Write `.claude/experiments/$ARGUMENTS/SKILL.md` using the template below.
   Fill every section from actual session results — no placeholders.
4. Append one line to `.claude/experiments/index.md`:
   `- [$ARGUMENTS]($ARGUMENTS/SKILL.md) — <one-line result summary>`
5. Confirm: "Retrospective saved to .claude/experiments/$ARGUMENTS/SKILL.md"

### SKILL.md template

```markdown
# Experiment: {name}

**Date:** {YYYY-MM-DD}
**Config:** `python src/train.py experiment={exp} +data=vianney {overrides}`

## Results

| Metric | Value |
|---|---|
| Top-1 accuracy | |
| Top-5 accuracy | |
| Epochs | |

## What Worked

- 

## Failed Attempts

| Symptom | Root Cause | Fix Applied |
|---|---|---|
| | | |

## Exact Hyperparameters

```yaml
# paste relevant cfg values here
```

## Notes

```
