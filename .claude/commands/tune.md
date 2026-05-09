## /tune

Analyze a training curve and propose the next hyperparameter experiment.

Usage: `/tune` — run after a training session when results are in `.claude/results/`.

### Prompt template — fill from training output or `.claude/results/`

TRAINING CURVE (last N epochs):
| epoch | train_loss | val_loss | val_acc |
|---|---|---|---|
| paste rows here |

CONFIG:
```yaml
# paste relevant block — lr, optimizer, scheduler, batch_size, num_frames
```

### Steps

1. Submit the filled template with: "Give 3 hypotheses for what this curve indicates, ranked by
   likelihood. No code yet."
2. For each hypothesis: "What single config change would test this? What result confirms or refutes it?"
3. Choose one experiment. Edit the config YAML only — do not change model architecture.
4. Run `/train`, then `/retrospective <name>` to log the result.
5. Never change more than one hyperparameter per run — otherwise the curve is uninterpretable.
