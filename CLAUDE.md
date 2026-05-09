# CLAUDE.md

## Response style
One sentence per update. No trailing summaries. State result, not process.
Between unrelated tasks (training → architecture → debugging): `/clear` to halve cache cost.
If Claude's direction is wrong after 2 corrections: `/clear` and restart. Rewind > correct.

## Commands

```bash
uv venv --seed && uv sync --group dev
uv run pytest                                          # all tests
uv run pytest tests/test_video_dataset.py              # single file
uv run ruff check . && uv run ruff format .
python src/train.py experiment=baseline_from_scratch +data=vianney
python src/evaluate.py training.checkpoint_path=src/best_model.pt +data=vianney
python src/create_submission.py training.checkpoint_path=src/best_model.pt +data=vianney
```

## Architecture

@.claude/context/project.md

## Code Conventions

**Imports** — explicit submodule paths: `from src.models.cnn_baseline import CNNBaseline` (no re-export at pkg root).

**Naming**: files/dirs `snake_case`, classes `PascalCase`. **Line length**: 120 (ruff enforced).

**Tensor contract** — all models: `(B, T, C, H, W)` → `(B, 33)`. Never break this.

**Adding a model** — 4-step checklist:
1. Implement `nn.Module` in `src/models/your_model.py` with `(B,T,C,H,W)` → `(B, 33)`.
2. Register in `src/train.py:build_model()`.
3. Add `src/configs/model/your_model.yaml`.
4. Add `src/configs/experiment/your_experiment.yaml`.

**Reads** — grep to locate, then `offset`/`limit=30`. Never read a file >100 lines without scoping.
**Bash output** — pipe through `| head -50`. **Git diffs** — `git diff --stat` first, then `git diff -- <path>`.
**Data paths** — always from Hydra config. Never hardcode `/Data/vianney.gauthier/...` in source.

## Agent Delegation

For training failures: use `/debug` first. For post-training iteration: use `/tune` first.
Spawn agents only when the slash command is insufficient.

| Trigger | Subagent | Model |
|---|---|---|
| Failing test / traceback (no clear root cause) | `debugger` | sonnet |
| Structural change >3 files | `refactorer` (`isolation: "worktree"`) | sonnet |
| Approach decision (architecture, new model) | `architect` (briefing first) | sonnet/opus |

**Output constraints** — `debugger`: SYMPTOM→ROOT CAUSE→FIX under 200 words; `architect`: write to `.claude/briefings/`, return path; `refactorer`: changed files+worktree path.

**Agent prompts must include output size cap** ("under 300 words" or "file paths only") to prevent context bleed.

**Verification map** — after any change, also run:

| Changed | Also run |
|---|---|
| `src/dataset/video_dataset.py` | `test_video_dataset.py` |
| `src/models/*.py` | `test_models.py`, `test_model_io.py` |
| `src/train.py` | `test_training_loop.py` |
| `src/utils.py` | `test_utils.py` |
| Multi-file refactor | `uv run pytest` (full suite) |

Never respond "done" without a passing test run. State explicitly if no coverage exists.

**Only the user commits/pushes.** Before commit: run `/simplify`. Before PR: run `/security-review`.

## Output Formats

**Training result**:
| Metric | Value |
|---|---|
| Top-1 accuracy | ... |
| Top-5 accuracy | ... |
Flag Top-1 < 0.5 with ⚠. `train.py` auto-writes `.claude/results/YYYYMMDD-<experiment>.json` (includes `warning: "TOP1_LOW"` when applicable).

**Debug**: `SYMPTOM:` → `ROOT CAUSE:` → `FIX:`  |  **Code change**: `file.py:line — what and why`

## Configuration

```bash
python src/train.py experiment=baseline_pretrained +data=vianney training.lr=0.001
```
Data paths in `src/configs/data/vianney.yaml` — edit there, never in source.
