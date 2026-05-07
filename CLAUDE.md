# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Commands

```bash
# Setup
uv venv --seed && uv sync --group dev

# Tests
uv run pytest                                          # all tests
uv run pytest tests/test_video_dataset.py              # single file
uv run pytest tests/test_models.py::test_name          # single test

# Lint & format
uv run ruff check .
uv run ruff format .

# Training
python src/train.py experiment=baseline_from_scratch +data=vianney
python src/train.py experiment=baseline_pretrained +data=vianney

# Evaluation
python src/evaluate.py training.checkpoint_path=src/best_model.pt +data=vianney

# Submission
python src/create_submission.py training.checkpoint_path=src/best_model.pt +data=vianney
```

## Architecture

@.claude/context/project.md

## Code Conventions

**Imports** — explicit submodule paths:

```python
from src.dataset.video_dataset import VideoDataset    # correct
from src.models.cnn_baseline import CNNBaseline       # correct
from src.models import CNNBaseline                    # wrong — no re-export at package root
```

**File/dir naming**: `snake_case`. **Class naming**: `PascalCase`.

**Tensor contract** — all models receive `(B, T, C, H, W)` and return `(B, 33)`. Never break this.

**Adding a new model** — 4-step checklist:
1. Implement `nn.Module` in `src/models/your_model.py` with `(B,T,C,H,W)` → `(B, 33)` I/O.
2. Register in `src/train.py:build_model()`.
3. Add `src/configs/model/your_model.yaml`.
4. Add `src/configs/experiment/your_experiment.yaml`.

**Reads** — grep to locate, then read with `offset`/`limit`. Never read a file >100 lines unless the whole file is the task.

**Bash output** — pipe any Bash command that may return many lines through `| head -50`.

**Git diffs** — always `git diff --stat` first; then per-file `git diff -- <path>`. Never run bare `git diff` (Hydra outputs alone can spike token count).

**Data paths** — always from Hydra config (`cfg.dataset.train_dir`, etc.). Never hardcode `/Data/vianney.gauthier/...` in source.

**Line length**: 120 (ruff enforced).

## Agent Delegation Rules

These are **MUST** rules, not suggestions. Before executing a task, check this list and delegate if it matches.

| Trigger | Required subagent | Model |
|---|---|---|
| File search / codebase exploration spanning >2 files | **MUST** use `explorer` | haiku |
| After ANY code change | **MUST** use `test-runner` — unit tests for changed files + mapped tests below | haiku |
| After any code change (before responding "done") | **MUST** use `linter` | haiku |
| After any non-trivial implementation | **MUST** use `code-reviewer` — write briefing first | sonnet |
| Failing test, error traceback, or unexpected behavior | **MUST** use `debugger` | sonnet |
| Structural change spanning >3 files | **MUST** use `refactorer` (with `isolation: "worktree"`) | sonnet |
| Approach decision before implementation | **MUST** use `architect` — write briefing first | sonnet → opus for system-wide |

**General rule**: if a task produces >50 lines of output, delegate.

### Verification map

After any code change, run the unit tests for changed files **plus**:

| Changed component | Also run |
|---|---|
| `src/dataset/video_dataset.py` | `test_video_dataset.py` |
| `src/models/*.py` | `test_models.py`, `test_model_io.py` |
| `src/train.py` | `test_training_loop.py` |
| `src/utils.py` | `test_utils.py` |
| Any multi-file refactor | Full suite: `uv run pytest` |

**Verification gate:** Never respond "done" without a passing test run. If changed code has no test coverage, say so explicitly — do not silently skip.

### Skills to invoke

- Before any commit the user approves → run `/simplify` on the changed files
- Before opening a PR → run `/security-review`

### Commit policy (non-negotiable)

**Only the user commits.** No subagent, and no main-session turn, may run `git commit`, `git push`, `git merge`, `git rebase`, `git tag`, or `git revert` unless the user explicitly asks for it in the current turn.

### Invocation patterns

- **Worktree isolation**: when calling `refactorer`, always pass `isolation: "worktree"`. Optional for `debugger` on large fixes.
- **Background execution**: use `run_in_background: true` for `test-runner` and `linter` when they run alongside other work.
- **Parallel research**: for broad searches, spawn up to 3 `explorer` agents in a single message with disjoint scopes.
- **Briefing files**: for complex delegations (>3 files, non-obvious constraints), write `.claude/briefings/<task>.md` first, then pass the path.
- **Output constraints**: `test-runner` → failures + first 10 traceback lines only; `linter` → error-level `file:line: message` only; `explorer` → file paths and line numbers, no quoted blocks; `code-reviewer` → write findings to `.claude/briefings/review-YYYYMMDD.md`, return path only; `architect` → write findings to `.claude/briefings/arch-YYYYMMDD.md`, return path only; `debugger` → `SYMPTOM:` → `ROOT CAUSE:` → `FIX:` only, under 200 words; `refactorer` → changed files list + worktree path only.

## Output Formats

**Training result** — always:
| Metric | Value |
|---|---|
| Top-1 accuracy | ... |
| Top-5 accuracy | ... |
Flag Top-1 < 0.5 with ⚠. Write full output to `.claude/results/YYYYMMDD-run.md`.

**Debug trace** — `SYMPTOM:` → `ROOT CAUSE:` → `FIX:`

**Code change** — `file.py:line — what changed and why`

## Configuration

No secrets needed (data is local). Hydra config tree under `src/configs/`:

```bash
# Select model + experiment
python src/train.py experiment=baseline_pretrained +data=vianney

# Override a single param
python src/train.py experiment=baseline_pretrained +data=vianney training.lr=0.001
```

Data paths are in `src/configs/data/vianney.yaml` — edit there, never in source code.
