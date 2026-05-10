## /lint

Run ruff on the full project and report errors.

### Steps
1. `uv run ruff check . | head -50`
2. `uv run ruff format --check . | head -20`
3. Report: number of errors by file. If clean, say "✓ no issues".
