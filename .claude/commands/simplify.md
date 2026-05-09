## /simplify

Pre-commit code quality sweep. Run before every commit.

### Steps

1. `uv run ruff check . --fix`
2. `uv run ruff format .`
3. Scan staged files for: unused imports, dead variable assignments, helper functions called only once that add no clarity, commented-out code blocks.
4. For each issue found: `file:line — removed/simplified X`
5. If nothing to simplify: `PASS: no simplifications needed`

### Rules

- Do not refactor beyond the staged diff — only touch what's being committed.
- Do not introduce abstractions. Remove them if they serve a single caller.
- Line length 120 (ruff enforced). Do not add comments explaining what code does.
