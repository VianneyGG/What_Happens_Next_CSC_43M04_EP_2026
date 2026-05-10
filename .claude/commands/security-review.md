## /security-review

Pre-PR security check. Run before opening any pull request.

### Steps

1. Scan all changed files (`git diff main...HEAD --name-only`) for:
   - Hardcoded paths, API keys, tokens, passwords, emails
   - `subprocess`, `os.system`, `eval`, `exec` calls with unsanitized input
   - Unchecked external inputs at system boundaries (CLI args, config values, file paths)
   - SQL/command injection vectors
   - Model checkpoint loading from untrusted sources (`torch.load` without `weights_only=True`)

2. For each issue: `RISK: file:line — description`
3. If no issues: `PASS: no security issues found`

### ML-specific checks

- `torch.load()` must use `weights_only=True` unless loading legacy checkpoints.
- Data paths must come from Hydra config — never from user input or environment variables directly.
- No hardcoded `/Data/...` paths in source (use `cfg.dataset.*`).
