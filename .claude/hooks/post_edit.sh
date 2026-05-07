#!/usr/bin/env bash
# PostToolUse: auto-format and lint Python files after Write|Edit
FILE=$(echo "$CLAUDE_TOOL_INPUT" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d.get('file_path', d.get('new_file_path','')))" 2>/dev/null)
[[ "$FILE" == *.py ]] || exit 0
cd "$(dirname "$0")/../.." || exit 0
uv run ruff format "$FILE" 2>/dev/null
uv run ruff check --fix "$FILE" 2>/dev/null
exit 0
