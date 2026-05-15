#!/usr/bin/env bash
# PreToolUse: block reading secret/sensitive files

FILE=$(echo "$CLAUDE_TOOL_INPUT" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d.get('file_path',''))" 2>/dev/null)

BASENAME=$(basename "$FILE")

# Block .env files
if echo "$BASENAME" | grep -qE '^\.env'; then
  echo "BLOCKED: reading .env files is not allowed." >&2
  exit 1
fi

# Block settings.local.json (may contain secrets/tokens)
if [ "$BASENAME" = "settings.local.json" ]; then
  echo "BLOCKED: reading settings.local.json is not allowed (may contain secrets)." >&2
  exit 1
fi

exit 0
