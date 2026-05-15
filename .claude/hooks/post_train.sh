#!/usr/bin/env bash
# PostToolUse: capture training output to .claude/results/ when train.py runs
TOOL_NAME=$(echo "$CLAUDE_TOOL_NAME" 2>/dev/null)
INPUT=$(echo "$CLAUDE_TOOL_INPUT" 2>/dev/null)

# Only act on python src/train.py calls
echo "$INPUT" | grep -q "src/train.py" || exit 0

RESULTS_DIR="$(dirname "$0")/../../.claude/results"
mkdir -p "$RESULTS_DIR"
DATE=$(date +%Y%m%d-%H%M)
OUT_FILE="$RESULTS_DIR/${DATE}-train.md"

# Write a header; training output itself is captured by the calling session
cat > "$OUT_FILE" <<EOF
# Training run — $DATE

Command: $(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('command',''))" 2>/dev/null)

_Output captured from session — see below_
EOF

exit 0
