#!/usr/bin/env bash
# PreToolUse: block dangerous Bash commands before execution

CMD=$(echo "$CLAUDE_TOOL_INPUT" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d.get('command',''))" 2>/dev/null)

# Block rm -rf outside src/outputs/
if echo "$CMD" | grep -qE 'rm\s+-[^ ]*r[^ ]*f|rm\s+-[^ ]*f[^ ]*r'; then
  if ! echo "$CMD" | grep -q 'src/outputs'; then
    echo "BLOCKED: rm -rf is only allowed on src/outputs/. Use 'rm -rf src/outputs/' explicitly." >&2
    exit 1
  fi
fi

# Block force push (belt-and-suspenders over the denylist)
if echo "$CMD" | grep -qE 'git\s+push\s+.*--force|git\s+push\s+.*-f\b'; then
  echo "BLOCKED: git push --force is not allowed." >&2
  exit 1
fi

# Warn on git reset --hard
if echo "$CMD" | grep -qE 'git\s+reset\s+--hard'; then
  echo "BLOCKED: git reset --hard requires explicit user confirmation. Run it yourself in the terminal." >&2
  exit 1
fi

# Block hardcoded /Data/ paths as CLI arguments to training scripts
if echo "$CMD" | grep -qE 'python\s+src/' && echo "$CMD" | grep -q '/Data/'; then
  echo "BLOCKED: hardcoded /Data/ path detected in training command. Use Hydra config (cfg.dataset.*) only." >&2
  exit 1
fi

# Block git checkout -- (working tree wipe)
if echo "$CMD" | grep -qE 'git\s+checkout\s+--\s'; then
  echo "BLOCKED: git checkout -- (destructive) requires explicit user confirmation." >&2
  exit 1
fi

# Block git clean -f or -fd
if echo "$CMD" | grep -qE 'git\s+clean\s+.*-[^ ]*f'; then
  echo "BLOCKED: git clean -f requires explicit user confirmation." >&2
  exit 1
fi

# Block git restore . (restores entire working tree)
if echo "$CMD" | grep -qE 'git\s+restore\s+\.'; then
  echo "BLOCKED: git restore . requires explicit user confirmation." >&2
  exit 1
fi

exit 0
