#!/usr/bin/env bash
# Push ~/.ssh/id_ed25519.pub to every reachable DSI machine.
# Machines not reachable directly are retried via ProxyJump through bugatti.
# Requires: sshpass (sudo apt-get install sshpass), nc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTS_FILE="${SCRIPT_DIR}/hosts_all.txt"
REACHABLE_FILE="${SCRIPT_DIR}/hosts_reachable.txt"
LOG_FILE="${SCRIPT_DIR}/ssh_setup.log"
KEY="${HOME}/.ssh/id_ed25519.pub"
JUMP="bugatti"

if [[ ! -f "$HOSTS_FILE" ]]; then
    echo "hosts_all.txt not found — run: uv run python scripts/generate_hosts.py first"
    exit 1
fi

if ! command -v sshpass &>/dev/null; then
    echo "sshpass is required: sudo apt-get install sshpass"
    exit 1
fi

if [[ ! -f "$KEY" ]]; then
    echo "SSH key not found at $KEY"
    exit 1
fi

read -r -s -p "Polytechnique password: " PASS
echo

> "$REACHABLE_FILE"
> "$LOG_FILE"

ok=0; fail=0; truly_unreachable=0

copy_key() {
    local host="$1"
    local use_jump="${2:-}"
    local opts=(-o StrictHostKeyChecking=no -o ConnectTimeout=10)
    [[ -n "$use_jump" ]] && opts+=(-o "ProxyJump=$JUMP")
    sshpass -p "$PASS" ssh-copy-id "${opts[@]}" -i "$KEY" "$host" &>/dev/null
}

while IFS= read -r line; do
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

    host="$line"
    hostname="${host##*@}"

    if nc -zw2 "$hostname" 22 &>/dev/null; then
        if copy_key "$host"; then
            echo "OK          $host"
            echo "OK $host" >> "$LOG_FILE"
            echo "$host" >> "$REACHABLE_FILE"
            ((++ok))
        else
            echo "COPY_FAIL   $host"
            echo "COPY_FAIL $host" >> "$LOG_FILE"
            ((++fail))
        fi
    else
        if copy_key "$host" jump; then
            echo "OK (jump)   $host"
            echo "OK_JUMP $host" >> "$LOG_FILE"
            echo "$host" >> "$REACHABLE_FILE"
            ((++ok))
        else
            echo "UNREACHABLE $host"
            echo "UNREACHABLE $host" >> "$LOG_FILE"
            ((++truly_unreachable))
        fi
    fi
done < "$HOSTS_FILE"

total=$((ok + fail + truly_unreachable))
echo ""
echo "Done: $ok/$total keys copied | $fail copy failures | $truly_unreachable unreachable"
echo "Reachable hosts saved to: $REACHABLE_FILE"
