#!/usr/bin/env bash
# Stop processes started by scripts/run_local.sh
cd "$(dirname "$0")/.."
RUNTIME="${RUNTIME:-.runtime}"
for f in "$RUNTIME"/*.pid; do
  [[ -f "$f" ]] || continue
  pid=$(cat "$f")
  if kill "$pid" 2>/dev/null; then echo "stopped $(basename "$f" .pid) ($pid)"; fi
  rm -f "$f"
done
