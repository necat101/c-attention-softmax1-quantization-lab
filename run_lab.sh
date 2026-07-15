#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# resolve zig
if [ -n "${ZIG_BIN:-}" ] && [ -x "$ZIG_BIN" ]; then
  ZIG="$ZIG_BIN"
elif command -v zig >/dev/null 2>&1; then
  ZIG="$(command -v zig)"
elif [ -x "$HOME/.local/bin/zig" ]; then
  ZIG="$HOME/.local/bin/zig"
elif [ -x "$HOME/bin/zig" ]; then
  ZIG="$HOME/bin/zig"
else
  echo "zig not found – set ZIG_BIN or install zig 0.14+" >&2
  exit 1
fi

# resolve python
if [ -n "${PYTHON_BIN:-}" ] && [ -x "$PYTHON_BIN" ]; then
  PY="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "python not found" >&2
  exit 1
fi

export ZIG_BIN="$ZIG"
export PYTHON_BIN="$PY"

echo "zig: $($ZIG version 2>/dev/null || echo unknown)"
echo "python: $($PY --version 2>&1)"
echo

"$PY" run_lab.py
echo
echo "running tests..."
"$PY" -m unittest -v
