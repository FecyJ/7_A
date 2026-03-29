#!/usr/bin/env bash
set -euo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SCRIPT_SOURCE" ]; do
  SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" >/dev/null 2>&1 && pwd)"
  SCRIPT_SOURCE="$(readlink "$SCRIPT_SOURCE")"
  [[ "$SCRIPT_SOURCE" != /* ]] && SCRIPT_SOURCE="$SCRIPT_DIR/$SCRIPT_SOURCE"
done

REPO_ROOT="$(cd -P "$(dirname "$SCRIPT_SOURCE")" >/dev/null 2>&1 && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "启动失败：未找到可执行 Python -> $PYTHON_BIN" >&2
  echo "可通过环境变量 PYTHON_BIN 指定解释器，例如：" >&2
  echo "  PYTHON_BIN=/path/to/python $0" >&2
  exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
  echo "REPO_ROOT=$REPO_ROOT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "STATUS=ok"
  exit 0
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

exec "$PYTHON_BIN" -m src.main "$@"
