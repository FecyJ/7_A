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
WORKSPACE_DIR="${AGENT_WORKSPACE:-$PWD}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--workspace PATH|-w PATH] [--check] [--help] [--] [agent args...]

Options:
  -w, --workspace PATH  Set the workspace directory the agent tools operate in.
                        Defaults to AGENT_WORKSPACE, then the current directory.
      --check           Print resolved paths and exit.
  -h, --help            Show this help.

Environment:
  AGENT_WORKSPACE       Default workspace path.
  PYTHON_BIN            Python interpreter, default: \$REPO_ROOT/venv/bin/python.
EOF
}

AGENT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workspace)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "启动失败：$1 需要提供路径。" >&2
        exit 2
      fi
      WORKSPACE_DIR="$2"
      shift 2
      ;;
    --workspace=*)
      WORKSPACE_DIR="${1#*=}"
      shift
      ;;
    --check)
      AGENT_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        AGENT_ARGS+=("$1")
        shift
      done
      ;;
    *)
      AGENT_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "启动失败：未找到可执行 Python -> $PYTHON_BIN" >&2
  echo "可通过环境变量 PYTHON_BIN 指定解释器，例如：" >&2
  echo "  PYTHON_BIN=/path/to/python $0" >&2
  exit 1
fi

if [[ ! -d "$WORKSPACE_DIR" ]]; then
  echo "启动失败：工作区不存在或不是目录 -> $WORKSPACE_DIR" >&2
  exit 1
fi

WORKSPACE_DIR="$(cd -P "$WORKSPACE_DIR" >/dev/null 2>&1 && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  export DOTENV_PATH="$REPO_ROOT/.env"
fi

if [[ "${AGENT_ARGS[0]:-}" == "--check" ]]; then
  echo "REPO_ROOT=$REPO_ROOT"
  echo "WORKSPACE_DIR=$WORKSPACE_DIR"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "DOTENV_PATH=${DOTENV_PATH:-}"
  echo "STATUS=ok"
  exit 0
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORKSPACE_DIR"

exec "$PYTHON_BIN" -m src.main "${AGENT_ARGS[@]}"
