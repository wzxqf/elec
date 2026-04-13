#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
CONFIG_PATH="${PROJECT_ROOT}/experiment_config.yaml"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "未找到配置文件: ${CONFIG_PATH}" >&2
  exit 1
fi

if ! command -v mamba >/dev/null 2>&1; then
  echo "未找到 mamba，请先安装并确保其在 PATH 中可用。" >&2
  exit 1
fi

VERSION="$(
  awk '
    $1=="project:" {in_project=1; next}
    in_project && $1=="version:" {print $2; exit}
    in_project && /^[^[:space:]]/ {exit}
  ' "${CONFIG_PATH}"
)"

if [[ -z "${VERSION}" ]]; then
  echo "无法从 ${CONFIG_PATH} 解析 project.version" >&2
  exit 1
fi

RUN_CMD=(
  mamba run -n elec_env
  python -m src.scripts.run_pipeline
)

echo "项目根目录: ${PROJECT_ROOT}"
echo "实验版本: ${VERSION}"
echo "结果目录: ${PROJECT_ROOT}/outputs/${VERSION}"
echo "执行命令: ${RUN_CMD[*]}"

if [[ "${1:-}" == "--dry-run" ]]; then
  exit 0
fi

if [[ $# -gt 0 ]]; then
  echo "不支持的参数: $*" >&2
  echo "用法: bash run_all.sh" >&2
  echo "调试: bash run_all.sh --dry-run" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
"${RUN_CMD[@]}"
