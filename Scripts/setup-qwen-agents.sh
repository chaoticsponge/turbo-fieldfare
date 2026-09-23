#!/usr/bin/env bash
set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd -- "$script_directory/.." && pwd)"
command -v uv >/dev/null || { echo 'Install uv, then rerun this script.' >&2; exit 1; }
export UV_CACHE_DIR="$project_directory/.build/qwen-agent-uv-cache"
agent_environment="$project_directory/.build/qwen-agent-venv"
if [[ ! -x "$agent_environment/bin/python" ]]; then
  uv venv --python 3.13 "$agent_environment"
fi
uv pip install --python "$agent_environment/bin/python" \
  -r "$script_directory/qwen-agent-requirements.txt"
"$agent_environment/bin/omlx" serve --help >/dev/null
echo 'Concurrent Qwen server dependencies installed. No model was downloaded or loaded.'
