#!/usr/bin/env bash
set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory/.."
swift build -c release
python3 Scripts/build_qwen_metal.py
