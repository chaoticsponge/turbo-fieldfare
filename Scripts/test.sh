#!/usr/bin/env bash
# Serial test runner. Shared Metal state makes in-process parallel tests
# unreliable. Pass any extra arguments through, for example --filter.

if [[ "${1:-}" == "--package-path" ]]; then
  shift 2
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory/.."
if [[ "${1:-}" == "--expert-streaming" ]]; then
  shift
  agent_python="$script_directory/../.build/qwen-agent-venv/bin/python"
  if [[ ! -x "$agent_python" ]]; then
    echo 'Run Scripts/setup-qwen-agents.sh before the synthetic MLX streaming tests.' >&2
    exit 1
  fi
  export PYTHONPATH="$script_directory${PYTHONPATH:+:$PYTHONPATH}"
  exec "$agent_python" -m unittest discover -s Tests/ExpertStreaming -v "$@"
fi
if [[ "${1:-}" == "--qwen-agents" ]]; then
  shift
  agent_python="$script_directory/../.build/qwen-agent-venv/bin/python"
  if [[ ! -x "$agent_python" ]]; then agent_python=python3; fi
  export PYTHONPATH="$script_directory${PYTHONPATH:+:$PYTHONPATH}"
  exec "$agent_python" -m unittest discover -s Tests/AgentServer -v "$@"
fi
if [[ "${1:-}" == "--qwen" ]]; then
  shift
  set -euo pipefail
  # `swift build --build-tests` omits testable imports in release builds.
  # Use the test build pipeline with execution disabled, then our entry point.
  swift test --no-parallel -c release --disable-xctest --disable-swift-testing
  python3 Scripts/build_qwen_metal.py
  exec bash "$script_directory/test.sh" --qwen-compiled "$@"
fi
# Explicit entry point for diagnosing toolchains whose `swift test` exits
# without invoking Swift Testing. Build tests first; this runs no model weights.
if [[ "${1:-}" == "--qwen-compiled" ]]; then
  shift
  set -euo pipefail
  test_bin="$(swift build -c release --show-bin-path)"
  test_bundle="$test_bin/TurboFieldfarePackageTests.xctest/Contents/MacOS"
  cp "$test_bin/mlx.metallib" "$test_bundle/mlx.metallib"
  test_platform="$(xcrun --sdk macosx --show-sdk-platform-path)"
  export DYLD_FRAMEWORK_PATH="$test_platform/Developer/Library/Frameworks:$test_platform/Developer/Library/PrivateFrameworks${DYLD_FRAMEWORK_PATH:+:$DYLD_FRAMEWORK_PATH}"
  export DYLD_LIBRARY_PATH="$test_platform/Developer/usr/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  swiftc -parse-as-library Scripts/QwenTestRunner.swift \
    -F "$test_platform/Developer/Library/Frameworks" \
    -o "$test_bin/QwenTestRunner"
  export QWEN_TEST_BUNDLE="$test_bundle/TurboFieldfarePackageTests"
  exec "$test_bin/QwenTestRunner" --no-parallel --filter Qwen "$@"
fi
exec swift test --no-parallel "$@"
