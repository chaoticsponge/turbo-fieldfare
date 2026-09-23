# Qwen3.8 27B 8-bit integration — September 22, 2026

The Mac app offers `mlx-community/Qwen3.8-27B-8bit` at revision
`815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9`. Its 17-file installation totals
29,531,519,120 bytes. Six weight-shard SHA-256 values come from the pinned
Hugging Face LFS metadata. All 11 non-weight assets were downloaded, verified
against repository metadata, and hashed. The safetensors index references
exactly the six cataloged shards. No weight shard was downloaded.

The configuration uses `qwen3_5`, 64 text layers, vocabulary size 248,320, and
affine 8-bit quantization with group size 64. The pinned Swift configuration
decoder, tokenizer, prompt continuation, and image processor accept these assets.

## Memory and scope

The package uses the existing native MLX/Metal quantized loading path and one
model owner in the sibling decode service. Switching unloads the previous model
first. Text prefill stays bounded and text continuations reuse the live cache.
The reusable allocation cache still defaults to 64 MiB.

New Chat and unload clear unused MLX allocations. Failed/cancelled loads also
clear unused allocations after the model factory's local references unwind.
The app rejects loading when the weight shards alone exceed Metal's recommended
working-set budget. Passing this lower-bound check does not guarantee that a
long context or image workload fits. No system memory limits are raised.

The 3-bit default and installation remain separate from the new 8-bit option.
This change provides model inference; autonomous browser, terminal, SSH, email,
and scheduled-task execution require a separate agent/tool integration.

## Environment

- Base commit: `4c6db1e698ea861609109d4bc517410ff302af46`, with existing uncommitted Qwen changes.
- Hardware: Mac15,6, Apple M3 Pro, 12 CPU cores, 18 GiB unified RAM.
- macOS: 26.1 (25B78).
- Swift: 6.2.4 (`swiftlang-6.2.4.1.4 clang-1700.6.4.2`).
- Available disk before validation: 109 GiB; `memory_pressure -Q`: 44% free.
- The documented model-process `pgrep` found no match (exit 1).

This is not the requested 64 GB Mac. No real-model inference or community
benchmark was attempted. Peak RAM, throughput, and end-to-end generation on
the target Mac remain unmeasured. GPU tests use tiny tensors, not model weights.

## Commands and results

Initial command:

```bash
QWEN_TOKENIZER_TEST_DIRECTORY="$PWD/.build/qwen38-8bit-metadata" \
  bash Scripts/test.sh --qwen > /tmp/qwen38-8bit-tests.log 2>&1
```

The sandboxed attempt exited 1 with compiler-cache errors:

```text
<unknown>:0: error: error opening '/Users/Sponge/.cache/clang/ModuleCache/Swift-3AJJN1UPT8RFK.swiftmodule' for output: /Users/Sponge/.cache/clang/ModuleCache: Operation not permitted
<unknown>:0: error: unable to load standard library for target 'arm64-apple-macosx14.0'
```

Repeating with compiler-cache access exited 1 because a cached header referenced
the checkout's previous location:

```text
error: PCH was compiled with module cache path '/Users/Sponge/turbo-fieldfare-1/.build/arm64-apple-macosx/release/ModuleCache/29UBYLX26W4T8', but the path is currently '/Users/Sponge/Downloads/turbo-fieldfare-1/.build/arm64-apple-macosx/release/ModuleCache/29UBYLX26W4T8'
1 error generated.
error: fatalError
```

Successful commands, preserving existing caches and using a new compiler cache:

```bash
bash Scripts/test.sh -c release --disable-xctest --disable-swift-testing \
  -Xcc -fmodules-cache-path="$PWD/.build/qwen64-module-cache" \
  -Xswiftc -module-cache-path -Xswiftc "$PWD/.build/qwen64-module-cache" \
  > /tmp/qwen38-8bit-build.log 2>&1
python3 Scripts/build_qwen_metal.py
QWEN_TOKENIZER_TEST_DIRECTORY="$PWD/.build/qwen38-8bit-metadata" \
  bash Scripts/test.sh --qwen-compiled --filter InstalledModelCatalogTests \
  --filter ConversationDocumentTests > /tmp/qwen38-8bit-validation.log 2>&1
```

All three commands exited **0**. Complete summary footers:

```text
Build complete! (275.97s)
MLX Metal library is up to date.
✔ Test run with 34 tests in 9 suites passed after 5.739 seconds.
```

The release app, sibling service, and test bundle were linked. Tests covered
catalogs, separate quantization receipts/downloads, picker selection,
weight-budget refusal, conversation documents, discovery, bounded prefill,
cancellation, sampling, and real tokenizer/image replay for the 8-bit package.
Opt-in full-model, external download, and unrelated text-model tokenizer suites
were disabled. Existing warnings in server, repacker, and unrelated tests were
not changed. `git diff --check` also exited 0.

Protocol notes: tests ran serially through `Scripts/test.sh`. Compilation and
the explicit test-entry stage were separated to work around relocated compiler
caches; execution was disabled only during the build stage. No model process
was launched or terminated, no checkpoint duplicated, and no cache purged.
The real-model preflight and community benchmark protocol were not used because
no real-model run was attempted.
