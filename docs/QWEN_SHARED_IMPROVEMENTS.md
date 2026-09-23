# Shared Qwen improvements — September 23, 2026

The native MLX path now evaluates cache state together after each text prefill
chunk. Intermediate chunks do not evaluate their unused vocabulary logits;
the final chunk evaluates both logits and cache before sampling. History replay
uses the selected prefill chunk size (one token when prefill is disabled),
instead of a hardcoded 128-token chunk. Cancellation is checked after evaluation.
These changes apply to all six bundled Qwen variants. The Gemma runtime is unchanged.

The dedicated agent launcher now recognizes all six pinned installation receipts
and verifies their files before loading. `--list-models` lists the packages;
`--model` selects a completed installation. Optional `--prefix-cache-gb 4`
enables bounded SSD prefix reuse, with no hot RAM cache and four initial cache
blocks. Cache directories are separated by model revision. The default remains
cache-disabled. See [the server guide](QWEN_AGENT_SERVER.md) for launch commands.

## Validation

Base commit: `4c6db1e698ea861609109d4bc517410ff302af46`, plus existing uncommitted
work and these edits. Hardware: Mac15,6, M3 Pro, 12 CPU cores, 18 GiB unified RAM.
OS: macOS 26.1 (25B78). Compiler: Apple Swift 6.2.4
(`swiftlang-6.2.4.1.4 clang-1700.6.4.2`).

Exact commands, run from the checkout root:

```bash
bash Scripts/test.sh -c release --disable-xctest --disable-swift-testing -Xcc -fmodules-cache-path="$PWD/.build/qwen64-module-cache" -Xswiftc -module-cache-path -Xswiftc "$PWD/.build/qwen64-module-cache" > /tmp/qwen-shared-improvements-build.log 2>&1
QWEN_TOKENIZER_TEST_DIRECTORY="$PWD/.build/qwen38-8bit-metadata" bash Scripts/test.sh --qwen-compiled --filter InstalledModelCatalogTests --filter ConversationDocumentTests > /tmp/qwen-shared-improvements-tests.log 2>&1
bash Scripts/test.sh --qwen-agents
python3 Scripts/serve-qwen-agents.py --list-models
git diff --check
```

All commands exited 0. Complete timing/result footers:

```text
Build complete! (83.00s)
✔ Test run with 35 tests in 9 suites passed after 6.310 seconds.

----------------------------------------------------------------------
Ran 10 tests in 0.025s

OK
```

The new recurrent toy-model test compares chunked and full-prompt final logits,
checks retained state, and checks continuation logits. Server tests recognize
every bundled receipt, refuse modified receipts, and validate optional SSD cache
settings through the installed oMLX parser. Existing cancellation, token framing,
history, installation, and launcher checks also passed.

Validation limits/protocol deviations: these are model-free checks, not community
benchmarks. Installed-model/download and separate text-tokenizer asset suites
remain opt-in and were skipped. The tokenizer tests used already downloaded
8-bit package metadata; no weights were downloaded or model processes launched.
The compiled-test runner and separate module-cache path avoid existing local
Swift Testing/module-cache issues; no caches were purged. Existing compiler
warnings remain. Compiler-cache access required sandbox escalation.

Full-model numerical equivalence, concurrent live HTTP/tool calls, throughput,
SSD-cache hit rates, and peak memory on the 64 GB target remain unmeasured.
The memory admission check and runtime guard do not guarantee that every context
length and concurrency combination fits. No speedup or RAM reduction is claimed
as a measurement from these tests.
