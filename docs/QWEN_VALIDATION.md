# Qwen app validation

This is a source conversion of the Mac app and its sibling decode service.
It is not a measured claim that a dense 27B model uses Gemma's former memory
budget. Initial validation used metadata only; the September 15 diagnostic below
loaded the subsequently installed checkpoint. Completed live generation, answer
quality, sustained memory pressure, and end-to-end UI parity remain unverified.

## Environment

- Base commit: `4c6db1e698ea861609109d4bc517410ff302af46`, with uncommitted Qwen changes.
- MacBook Pro Mac15,6, M3 Pro, 12 CPU cores, 18 GB unified memory.
- macOS 26.1 (25B78).
- Apple Swift 6.2.4 (`swiftlang-6.2.4.1.4 clang-1700.6.4.2`).
- Xcode macOS 26.2 SDK; Metal Toolchain 17C7003j.
- Initial disk availability: 172 GiB; `memory_pressure -Q`: 33% free.
- Work and validation: September 10–11, 2026.

## Build

```bash
bash Scripts/build-qwen.sh
```

The script builds the executables and compiles the pinned MLX shaders. A direct
`swift build -c release` previously completed with exit 0 and footer
`Build complete! (36.37s)`. The shader compiler built all nine source files;
its final output was `Built /Users/Sponge/turbo-fieldfare-1/.build/arm64-apple-macosx/release/mlx.metallib`.
Four upstream shader warnings concerned C++17 `constexpr if` extensions.
Existing Swift warnings in server/repacker code remain.

## Model-free tests

The focused suites cover pinned file checksums, incomplete installations,
prompt continuation, bounded text prefill, cancellation, the real chat template,
and prepared-image persistence/replay. Tokenizer tests use only pinned metadata
in `.build/qwen-tokenizer-check`, with no weight shards.

Build and execute through the serial runner's explicit Qwen entry point:

```bash
QWEN_TOKENIZER_TEST_DIRECTORY="$PWD/.build/qwen-tokenizer-check" \
  bash Scripts/test.sh --qwen
```

`--qwen` builds the release test bundle and `mlx.metallib`, then calls
`--qwen-compiled`. That places the library beside the bundle, loads the matching Xcode
Testing framework, and invokes Swift Testing explicitly with serial execution.
Without the metadata environment variable, two tokenizer/image tests skip.

Final validation command (exit **0**):

```bash
QWEN_TOKENIZER_TEST_DIRECTORY="$PWD/.build/qwen-tokenizer-check" \
  bash Scripts/test.sh --qwen --filter ConversationDocumentTests
```

Complete final build/shader/test summary:

```text
Build complete! (133.71s)
MLX Metal library is up to date.
✔ Test run with 17 tests in 4 suites passed after 2.657 seconds.
```

This includes nine Qwen tests and eight existing conversation-document tests.
The release Mac app and sibling decode service were linked in that build.
Shell-script syntax checks, Python-script compilation, and `git diff --check`
also completed with exit 0. No real-model timing footer exists because no
model inference was run.

## Diagnostic attempts and limitations

- Ordinary `Scripts/test.sh -c release --filter Qwen` exited 0 while reporting
  zero XCTest tests and no Swift Testing run. Adding `--enable-swift-testing`
  and `--disable-xctest` still produced no test report. These are **not passes**.
- The corrected test-bundle build completed with exit 0 and footer
  `Build complete! (149.42s)`.
- Direct toolchain-helper diagnosis initially exited 133: `Library not loaded:
  @rpath/Testing.framework/Versions/A/Testing`. Providing framework paths fixed
  loading but the helper still returned without tests. The explicit entry point
  runs the compiled suites. Its approach follows Swift's documented
  [Darwin test-bundle loading and framework requirements](https://github.com/swiftlang/swift-testing/blob/main/Documentation/CommandlineDebugging.md).
- A test compile error from a throwing expression nested inside `#require`
  was fixed by evaluating the expression before the macro.
- The first real nine-test run failed two metadata tests; it is not counted as
  successful validation. The thinking fixture was corrected to use the pinned
  template's `reasoning_content` field. The image failure revealed that the
  pinned MLX `prompt:images:` initializer leaves its flattened image array empty.
  Production now uses `UserInput(chat:processing:)`, and verifies that every
  requested image produced a prepared frame. Both tests subsequently passed.
- The initial one-command script used `swift build --build-tests` in release
  mode and exited 1 with `module 'TurboFieldfareFormat' was not compiled for
  testing`. It now uses the Swift test-build pipeline with execution disabled,
  followed by the explicit Testing entry point.
- The initially missing Metal Toolchain was installed with
  `xcodebuild -downloadComponent MetalToolchain` (exit 0). An unsupported shader
  language spelling was corrected to `-std=metal3.2`.
- Tests were launched through `Scripts/test.sh`. A debugger also launched the
  toolchain helper solely to investigate its early exit; only those diagnostic
  processes were stopped. No existing app or model process was terminated.
- No model benchmark protocol was run. The original Gemma-pack precondition
  does not validate the replacement Qwen installation; a real Qwen run requires
  its completed checksum-verified receipt plus the OS, disk, pressure, and
  single-process checks in `AGENTS.md`.

The selected download is a mixed 3-bit MLX encoding, not the requested BF16
encoding. It totals 12,729,681,276 bytes. This size alone does not establish
runtime suitability on an 18 GB machine, especially during image prefill.

## Download timeout fix — September 14, 2026

The app reported `NSURLErrorDomain -1001` on the pinned `chat_template.jinja`.
An isolated Foundation URLSession download reproduced the same timeout
(`_kCFStreamErrorCodeKey=-2102`). The system curl fetched that same 8,950-byte
file with HTTP 200 and the catalog's SHA-256, including with user curl
configuration disabled. This establishes a working alternative transport;
the underlying CFNetwork/network interaction was not diagnosed further.

The Mac installer now uses `/usr/bin/curl` with HTTPS-only redirects, normal
certificate verification, a 20-second connection timeout, a 60-second stall
timeout, and three retries for transient failures. Files remain on disk and
must pass the existing pinned checksum before installation. Cancellation waits
for this installer-owned process to stop before cleaning its temporary output
and releasing the lock. The system transport's timeout/retry behavior is
documented in the [curl manual](https://curl.se/docs/manpage.html).

Validation command, exit **0**, on the same macOS 26.1 / Swift 6.2.4 installation:

```bash
QWEN_DOWNLOAD_SMOKE_TEST=1 \
QWEN_TOKENIZER_TEST_DIRECTORY="$PWD/.build/qwen-tokenizer-check" \
  bash Scripts/test.sh --qwen
```

```text
Build complete! (101.15s)
MLX Metal library is up to date.
✔ Test pinnedFirstFileDownloadsAndPassesItsChecksum() passed after 1.337 seconds.
✔ Test run with 12 tests in 4 suites passed after 6.939 seconds.
```

The new tests cover HTTP error/partial-response refusal, temporary-output
cleanup, cancellation, and the live pinned metadata download. The Mac app and
decode service were rebuilt. `git diff --check` passed. Disk availability was
156 GiB. The existing app process and `scratch/qwen3.8-27b.mlx.partial` were left
untouched; the user must quit and reopen the app to use the rebuilt downloader.
No model weights were downloaded, so full-checkpoint transfer and inference
remain unmeasured. The live download test is opt-in through
`QWEN_DOWNLOAD_SMOKE_TEST=1`; it downloads only the first metadata file.

## First-token crash — September 15, 2026

Two decode-service crash reports (15:04:19 and 15:05:01) recorded `SIGTRAP` in
`TokenRing.append` / `mlx_where`, called by `TokenIterator.convertToToken`.
The service exited while sampling its first token, so the UI received
`unexpectedEOF`. These reports show an MLX assertion, not an OS memory kill.

The Qwen adapter supplied `[1, tokenCount]` to both the VLM and MLX's repetition
processor. The latter expects a flat token list: it counted only one prompt
token and built a ring buffer incompatible with its update mask. `QwenSampling`
now flattens the prompt for the penalty processor while preserving the model's
batched input and keeping repetition penalties enabled.

A model-free regression exercises the complete iterator through two sampled
tokens with penalties 1.0 and 1.1. Initial validation:

```bash
QWEN_TOKENIZER_TEST_DIRECTORY="$PWD/.build/qwen-tokenizer-check" \
  bash Scripts/test.sh --qwen
```

```text
Build complete! (74.23s)
✔ Test run with 13 tests in 4 suites passed after 5.219 seconds.
```

Exit 0; the opt-in live download test was skipped. A separate installed-model
test is opt-in via `QWEN_MODEL_SMOKE_TEST=1` and uses the actual IPC client and
sibling decode service for a greeting and continuation. Run it only after the
single-model preflight. It loads the existing Qwen installation with full
SHA-256 verification, uses 4K context and a 32-token cap per turn, and unloads
and shuts down only its own service afterwards.

### Live diagnostic and residency change

On September 15, the single-model preflight passed: 135 GiB disk available,
63% free from `memory_pressure -Q`, no existing model process, and all pinned
Qwen files and receipt present. Hardware, OS, Swift, and base commit were as
listed above. The replacement Qwen installation was used instead of the original
AGENTS.md Gemma pack; this was a diagnostic, not a community benchmark.

```bash
QWEN_MODEL_SMOKE_TEST=1 \
QWEN_TOKENIZER_TEST_DIRECTORY="$PWD/.build/qwen-tokenizer-check" \
  bash Scripts/test.sh --qwen-compiled > /tmp/qwen-hello-smoke.log 2>&1
```

That initial test used a 128-token cap. The model loaded and passed the previous
assertion site, but no turn completed within about eight minutes. A one-second
`sample 10758 1 10 -file /tmp/qwen-service-wait.txt` diagnostic found generation
waiting in MLX GPU evaluation, with a 12.2 GB footprint and 12.5 GB peak. This
stack sampling deviates from the no-profiling benchmark protocol; no throughput
measurement or benchmark claim is made.

Only this test's launchd service was stopped with `launchctl bootout`. Its forced
disconnect caused the following failure, not a new spontaneous assertion:

```text
unexpectedEOF
Test helloAndContinuationFinishWithoutLosingTheRuntime() failed after 507.636 seconds.
Test run with 14 tests in 5 suites failed after 512.480 seconds with 1 issue.
```

The command exited 1. No user-owned process was terminated. The smoke test now
logs load/token progress and caps each turn at 32 tokens.

Generation now requests MLX wired residency up to the GPU's recommended working
set and releases the ticket after completion or cancellation. This does not
change system kernel limits. Paging is a possible cause of the long wait, not
an established diagnosis; the residency change has no measured speedup yet.
Thinking remains enabled with low reasoning effort at the user's request.

Final release build and model-free validation, both exit 0:

```bash
bash Scripts/test.sh -c release --disable-xctest --disable-swift-testing
QWEN_TOKENIZER_TEST_DIRECTORY="$PWD/.build/qwen-tokenizer-check" \
  bash Scripts/test.sh --qwen-compiled
```

```text
Build complete! (81.67s)
✔ Test run with 14 tests in 5 suites passed after 5.480 seconds.
```

The installed-model suite and live download test were disabled in this final
run. A sandboxed attempt could not write the Swift module cache; the successful
run had compiler-cache access. Further live validation was deferred because the
user's app and decode service were running; the single-model rule prohibits a
second inference process. Quit and reopen the app to load the rebuilt service.
