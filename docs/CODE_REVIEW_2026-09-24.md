# Runtime efficiency and security review — 2026-09-24

This review changes executable code and adds regression tests. It covers the
Python specialist fleet, expert streaming/cache/read-ahead paths, package
installer and launcher, native Swift HTTP ingress, and the tracked-symlink
repository check. Native inference, download, and application lifecycle paths
were inspected selectively. It is not an exhaustive audit of all 294 Swift
source files, a dependency vulnerability scan, or a security certification.

## Implemented changes

- Precompute each expert layer's tensor names and byte sizes once per model,
  rather than rebuilding them for every expert load and prefetch.
- Skip adaptive-cache counter aggregation between adjustment intervals.
  Cache decisions and health snapshots refresh at evaluated boundaries at most
  once every two seconds; tensor operations and precision are unchanged.
- Release decoded JSON and upload buffers before requests wait for admission;
  serialize forwarded requests as compact UTF-8 instead of escaped ASCII.
- Bound concurrent uploads as well as queued/active POSTs. Enforce upload
  deadlines, media type, body size, JSON complexity, unique keys, finite
  numbers, and valid Unicode before handing requests to the engine.
- Reject foreign Host/Origin and cross-site browser requests in both fleet
  routing and the native Swift server, before request-body handling. Binding
  loopback alone did not provide this browser boundary.
- Centralize regular-file opens: reject symlink and special-file substitution,
  reject multiply-linked write targets, and validate before truncating files.
  Make the server state directory private to its owner. Downloads require HTTPS
  through redirects as well as on the initial URL; package hashing remains.
- Replace shell interpolation in the Ruby symlink check with argument-vector
  process invocation. A regression fixture uses a repository path with spaces
  and a literal `$(touch PWNED)` filename and verifies no command executes.
- Extend the existing compiled Swift test entry point with `--compiled`, so
  non-Qwen suites can run without implicitly selecting every Qwen test.

The router still uses explicit rules, not a new LLM. The harness creates agents
and executes tools; the local service routes and schedules their inference.
The earlier role budgets, token-aware admission, adaptive caches, prefix reuse,
and bounded read-ahead remain runtime implementations with regression coverage.

## Validation

Base commit: `9fa39aefb33e95cf6320bd24c938a8cff1e2c736`, with this review's
working-tree changes. Hardware: Apple M3 Pro, Mac15,6, 12 CPU cores, 18 GiB RAM.
macOS 26.1; Apple Swift 6.2.4, swiftlang-6.2.4.1.4,
clang-1700.6.4.2; arm64-apple-macosx26.0. These are correctness tests, not
performance benchmarks or estimates of the target 64 GB Mac's throughput.

Exact final commands and complete test timing footers:

```bash
bash Scripts/test.sh --qwen-agents > /tmp/audit-tests.log 2>&1
```

Exit 0:

```text
----------------------------------------------------------------------
Ran 80 tests in 1.129s

OK
```

```bash
bash Scripts/test.sh --compiled --filter 'ServerLoopbackPolicyTests|HTTPServerTests|ServerIngressHardeningTests' > /tmp/audit-swift-runtime.log 2>&1
```

Exit 0:

```text
✔ Test run with 37 tests in 4 suites passed after 4.171 seconds.
```

```bash
bash Scripts/test.sh --expert-streaming > /tmp/audit-mlx-tests.log 2>&1
```

Exit 0:

```text
----------------------------------------------------------------------
Ran 11 tests in 0.855s

OK
```

The MLX suites use tiny synthetic weights, including Qwen/GLM prefill and cached
continuation equivalence, read-ahead, SSD prefix restoration, and separate
engine threads. Native HTTP tests use fake backends. No full checkpoint was
downloaded or loaded, and no existing application was terminated.

Build/runner deviations: the initial release test command
`bash Scripts/test.sh -c release --filter 'ServerLoopbackPolicyTests|HTTPServerTests|ServerIngressHardeningTests'`
exited 1 because the checkout had moved and its cached PCH referenced the old
module-cache directory. The diagnostic was:

```text
error: PCH was compiled with module cache path '/Users/Sponge/turbo-fieldfare-1/.build/arm64-apple-macosx/release/ModuleCache/29UBYLX26W4T8', but the path is currently '/Users/Sponge/Downloads/turbo-fieldfare-1/.build/arm64-apple-macosx/release/ModuleCache/29UBYLX26W4T8'
```

Retried without deleting caches:

```bash
bash Scripts/test.sh -c release -Xcc -fmodules-cache-path=/private/tmp/turbofieldfare-audit-module-cache -Xswiftc -module-cache-path -Xswiftc /private/tmp/turbofieldfare-audit-module-cache --filter 'ServerLoopbackPolicyTests|HTTPServerTests|ServerIngressHardeningTests' > /tmp/audit-swift-tests.log 2>&1
```

Exit 0, build completed in 325.86s, but the normal runner executed zero tests:

```text
Test Suite 'Selected tests' started at 2026-09-24 10:39:38.085.
Test Suite 'TurboFieldfarePackageTests.xctest' started at 2026-09-24 10:39:38.088.
Test Suite 'TurboFieldfarePackageTests.xctest' passed at 2026-09-24 10:39:38.088.
Executed 0 tests, with 0 failures (0 unexpected) in 0.000 (0.000) seconds
Test Suite 'Selected tests' passed at 2026-09-24 10:39:38.088.
Executed 0 tests, with 0 failures (0 unexpected) in 0.000 (0.003) seconds
```

The compiled entry point above then actually ran the selected Swift Testing
suites. Existing compiler warnings remain. Process inspection required leaving
the sandbox after its first attempt could not access sysmond; the successful
inspection found no model process. Before synthetic MLX tests, memory pressure
reported 51% free and disk reported 103 GiB available. The full Gemma-pack
preflight and community benchmark protocol do not apply to these synthetic,
model-free checks and were not run. Tests ran serially through `Scripts/test.sh`.

## Remaining limits

There is no measured full-fleet throughput or peak memory on the target Mac,
and no live Hermes/OpenClaw session was exercised here. Concurrent synthetic
engines demonstrate correctness, not a throughput gain. All models do not have
to stay resident together; admission may queue or evict according to budgets.

The API remains unauthenticated and loopback-only. Host/Origin checks are a
browser boundary, not authentication against other local applications. File
checks do not isolate this service from a malicious process running as the same
user that can rewrite its code or replace parent directories. Ingress limits
reduce transient allocations but do not cap the whole Python/Metal process.
Model quality, arbitrary tool-call reliability, and parity with frontier closed
models have not been established by this review.
