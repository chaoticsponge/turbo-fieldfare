# Defensive hardening validation

Validated September 25–26, 2026 on changes based on commit
`72f3991554a1989a209a86a3bf35505af9baede5`. Hardware: Apple M3 Pro,
Mac15,6, 12 CPU cores, 18 GiB unified memory; macOS 26.1 (25B78),
Swift 6.2.4. These are correctness checks, not throughput or peak-memory
measurements for the target 64 GB Mac.

## Implemented protections

- Both agent launch modes use the restricted router, including single-model
  automatic routing, token-aware admission, and Host/Origin checks.
- Image counts, encoded size, dimensions and aggregate pixels are checked
  before upstream token counting can decode images. Unsupported input fails
  closed. Transport connections and incomplete-header waiting are bounded.
- Repacking validates architecture dimensions, attention relationships,
  finite numeric values, quantization and wire-shape conversions before
  planning allocations. Configuration reads are bounded and reject symlinks.
- Conversation directories use mode 0700 and new persisted images use 0600.
  Existing image files are protected by the private directory; their file
  modes are not recursively rewritten.
- Swift Crypto is pinned to 4.5.1, the fixed version identified in
  [Apple's advisory](https://github.com/apple/swift-crypto/security/advisories/GHSA-8q93-f6xh-4f6f).
  No affected RSA extras call path was identified in application code.
- Python runtime dependencies use a 114-package lock with registry artifact
  hashes and fixed Git commits. Cache identity includes the lock contents.
  This is not a hermetic build: isolated build dependencies remain outside it.
- CI actions use fixed commits and checkout does not persist credentials.
  Swift CI requires a successful nonempty test footer. Agent defenses run in
  CI; synthetic MLX tests run only where Metal is available, with an explicit
  warning otherwise. Remote workflow execution has not been verified here.
- Security-reporting and release-check links now identify this fork. The
  repository owner must have GitHub private vulnerability reporting enabled;
  that repository setting was not verified or changed in this session.

## Commands and results

Run from the repository root, serially. Final executions below exited 0.

```bash
swift package resolve
bash Scripts/setup-qwen-agents.sh
bash Scripts/test.sh -c release -Xcc -fmodules-cache-path=/private/tmp/turbofieldfare-audit-module-cache -Xswiftc -module-cache-path -Xswiftc /private/tmp/turbofieldfare-audit-module-cache --disable-xctest --disable-swift-testing
bash Scripts/test.sh --compiled --filter 'ArchInfoValidationTests|PrivateDirectoryTests|ConversationImageWriterTests|ConversationStoreTests|SafetensorsHostileHeaderTests|RemotePayloadCopyTests|RemoteRangeTransferTests|RemoteDownloadSessionTests|GTurboFormatCompatibilityTests|GTurbo.*CodecTests|InstallLockTests|VisionPackMutationLockTests|ServerIngressHardeningTests|ServerLoopbackPolicyTests'
bash Scripts/test.sh --qwen-agents
bash Scripts/test.sh --expert-streaming
uv pip sync --python .build/qwen-agent-venv/bin/python --dry-run Scripts/pylock.qwen-agents.toml
```

Complete timing/result footers:

```text
Build complete! (68.94s)
✔ Test run with 137 tests in 19 suites passed after 3.931 seconds.

----------------------------------------------------------------------
Ran 88 tests in 0.932s

OK

----------------------------------------------------------------------
Ran 12 tests in 0.595s

OK

Checked 114 packages in 28ms
Would make no changes
```

The setup script initially exited 2 (`unexpected argument '-r' found`).
After correcting the lock-file argument, its final run exited 0:

```text
Checked 114 packages in 15ms
Concurrent Qwen server dependencies installed. No model was downloaded or loaded.
```

Additional checks passed: workflow YAML and lock TOML parsing, shell/Ruby
syntax, `git diff --check`, `ruby Scripts/check_tracked_symlinks.rb`
(`no tracked symlinks`) and `ruby Scripts/check_markdown_links.rb`
(`checked 33 Markdown files; all local links and anchors resolve`).

The lock check used uv 0.12.6 and Python 3.13.15. uv emitted its experimental
pylock-format warning. The lock was generated from the tested environment,
preserving its registry versions and five Git revisions. Regenerate it only
as part of a reviewed dependency update; do not replace it with an unpinned
installation from the top-level requirements file.

The 12 MLX tests include a real upstream ASGI application test with no lifespan,
network listener or model startup. Both launch configurations reject unsupported
endpoints and foreign hosts before model admission. Small generated tensors
exercise expert streaming, caches, read-ahead and two engine threads.

Protocol deviations: native tests used the compiled test entry point because
this toolchain's standard entry point previously reported success without
executing tests. Only the focused native suites above ran for this change.
Synthetic MLX tests used tiny generated fixtures instead of an installed Gemma
pack; no full model was downloaded, duplicated or loaded. Process checks found
no model process; memory-pressure free percentage was 43%, disk available 47 GiB.
Metal, package resolution and module-cache access required sandbox escalation.
Full-size multi-model inference, live Hermes/OpenClaw integration and target-Mac
performance remain unmeasured. Tests establish bounded behaviors, not a claim
that the application is free of vulnerabilities.
