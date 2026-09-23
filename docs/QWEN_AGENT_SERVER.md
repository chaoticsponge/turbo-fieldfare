# Concurrent Qwen server for Hermes and OpenClaw

For multiple specialist models and automatic prompt routing, see
[the local agent fleet guide](LOCAL_AGENT_FLEET.md). This page describes the
original single-model mode, which remains available.

The native Swift app still generates one response at a time. Its existing
`TurboFieldfareServer` command serves Gemma with a serial generation queue.
For concurrent Qwen agents, use the dedicated mode below instead of the app.

This mode uses [oMLX v0.6.4](https://github.com/jundot/omlx/tree/1d7826185c5b5b69b38b27cbe57d7597b7551fd7),
an MLX/Metal server with continuous batching and an OpenAI-compatible API.
It reads any of the app's six verified Qwen installations directly. It does not duplicate
the checkpoint, load one model per subagent, or alter the native Swift runtime.
Hermes/OpenClaw create the agents and execute tools; the server only generates
responses and structured tool calls. Their normal tool permissions still apply.

## Start on the 64 GB Mac

Download **Qwen3.8 27B MLX 8-bit (64 GB Mac)** using the app, then quit the app.
Do not start this mode alongside another model process. It stays on
`127.0.0.1`; do not proxy, tunnel, or expose it.

From the checkout:

```bash
bash Scripts/setup-qwen-agents.sh
python3 Scripts/serve-qwen-agents.py --check
python3 Scripts/serve-qwen-agents.py
```

Setup needs `uv` and installs pinned oMLX in `.build/qwen-agent-venv`, using
Python 3.13. It does not download weights or start inference. The launcher
requires Apple Silicon, macOS 26+, Swift 6.2+, at least 2 GiB of free disk,
at least 15% free in `memory_pressure -Q`, no existing model process, enough
RAM, and the completed checksum-verified Qwen installation.

Default model path: `scratch/qwen3.8-27b-8bit.mlx`. For an installation elsewhere:

```bash
python3 Scripts/serve-qwen-agents.py --model /absolute/path/qwen3.8-27b-8bit.mlx
```

To select another installed Qwen package, pass that package's directory. List
the supported repositories and pinned revisions without loading weights:

```bash
python3 Scripts/serve-qwen-agents.py --list-models
python3 Scripts/serve-qwen-agents.py --model scratch/qwen3.5-4b-4bit.mlx
```

Supported packages are Qwen3.5 4B/9B 4-bit, Qwen3 14B/32B 4-bit, and
Qwen3.8 27B 3-bit/8-bit. The receipt must exactly match a bundled catalog;
arbitrary checkpoints and Gemma `.gturbo` packs are not accepted by this mode.
The same shared-weight batching engine and memory checks apply to each package.
Text-only models must be declared with `input: ["text"]` in OpenClaw, and
cannot process screenshot/image tasks. Use the selected folder's model ID in
both parent and child agent configuration.

Verification reads all files with a bounded 1 MiB hashing buffer before launch.
Keep the terminal open while serving. Ctrl-C stops only the server owned by
this launcher. Stop it before reopening the native app. This is a separate API
mode; the app does not connect to its live engine.

## Concurrency and memory

- Two simultaneous generations by default; extra work is scheduled by oMLX.
- One shared model, with independent request KV/recurrent states.
- 65,536 tokens maximum context per request, including the conversation budget.
  Hermes currently documents a 64K minimum for custom endpoints.
- Chunked prefill and decode fairness are enabled.
- On a 64 GiB Mac, the memory-guard ceiling is 48 GiB, with soft/hard watermarks
  at 85%/95%. This is an enforcement policy, not a measured peak or hard OS limit.
- No additional oMLX hot/SSD prefix cache by default. Active inference still
  needs KV state; optional SSD prefix reuse is described below.
- Only the selected model directory is scanned. Hub discovery and implicit
  downloads are disabled. Each run uses fresh settings under
  `.build/qwen-agent-server`, avoiding existing global oMLX settings.

Concurrency shares GPU time and bandwidth; it does not promise twice the speed.
Long contexts and images can cause the memory guard to defer or abort a request.
Do not disable the guard to force a workload to fit. Start with two, measure
your real agent tasks, then consider increasing concurrency:

```bash
python3 Scripts/serve-qwen-agents.py --concurrency 3 --context 65536
```

The launcher allows 1–4 active requests and 4K–64K contexts. Lower contexts can
save memory for other clients, but must match that client's actual requirements.
The old app's 64 MiB allocation-cache setting does not govern this separate engine.

### Reuse repeated agent prompts

Tool loops repeatedly send earlier messages and tool definitions. Enable bounded
SSD prefix caching to let oMLX reuse matching prior prompt state:

```bash
python3 Scripts/serve-qwen-agents.py --prefix-cache-gb 4
```

Choose 2, 4, 8, or 16 GiB, or 0 (default) to disable this cache. The cache lives
under `.build/qwen-agent-server/prefix-cache/<model-revision>` and can survive
server restarts. Each checkpoint has a separate directory. There is no extra
hot RAM cache, and initial cache-block allocation is limited to four blocks.
The disk preflight includes the requested cache size plus 2 GiB for state/logs.
Cached state represents prior prompts; it stays local like the conversation
history. This can save prefill work for matching prefixes, but introduces SSD
I/O and some cache-management memory. Its speed and memory tradeoffs have not
been measured here. It cannot shrink the model weights or the active KV state.

## Connect clients

Check discovery without generating a response:

```bash
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
```

With the default folder name, use:

| Setting | Value |
| --- | --- |
| API base URL | `http://127.0.0.1:8080/v1` |
| Protocol | OpenAI Chat Completions |
| Model ID | `qwen3.8-27b-8bit.mlx` |
| API key placeholder | `local` |
| Context window | `65536` |
| Output budget | `4096` |

The model ID is the installation folder name. Confirm it using `/v1/models`
when using a custom path. The local key is a client placeholder, not remote auth.

### Hermes

Run `hermes model`, choose **Custom endpoint**, and enter the values above.
The interactive setup records the endpoint, model, key, and context length.
Then merge this into the existing Hermes configuration:

```yaml
delegation:
  max_concurrent_children: 2
  max_spawn_depth: 1
```

Children inherit the parent's model unless overridden. Two child agents plus
the parent can exceed two outstanding calls; the server's active-generation
limit still controls GPU work. See the official
[local endpoint setup](https://hermes-agent.nousresearch.com/docs/reference/faq)
and [delegation guide](https://hermes-agent.nousresearch.com/docs/guides/delegation-patterns).

### OpenClaw

Merge [the example provider and subagent settings](examples/openclaw-qwen-agents.json)
into your existing OpenClaw configuration. Preserve your current tool permissions,
workspaces, accounts, and channels. The example sets the same local model for
parent and children and caps subagent concurrency at two. It does not install
OpenClaw or connect accounts. See the official
[local model guide](https://docs.openclaw.ai/gateway/local-models) and
[subagent guide](https://docs.openclaw.ai/tools/subagents).

## Validation and limitations — September 23, 2026

The dedicated environment was installed successfully on the development Mac:
oMLX 0.6.4, MLX 0.32.0, MLX-LM 0.31.3, Python 3.13.15. The release CLI's
`serve --help` completed without loading a model. Configuration validation uses
the installed engine's own settings parser, not a mock.

Environment: base commit `4c6db1e698ea861609109d4bc517410ff302af46` plus existing
uncommitted work; Mac15,6 / M3 Pro / 12 CPU cores / 18 GiB RAM;
macOS 26.1 (25B78); Apple Swift 6.2.4
(`swiftlang-6.2.4.1.4 clang-1700.6.4.2`). Free disk was 109 GiB and
`memory_pressure -Q` reported 40% free.

```bash
bash Scripts/setup-qwen-agents.sh > /tmp/qwen-agent-setup.log 2>&1
bash Scripts/test.sh --qwen-agents
bash -n Scripts/setup-qwen-agents.sh Scripts/test.sh
git diff --check
```

All exited 0 after correcting a test's macOS `/var` versus `/private/var`
canonical-path comparison. Initial tests exited 1 with that assertion failure.
The successful model-free test footer was:

```text
Ran 8 tests in 0.017s

OK
```

Coverage includes tampered/missing weights, foreign receipts, unexpected model
files, symlink refusal, low-memory refusal, existing-process refusal, local-only
launch arguments, and real upstream settings validation.

`python3 Scripts/serve-qwen-agents.py --check` initially exited 1 because the
system Python returned an empty macOS version string. Detection now uses
`sw_vers`. Rechecking exited 1 with the intended refusal:

```text
error: Insufficient unified RAM for this 8-bit model plus concurrent sessions.
```

No model was loaded or downloaded, and no existing process was stopped. Tests
ran through the repository's serial runner. No Swift rebuild was needed for
these separate scripts. No community benchmark protocol was run; live HTTP
generation, overlapping streams, image/tool loops, agent-client behavior, peak
memory, and throughput remain unverified on the 64 GB target Mac. The existing
Gemma-pack model-run preflight was not exercised because this validation never
started a model. Package installation required network/compiler-cache access;
it stayed in an isolated environment and did not alter the Swift dependencies.
