# Expert-streaming validation — September 23, 2026

## Bounded read-ahead — September 24

Base commit `acb1930` plus read-ahead changes. Hardware/environment remains
Mac15,6 / M3 Pro / 12 CPU cores / 18 GiB RAM, macOS 26.1 (25B78), Swift 6.2.4
(`swiftlang-6.2.4.1.4 clang-1700.6.4.2`), Python 3.13.15, oMLX 0.6.4,
MLX 0.32.0, MLX-LM 0.31.3.

Exact commands, all exit 0:

```bash
bash Scripts/test.sh --qwen-agents > /tmp/read-ahead-tests.log 2>&1
bash Scripts/test.sh --expert-streaming > /tmp/read-ahead-mlx-tests.log 2>&1
.build/qwen-agent-venv/bin/python Scripts/agent_runtime.py /tmp/read-ahead-routes.json serve --help > /tmp/read-ahead-bootstrap.log 2>&1
git diff --check
```

Complete final test footers:

```text
----------------------------------------------------------------------
Ran 72 tests in 0.833s

OK
```

```text
----------------------------------------------------------------------
Ran 11 tests in 0.775s

OK
```

Read-ahead tests verify one pending expert per store, a shared byte cap including
read-copy headroom, background file reads with decoding on the caller thread,
budget release after read/decode errors, close waiting for reads, oversized/stale
fallback, and admission reservations. Synthetic Qwen and GLM prefill and cached
continuation match their normal MLX paths within `atol=rtol=2e-4`. The loader hook
wires both adaptive caches and read-ahead; simultaneous model threads share a
staging pool. An injected compute failure verifies pending reads drain before
file-descriptor cleanup.

The bootstrap extends the prefix/adaptive fixture with a 16 MiB read-ahead pool
and per-model staging reservations derived from the pinned expert sizes. It
prints help without loading a model or opening a listener. Approved Metal access
was used for the synthetic suite and bootstrap. No full checkpoint was downloaded
or loaded, and no existing process was terminated. No model-run/community
benchmark protocol was performed. The test durations are not throughput
measurements; full-model SSD overlap, speed, peak RAM, and sustained harness
workloads remain unmeasured. No Swift source was changed or rebuilt.

## Adaptive cache follow-up

Validated with HEAD `40d6b1920b077c1b63630ee23890367c5fb5fd28` plus working-tree
changes, on the same Mac15,6 / M3 Pro / 12 CPU cores / 18 GiB, macOS 26.1
(25B78), Swift 6.2.4 environment detailed below. Exact test commands, exit 0:

```bash
bash Scripts/test.sh --qwen-agents > /tmp/adaptive-cache-tests.log 2>&1
bash Scripts/test.sh --expert-streaming > /tmp/adaptive-cache-mlx-tests.log 2>&1
```

Complete final footers, respectively:

```text
----------------------------------------------------------------------
Ran 63 tests in 0.482s

OK
```

```text
----------------------------------------------------------------------
Ran 7 tests in 0.632s

OK
```

Added checks cover pool/per-model ceilings, startup space for unloaded models,
reload after shrinking, shared growth headroom, pressure and telemetry failure,
adjustment intervals, weak ownership, LFU retention during resize, peak-cache
admission reservations, and invalid CLI combinations. The numerical suite
compares both synthetic architectures before growth, after growth, and after
shrinkage. Separate engine threads also run with a shared adaptive controller;
the real loader hook registers its store with the controller.

Bootstrap and telemetry command, exit 0:

```bash
PYTHONPATH=Scripts .build/qwen-agent-venv/bin/python - <<'PY' > /tmp/adaptive-cache-bootstrap.log 2>&1
import sys
from expert_streaming_mlx import sample_memory
used, available, total = sample_memory()
assert used >= 0 and 0 <= available <= total and total > 0
print('macOS/MLX memory telemetry available; no model loaded.')
from agent_runtime import main
sys.argv = ['agent_runtime.py', '/tmp/adaptive-cache-routes.json', 'serve', '--help']
main()
PY
python3 Scripts/serve-qwen-agents.py --help > /tmp/adaptive-cache-help.log
git diff --check
```

All exited 0. The temporary bootstrap routes extend the header-derived routes
below with 64 MiB minimum, 1,024 MiB maximum, a 1,024 MiB shared pool, and the
48 GiB memory ceiling. Each streamed model's admission reservation adds the
difference between its maximum and initial 256 MiB cache. The bootstrap reports
valid memory telemetry and prints server help without starting a listener.

The synthetic MLX checks and bootstrap used approved Metal access. No full
checkpoint was downloaded or loaded, no existing process was stopped, and no
model-run or community benchmark protocol was exercised. The tests are tiny
fixtures, not full-model measurements. Sustained SSD traffic, throughput, macOS
pressure response, and peak RAM on the target Mac remain unmeasured. Idle caches
resize at the next inference boundary or disappear through existing model
unloading; no background pressure daemon was introduced.

The implementation began on base commit
`4c6db1e698ea861609109d4bc517410ff302af46`, alongside existing Qwen/fleet work.
At final documentation and bootstrap verification the checkout HEAD was
`40d6b1920b077c1b63630ee23890367c5fb5fd28`.
Hardware: Mac15,6, M3 Pro, 12 CPU cores, 18 GiB unified RAM.
macOS 26.1 (25B78), Swift 6.2.4
(`swiftlang-6.2.4.1.4 clang-1700.6.4.2`). Python 3.13.15,
oMLX 0.6.4, MLX 0.32.0, MLX-LM 0.31.3.

## Tests

Exact commands, both exit 0:

```bash
bash Scripts/test.sh --qwen-agents > /tmp/expert-streaming-server-tests.log 2>&1
bash Scripts/test.sh --expert-streaming > /tmp/expert-streaming-tests.log 2>&1
```

Complete final footers, respectively:

```text
----------------------------------------------------------------------
Ran 39 tests in 0.054s

OK
```

```text
----------------------------------------------------------------------
Ran 6 tests in 0.179s

OK
```

The first suite covers server routing, concurrency/admission, package validation,
selective file reads, invalid tensor layouts, byte-bounded caching, layer cache
reuse across repeated scans, output-budget overrides, and fleet configuration.
The second uses tiny synthetic quantized Qwen-MoE and GLM-MoE checkpoints.
Streamed outputs match the normal loader within `atol=rtol=2e-4` for prefill and
cached continuation. It also covers batched duplicate routes, BF16 bit
preservation, cache reuse, file-descriptor cleanup, the installed engine's loader
hook, failure without resident-loader fallback, and separate engine threads.
An assertion rejects calls to the full-shard `mx.load` path during streamed load.

The synthetic suite used approved access to Metal outside the restricted
sandbox. It downloads no weights. Its small fixtures are not the installed
Gemma pack or full fleet models. No community benchmark was run; test duration
is not an inference-speed measurement. No Swift source changed in this work,
so the previous Swift build was not repeated.

## Checkpoint layout inspection

Only tiny pinned configuration/index files and HTTP Range responses containing
safetensors headers were retrieved. The header inspection required status 206
and an exact Content-Range matching the catalog's file length; a full-file 200
response was refused. No tensor payload or full checkpoint was downloaded.

| Role | Resident bytes | Expert bytes on disk | Cache bytes | Largest expert bytes | Planned weight bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| coder | 873459712 | 16307453952 | 268435456 | 2654208 | 1432677376 |
| research | 1224225792 | 15627976704 | 268435456 | 5308416 | 1876443392 |

The profiles cover 48 Qwen expert layers and 46 GLM expert layers. These numbers
are checkpoint-header arithmetic. They do not measure allocations, filesystem
cache, KV state, or peak process memory. Normal launch still verifies every
installed payload against its pinned hash before serving it.

## Bootstrap

A temporary routes file was generated from `agent_models.route_config` for all
roles with the 3-bit worker. The two header-derived profiles above supplied
`resident_weight_bytes`; their `expert_streaming` settings used the standard
local model paths, 256 MiB caches, and 16-row chunks. Admission was 40.8 GiB with
two concurrent requests.

```bash
.build/qwen-agent-venv/bin/python Scripts/agent_runtime.py /tmp/expert-streaming-routes.json serve --help > /tmp/expert-streaming-bootstrap.log 2>&1
python3 Scripts/serve-qwen-agents.py --help
bash -n Scripts/test.sh Scripts/setup-qwen-agents.sh
git diff --check
```

All exited 0. The bootstrap printed the installed server's help after installing
the streaming hook and routing middleware. It used approved Metal-framework
access and did not start a listener or load weights.

## Remaining validation

No full checkpoint inference, fleet server, or client harness was started. No
existing model process was terminated. The earlier fleet preflight refused this
18 GiB development Mac, and the selected full checkpoints are not installed.
The model-run preflight and community benchmark protocol were therefore not
executed for this work; synthetic fixtures and metadata inspection are the
explicit scope of the checks above.

Full-model loading, tool calling, peak RAM, SSD traffic, throughput, and sustained
Hermes/OpenClaw concurrent workloads require validation on the target Mac.
Streaming remains opt-in until those results are available. Neither the small
numerical tests nor the estimated weight budgets establish a 2 GiB total-RAM
claim or a performance ceiling.
