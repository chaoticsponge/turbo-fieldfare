# Expert streaming for local agents

The dedicated fleet server can stream routed expert weights from SSD for the
pinned **Qwen3-Coder-30B-A3B-Instruct-4bit** and **GLM-4.7-Flash-4bit** packages.
Enable it explicitly; the default remains normal resident MLX inference.
Full-checkpoint speed and peak memory have not been measured yet.

After installing the selected models using the [fleet guide](LOCAL_AGENT_FLEET.md):

```bash
python3 Scripts/serve-qwen-agents.py --fleet --worker-precision 3bit \
  --stream-experts --expert-cache-mb 256 --check
python3 Scripts/serve-qwen-agents.py --fleet --worker-precision 3bit \
  --stream-experts --expert-cache-mb 256
```

Run only one model process, keep the server on `127.0.0.1`, and close the native
app first. All selected packages must be fully installed and verified. Streaming
reduces weight residency, not download size or SSD storage requirements.

Hermes/OpenClaw use the same `http://127.0.0.1:8080/v1` endpoint and `auto` model.
A specialist subagent can request `coder` or `research` explicitly. All calls
to a model share one expert cache, rather than allocating a cache per subagent.
The existing admission queue supports overlapping requests across models;
streaming does not increase the default two-active-request limit.

## How memory is reduced

The loader reads the safetensors headers, retains ordinary weights in unified
memory, and replaces the routed expert layers before MLX can materialize their
full weight banks. Each layer loads only the experts selected by the model's
router, using bounded reads from the original checkpoint files. MLX's quantized
Metal matrix operations consume these weights directly. Shared experts and
attention weights stay resident.

The cache budget is per model, divided between its expert layers. Each layer
retains frequently used experts; this prevents one layer from evicting every
other layer's reusable weights. Temporary expert buffers are evaluated before
release. Allocator cleanup uses the pinned engine's synchronization and buffer
lock, including when different models run on separate engine threads.

No expert routes are dropped, no weights are requantized, and no duplicate
checkpoint or repack is created. Numerical equivalence was checked on small
synthetic models, including prefill, batches, and cached continuation; this is
not full-model quality validation.

With the default 256 MiB cache per model, the pinned checkpoint headers give:

| Model | Ordinary resident weights | Planned weights/cache/staging budget |
| --- | ---: | ---: |
| Qwen3-Coder 30B-A3B | 0.81 GiB | 1.33 GiB |
| GLM-4.7-Flash | 1.14 GiB | 1.75 GiB |

These are **header-based estimates, not measured peak RAM**. The plan includes
25% core-weight headroom, the cache, two largest-expert buffers, and 64 MiB extra
staging space. The admission scheduler adds its existing weight margin and
session reservations. KV state, activations, runtime allocations, and filesystem
cache still consume memory. The actual-memory guard remains enabled.

`--expert-cache-mb` accepts 64, 128, 256, or 512. Smaller caches save residency
but cause more SSD reads; if an expert exceeds its layer's share, it runs
uncached. `--expert-chunk-rows` accepts 1, 4, 8, 16, or 32 (default 16), bounding
the rows grouped for each streamed layer. Smaller chunks can increase overhead.
Host routing, synchronization, and repeated SSD reads can make this substantially
slower than resident inference, especially with simultaneous agents. Measure
real tasks before choosing it for latency-sensitive work.

## Adaptive expert caches

Add `--adaptive-expert-cache` to let streamed models adjust their caches using
memory availability and cache misses:

```bash
python3 Scripts/serve-qwen-agents.py --fleet --worker-precision 3bit \
  --stream-experts --adaptive-expert-cache \
  --expert-cache-mb 256 --expert-cache-max-mb 1024 --expert-cache-pool-mb 1024
```

These defaults start each streamed model at 256 MiB, permit at most 1,024 MiB
per model, and share a 1,024 MiB pool. The minimum is 64 MiB per model. The pool
must fit all configured models' initial caches. Startup space stays reserved
for unloaded models, so one busy model cannot prevent another from loading.
Unused space above those startup reservations can be shared. Without the flag,
the existing fixed-cache behavior is preserved.

The controller samples memory at most once every two seconds during inference.
It compares macOS available memory and the larger of process RSS or MLX
active-plus-allocator-cache bytes against the configured memory ceiling. It:

- Grows by at most 64 MiB per adjustment when at least 32 recent expert lookups
  have a miss rate of 10% or more, process usage is below 70% of the ceiling, and
  macOS reports at least 20% memory available.
- Halves the cache, down to 64 MiB, when usage reaches 85% of the ceiling or
  macOS available memory reaches 10% or less. Missing telemetry also shrinks
  rather than granting additional memory.
- Holds the current budget between those thresholds. A two-second adjustment
  interval limits repeated grow/shrink cycles.

Changes happen only at evaluated expert-layer chunk boundaries, on that model's
inference thread while its store lock is held. Shrinking evicts less-used
experts, followed by the engine's synchronized allocator cleanup. The controller
never resizes another thread's live cache. Idle models retain their cache until
their next inference boundary or existing idle unload; this is not a background
macOS memory-pressure daemon.

Admission reserves each model's **maximum** cache size up front, including the
usual weight margin. Growing a cache therefore cannot borrow another agent's
session reservation. This is deliberately conservative: a shrunken cache does
not immediately create new admission capacity, and configuring large maxima
can reduce concurrency even when current caches are smaller. With the example
above, each model reserves 768 MiB more cache capacity than fixed 256 MiB mode,
before the admission margin. The fixed-cache weight table above still describes
the 256 MiB configuration.

`GET /health` exposes `adaptive_expert_cache`: the shared pool, committed startup
space, latest sampled pressure, per-model budgets, retained expert bytes,
hits/misses, evictions, and the last adjustment reason. Retained expert bytes
exclude allocator, KV, and runtime memory; counters update at inference
boundaries. Cache growth happens on demand and does not preload unused experts.
Full-model throughput and peak RAM with this mode remain unmeasured.

## Other models and efficiency controls

The dense Qwen 27B worker cannot use routed-expert streaming: its dense layers
need their weights for every token. Its existing 3-bit package reduces weight
memory, with a possible quality tradeoff versus 8-bit. The extraction, embedding,
and reranking models retain their existing quantized resident loaders. The old
Gemma `.gturbo` runtime is separate; this option does not add Gemma to the fleet.

Fleet idle-unload eligibility is now 120 seconds for the worker, 180 for
extraction, 300 for coder/research, and 600 for the small retrieval models.
Idle models may be reclaimed earlier under admission pressure. Extraction
defaults to at most 2,048 output tokens and disables thinking; explicit caller
budgets and sampling settings take precedence. Fleet mode enables 4 GiB of
[shared SSD prompt reuse](SHARED_PREFIX_REUSE.md) by default without a separate
hot RAM cache; `--prefix-cache-gb 0` disables it.

Streaming is limited to the two configured local model paths and supported
stacked affine 4-bit expert tensors. Unsupported or corrupt streamed checkpoints
fail explicitly instead of silently loading the full expert bank. The loader
hook is process-local and does not edit installed oMLX/MLX packages. Gate/up
fusion is disabled only for streamed models because it would rebuild expert
weight banks.

## Validation

```bash
bash Scripts/test.sh --qwen-agents
bash Scripts/test.sh --expert-streaming
```

The second suite requires the dedicated environment installed by
`Scripts/setup-qwen-agents.sh` and Metal access. It creates tiny synthetic
checkpoints; it downloads no model weights. See the
[validation record](EXPERT_STREAMING_VALIDATION.md) for results and limitations.
