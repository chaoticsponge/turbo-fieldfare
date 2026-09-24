# Shared prompt-prefix reuse

Fleet mode now enables the pinned engine's shared SSD prefix cache by default,
with a configured 4 GiB limit and no extra hot RAM cache:

```bash
python3 Scripts/serve-qwen-agents.py --fleet --worker-precision 3bit
```

It also works with `--stream-experts --adaptive-expert-cache`. Use
`--prefix-cache-gb 0` to disable reuse, or choose 2, 4, 8, or 16 GiB. Single-model
mode remains disabled by default and can opt in with the same option. Restart
through the launcher for the new settings to take effect.

## What agents share

Agents using the same underlying model share cached computation for identical
leading prompt tokens. Matching system instructions, tool definitions, repository
context, and conversation prefixes can therefore avoid repeated prefill. A role
alias, the underlying model ID, and `auto` resolving to that same model all use
the same engine cache. Different models do not share KV state.

The engine matches token prefixes and model identity, not semantic similarity.
Changed tools, instructions, images, or earlier conversation tokens can break a
match. Branching conversations reuse their common prefix and then compute their
own suffixes. This is not response caching, and each agent still generates a new
answer with its own active conversation state. Concurrent requests can reuse
published blocks; simultaneous cold requests are not guaranteed to avoid
duplicate initial prefill.

To make Hermes/OpenClaw prompts reusable:

1. Keep common system instructions and tool definitions identical and in a
   stable order for agents using the same role.
2. Put shared repository or task background before each agent's specific task.
3. Put changing timestamps, job IDs, and task-specific details later, where the
   harness allows it without changing instruction priority.
4. Use an explicit specialist for a persistent subagent so successive calls stay
   on the same model.

The server preserves supplied messages, tools, and their ordering. It does not
rewrite instructions, remove timestamps, or merge unrelated conversations.
No custom client field or agent ID is required. Some reuse operates on complete
blocks (normally 256 tokens, adjusted by the engine for certain cache layouts),
so very short prefixes may show no savings. Do not pad prompts just to fill a
cache block.

## Persistence and memory

The cache lives under `.build/qwen-agent-server/prefix-cache/<identity>` and can
survive model unloading and server restarts. Its identity includes the sorted
selected model packages, pinned revisions, runtime package versions and engine
requirements, and streamed-versus-resident execution mode. Reordering the same
role selection preserves it; a changed revision/runtime or selected model set
uses a different namespace. The engine additionally separates model identities
and multimodal keys within that namespace. Old namespaces are not deleted
automatically, so the configured limit is not a cap on all historical cache
directories combined.

The hot RAM prefix cache stays at zero; metadata, pending SSD writes, cache
restoration, and active KV state still require memory. Four initial metadata
blocks keep startup allocation small. Token-aware admission continues to reserve
full session memory rather than assuming a cache hit. Reuse primarily saves
prefill work; it does not guarantee lower peak RAM or faster SSD-heavy workloads.

## Verify reuse

`GET /health` now includes `prefix_cache`, with configured storage/limit,
compatibility namespace, and per-role live counters:

- `unloaded`: no model engine is loaded yet.
- `ready`: the engine exposes cache statistics, including hits, misses, and
  `tokens_saved`.
- `unavailable`: a loaded engine has no usable cache statistics, including an
  engine that could not initialize its cache. Requested caching alone is not
  reported as a successful hit.

Counters belong to the live engine and may reset after unloading. The engine's
OpenAI responses report reuse in `usage.prompt_tokens_details.cached_tokens`.
For streamed requests, request `"stream_options": {"include_usage": true}` to
receive its final usage event. The router passes these fields through.

Compare two calls to the same explicit role with a substantial identical prefix
and different final tasks. Inspect cached tokens and prefill time; do not infer
a hit merely from the configured cache size.

Validation uses real pinned-engine block matching and SSD storage with synthetic
data, including SSD restart and Qwen/GLM continuation equivalence. Full-model
Hermes/OpenClaw hit rates, throughput, and RAM savings remain unmeasured. See the
[validation record](LOCAL_AGENT_FLEET_VALIDATION.md).
