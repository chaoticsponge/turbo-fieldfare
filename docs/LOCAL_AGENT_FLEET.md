# Local specialist models and prompt routing

Use this mode to let Hermes or OpenClaw call multiple local specialists through
`http://127.0.0.1:8080/v1`. It is a dedicated server mode, separate from the Mac
chat app. The runtime can keep **different models generating concurrently**;
all inference stays in one MLX/Metal process on the Mac. There is no cloud model
fallback and no extra model used to classify prompts.

## Selected models

These are practical role choices for a 64 GB Mac, prioritizing supported MLX
packages, task specialization, and memory. They are not a measured ranking or a
claim of parity with frontier closed models. Sizes below are pinned package
sizes, not peak inference memory.

| API role | Model | Package size | Purpose |
| --- | --- | --- | --- |
| `worker` | Qwen3.8 27B, existing 3-bit or 8-bit | 11.86 / 27.50 GiB | General chat, planning, tool coordination, images |
| `coder` | Qwen3-Coder-30B-A3B-Instruct, 4-bit | 16.02 GiB | Coding, debugging, repository work |
| `research` | GLM-4.7-Flash, 4-bit | 15.71 GiB | Research synthesis, evidence comparison, long text |
| `extract` | Qwen3.5-9B, 4-bit | 5.57 GiB | Scraped-text extraction and classification |
| `embed` | Qwen3-Embedding-0.6B, 8-bit | 0.60 GiB | Text retrieval vectors |
| `rerank` | Qwen3-Reranker-0.6B, 4-bit | 0.32 GiB | Reorder retrieved evidence against a query |

[Qwen3-Coder's model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
describes its coding/tool specialization and 30.5B total / 3.3B active parameters.
[GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) supplies the requested
30B-class GLM specialist; its assignment to research is an application choice,
not evidence that it always beats Qwen at research.
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) provides a smaller worker.
The compact [Qwen retrieval family](https://github.com/QwenLM/Qwen3-Embedding)
keeps retrieval overhead low. Larger retrieval models may improve quality on
some corpora, but were not benchmarked here. Exact MLX repositories, immutable
revisions, file sizes, and SHA-256 hashes are in `Scripts/agent-models` and the
existing app resources, listed by the installer below.

## Install and start on the 64 GB Mac

The concurrent profile below uses the existing **3-bit 27B worker**, leaving room
for it to overlap with the coder or research model. Keep `--worker-precision`
the same during installation and launch. Choose `8bit` instead for the existing
higher-precision worker; some large-model pairs will then queue for RAM.

```bash
bash Scripts/setup-qwen-agents.sh
python3 Scripts/manage-agent-models.py list --worker-precision 3bit
python3 Scripts/manage-agent-models.py install all --worker-precision 3bit
python3 Scripts/serve-qwen-agents.py --fleet --worker-precision 3bit --check
python3 Scripts/serve-qwen-agents.py --fleet --worker-precision 3bit
```

Installation is explicit: `list`, setup, and server startup never download
weights. `install all` downloads about **50.1 GiB** with the 3-bit worker, or
**65.7 GiB** with the 8-bit worker, less already verified files. It streams to
resumable `.part` files, verifies each SHA-256, and writes the install receipt
last. Rerun the same install command after interruption. A corrupt completed
file or partial fails clearly rather than being accepted. Repo Python files are
excluded and remote model code is not trusted. Existing app installations at
the same paths are reused without copying weights.

To install selected roles, use e.g. `install coder research`. To serve a subset,
use `--fleet --roles worker coder research --worker-precision 3bit`. An automatic
route to an omitted role returns an explicit error, not a silent substitution.
`--root` on the installer and `--model-root` on the server select another
installation parent. All selected installations must be complete before launch.

Quit the native app before starting the dedicated server. The launcher checks
macOS, Swift, memory pressure, disk space, running model processes, RAM admission,
and every installed payload. Keep it on loopback and keep the terminal open;
Ctrl-C stops its own server. No second inference process or remote proxy is used.

## Concurrent scheduling and memory

- Two active API requests by default, across **the same or different models**.
  `--concurrency 3` or `4` is supported, subject to the same memory admission.
- Shared weights are counted once per active model. Each active request reserves
  3 GiB of session headroom; weight reservations include a 5% margin.
- On a 64 GiB machine, admission uses 40.8 GiB, the soft watermark of the existing
  48 GiB oMLX guard. The guard also monitors actual runtime memory and can defer
  or abort work. Reservations are estimates, not a guaranteed maximum.
- Idle models remain loaded if they fit. Before admitting another model, the
  router reclaims least-recently-used idle models as needed. Active models are
  never deliberately unloaded by the router.
- A FIFO queue allows 16 waiting requests and waits at most 300 seconds. Full
  queues return 429; admission timeouts return 503. Disconnects remove queued
  requests. A streaming request retains its reservation through its final chunk.
- Coder + research, or the 3-bit worker + either specialist, fit the admission
  estimate on 64 GiB. An 8-bit worker + coder does not and therefore queues.
  These are arithmetic admission checks, not measured peak-memory results.

All six roles are callable, but **all six models need not be resident at once**.
Their aggregate weights plus KV state leave too little room to guarantee that
on 64 GB. The scheduler permits useful overlap without requiring all weights in
RAM. GPU time and bandwidth are still shared, so overlap does not imply twice
the throughput. Long contexts may hit the actual-memory guard before the nominal
65,536-token API context cap. Start with two active calls and measure real tasks.

Optional `--prefix-cache-gb 4` enables bounded SSD prefix reuse, with no separate
hot RAM cache. The cache directory is tied to the selected revision set, and
oMLX separates model cache entries. Loading an evicted model still has a cost.
Idle models are eligible for automatic unloading after five minutes.

## Routing behavior

Send `model: "auto"` (or omit it) to `/v1/chat/completions`:

1. Images go to the vision-capable general worker.
2. Coding/debugging/repository instructions go to `coder`.
3. Research/evidence/citation requests go to `research`.
4. Extraction/classification/scraping-text requests go to `extract`.
5. Very long user text goes to `research`; other prompts use `worker`.

These are transparent English keyword rules, not a semantic routing model.
Ambiguous and other-language prompts generally use the worker. A request to
write a scraper is coding; extracting fields from already scraped HTML is
extraction. Role selection uses user intent, excluding tool outputs and assistant
text. Simple follow-ups such as “continue” reuse the prior substantive user
intent. Explicit `model: "coder"`, `"research"`, `"worker"`, or `"extract"`
overrides routing and is preferable for a persistent specialist subagent.

Tool definitions, call IDs, history, and streaming deltas pass through. Defaults
are role-specific, and explicit caller sampling settings win. Extraction disables
thinking by default. Responses include `X-Agent-Role` and `X-Agent-Model` headers;
the response body names the actual model. No prompt log is added by the router.
Missing models and unsupported images fail explicitly. Hermes remains responsible
for tool execution, approvals, browsing, sending email, and creating subagents.
Routing itself does not perform those actions.

## Hermes connection

There is no Hermes configuration on this development Mac. On the target Mac,
merge [the local inference settings](examples/hermes-local-fleet.yaml) into your
Hermes configuration, preserving tool permissions and account settings. Use the
same machine because the endpoint is loopback-only. The main model is `auto`,
provider `custom`, endpoint `http://127.0.0.1:8080/v1`, context `65536`. A local
key placeholder can be entered in `hermes model` if requested.

The example clears primary/delegation cloud fallbacks and sets common auxiliary
LLM tasks to the local API. Review any other existing auxiliary overrides,
per-agent model overrides, plugins, or fallbacks before claiming the entire
harness is local. The server never calls cloud models; it cannot override a
Hermes plugin that independently calls one. Web scraping itself still accesses
websites. See Hermes's [custom provider setup](https://hermes-agent.nousresearch.com/docs/integrations/providers)
and [auxiliary model configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration/#auxiliary-models).

[The OpenClaw fragment](examples/openclaw-local-fleet.json) exposes `auto` and
explicit chat roles, with both parent and subagent defaults set to the local
router. It is an alternative client configuration, not another model server.

## API checks and retrieval

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/models
curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Debug this Python function: def f(): return missing_name"}],"stream":true}'
```

Embedding and reranking are **separate operations**, not chat completions:

```bash
curl --fail http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embed","input":["Instruct: Retrieve relevant job listings\nQuery: remote Swift developer","Remote Swift developer role in Dubai"]}'
curl --fail http://127.0.0.1:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"rerank","query":"remote Swift developer","documents":["Remote Swift developer role","Onsite accounting position"],"top_n":1}'
```

A retrieval client embeds document chunks, stores vectors, retrieves nearest
neighbors, reranks the candidates, and sends selected evidence to the chat role.
The server provides the two model endpoints; it does not create a vector index or
automatically ingest private files. Hermes needs a retrieval tool/client to use
them. Embedding requests accept up to 32 strings of 8,192 characters each;
reranking accepts up to 32 documents of 16,384 characters and an 8,192-character
query. Oversized inputs are rejected, not silently truncated.

Only health, model discovery, chat completions, embeddings, and reranking are
exposed in fleet mode. Responses/Anthropic/admin/model-load routes are blocked
so they cannot bypass admission. The single-model server mode is unchanged.

## Validation

See [the validation record](LOCAL_AGENT_FLEET_VALIDATION.md). Real multi-model
inference, Hermes tool loops, throughput, and peak RAM still require validation
on the target 64 GB Mac. No weights were downloaded just to run tests.
