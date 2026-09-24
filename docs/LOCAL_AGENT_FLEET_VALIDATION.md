# Local agent fleet validation — September 23, 2026

## Shared prefix reuse (completed September 24)

Same M3 Pro / 18 GiB, macOS 26.1, Swift 6.2.4 environment and base HEAD
`40d6b1920b077c1b63630ee23890367c5fb5fd28` plus working-tree changes.
Commands, all exit 0:

```bash
bash Scripts/test.sh --qwen-agents > /tmp/prefix-reuse-tests.log 2>&1
bash Scripts/test.sh --expert-streaming > /tmp/prefix-reuse-mlx-tests.log 2>&1
.build/qwen-agent-venv/bin/python Scripts/agent_runtime.py /tmp/prefix-reuse-routes.json serve --help > /tmp/prefix-reuse-bootstrap.log 2>&1
git diff --check
```

Complete test footers:

```text
----------------------------------------------------------------------
Ran 67 tests in 0.702s

OK
```

```text
----------------------------------------------------------------------
Ran 9 tests in 0.830s

OK
```

Coverage includes fleet defaults/opt-out, parsed engine settings, compatible
namespace stability and invalidation, and loaded/unavailable cache statistics.
The installed engine's block manager reuses the same prefix blocks across two
threads and rejects mismatched token/model/multimodal identities. Its real SSD
cache stores tiny Qwen and GLM KV states, closes, reopens, and restores them;
continuation logits match within `atol=rtol=2e-4`. These are synthetic fixtures,
not full-model or harness benchmarks.

The bootstrap routes extend the adaptive-cache fixture with enabled SSD prefix
reuse and a namespace produced from local runtime distribution metadata. It
prints help only. Metal access was approved for the synthetic suite/bootstrap.
No full model was downloaded, no listener or model process was started, and no
existing process was terminated. No model-run/community benchmark protocol was
performed; full-model cache hit rate, throughput, and RAM savings remain unmeasured.

## Token-aware admission (follow-up)

Same M3 Pro / 18 GiB, macOS 26.1, Swift 6.2.4 environment and HEAD
`40d6b1920b077c1b63630ee23890367c5fb5fd28` plus working-tree changes.
Exact commands, all exit 0:

```bash
bash Scripts/test.sh --qwen-agents > /tmp/token-admission-tests.log 2>&1
.build/qwen-agent-venv/bin/python Scripts/agent_runtime.py /tmp/expert-streaming-routes.json serve --help > /tmp/token-admission-bootstrap.log 2>&1
python3 Scripts/serve-qwen-agents.py --help > /tmp/token-admission-help.log
git diff --check
```

Complete test footer:

```text
----------------------------------------------------------------------
Ran 55 tests in 0.319s

OK
```

New tests cover GQA/hybrid/latent cache arithmetic, GLM attention workspace,
prompt/output growth, tools and Unicode history, image context reservations,
fallback and retrieval headroom, local configuration profile generation,
actual-count rejection when a template exceeds its estimated reservation,
variable-size concurrent leases, out-of-order release, queued cancellation,
oversized requests, idle eviction using session bytes, SSE lifetime, and failure
cleanup. The existing concurrent-routing tests also pass with the updated
callback carrying total reserved session bytes instead of request count.

The server bootstrap used approved Metal access, printed help, and loaded no
weights or listener. Its old temporary routes also verify backward-compatible
fixed-reservation fallback. No model process was started or terminated, and no
checkpoint was downloaded. No inference benchmark or model-run protocol was
performed. Cache estimates were checked against the pinned MLX implementation;
workspace margins and actual full-model peak RAM remain unmeasured. This work
does not change Swift code or claim a measured concurrency/speed improvement.

## Role-specific context budgets (follow-up)

Validated on the same M3 Pro / 18 GiB, macOS 26.1, Swift 6.2.4 environment
described below, with HEAD `40d6b1920b077c1b63630ee23890367c5fb5fd28` plus
working-tree changes. Exact commands, all exit 0:

```bash
bash Scripts/test.sh --qwen-agents > /tmp/role-context-tests.log 2>&1
.build/qwen-agent-venv/bin/python Scripts/agent_runtime.py /tmp/expert-streaming-routes.json serve --help > /tmp/role-context-bootstrap.log 2>&1
git diff --check
```

Complete test footer:

```text
----------------------------------------------------------------------
Ran 45 tests in 0.294s

OK
```

New coverage verifies role defaults, the global ceiling, explicit overrides,
invalid CLI arguments before preflight, matching routes and installed-engine
settings, output-token aliases, prompt-plus-output boundary rejection, and
concurrent request-local reservation isolation. Discovery advertises effective
limits. The engine validator test supplies synthetic token counts; it does not
load a tokenizer or a model. The bootstrap uses the temporary routes described
in [expert-streaming validation](EXPERT_STREAMING_VALIDATION.md), installs both
runtime hooks, and prints help without starting a listener. Approved Metal
access was used for this import check. No full-model inference or benchmark was
run; there are no measured RAM savings or additional model-run protocol results.

Base commit `4c6db1e698ea861609109d4bc517410ff302af46`, with preexisting
uncommitted work and this implementation. Machine: Mac15,6 / M3 Pro / 12 CPU
cores / 18 GiB RAM. macOS 26.1 (25B78), Apple Swift 6.2.4
(`swiftlang-6.2.4.1.4 clang-1700.6.4.2`). Dedicated environment: Python 3.13.15,
oMLX 0.6.4, MLX 0.32.0, MLX-LM 0.31.3.

## Model-free tests

```bash
bash Scripts/test.sh --qwen-agents > /tmp/agent-fleet-tests.log 2>&1
python3 Scripts/manage-agent-models.py list --worker-precision 3bit
bash -n Scripts/test.sh Scripts/setup-qwen-agents.sh
git diff --check
```

All commands exited 0. Complete test footer:

```text
----------------------------------------------------------------------
Ran 31 tests in 0.048s

OK
```

Coverage includes concurrent streams from two different model IDs, shared-weight
accounting, admission of coder/research and compact-worker pairs, queuing an
oversized pair, queue fairness/cancellation/backpressure, active-model protection
during idle eviction, preservation of tool payloads/SSE, explicit role overrides,
image rejection, separate bounded embedding/rerank endpoints, unknown-model and
bypass rejection, pinned receipts/hashes, and verified resumable downloads.
The fleet settings are parsed by the installed oMLX configuration classes,
including model engine types, idle timeout, and model directory allowlists.
These tests use fixtures rather than real inference engines/weights.

The first test attempt failed with `ModuleNotFoundError: No module named
'agent_models'`; the serial runner now supplies `Scripts` on `PYTHONPATH`.

## Engine bootstrap

```bash
PYTHONPATH=Scripts python3 - <<'PY'
import json
from pathlib import Path
from agent_models import ROLES, route_config
Path('/tmp/agent-fleet-routes.json').write_text(json.dumps({
    'routes': route_config(ROLES), 'concurrency': 2,
    'budget_bytes': 40.8 * 1024**3}))
PY
.build/qwen-agent-venv/bin/python Scripts/agent_runtime.py /tmp/agent-fleet-routes.json serve --help > /tmp/agent-fleet-bootstrap.log 2>&1
```

The bootstrap exited 0 and printed the installed server's help. It imports the
actual server and installs the routing middleware without starting a listener or
loading weights. The first sandboxed attempt exited 134:

```text
libc++abi: terminating due to uncaught exception of type std::runtime_error: [metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible.
```

The successful run used approved GPU-framework access outside that sandbox.

## Pinned architecture and tool-parser resolution

The following metadata-only command exited 0 with approved network/GPU-framework
access. It retrieves tiny pinned configuration/template files and checks their
catalog hashes; it does not construct a model.

```bash
PYTHONPATH=Scripts .build/qwen-agent-venv/bin/python - <<'PY' > /tmp/agent-fleet-architecture.log 2>&1
import hashlib,json,urllib.request
from agent_models import catalog
from mlx_lm.utils import _get_classes
from mlx_lm.tokenizer_utils import _infer_tool_parser
for role,expected in [('coder','qwen3_coder'),('research','glm47')]:
 c=catalog(role)
 def read(name):
  entry=next(f for f in c['files'] if f['name']==name)
  with urllib.request.urlopen(f"https://huggingface.co/{c['repoID']}/resolve/{c['revision']}/{name}",timeout=30) as response:
   data=response.read()
  assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
  return data.decode()
 config=json.loads(read('config.json'))
 model,args=_get_classes(config)
 args.from_dict(config)
 parser=_infer_tool_parser(read('chat_template.jinja'))
 assert parser==expected,(role,parser)
 print(role,config['model_type'],model.__module__,parser)
print('Pinned configurations and built-in tool parsers resolved; no model instantiated.')
PY
```

Complete output:

```text
coder qwen3_moe mlx_lm.models.qwen3_moe qwen3_coder
research glm4_moe_lite mlx_lm.models.glm4_moe_lite glm47
Pinned configurations and built-in tool parsers resolved; no model instantiated.
```

Catalog preparation fetched file metadata and small files, using Hub LFS SHA-256
values for large payloads. Searches for an 8-bit reranker and 4B embedding package
under particular repository IDs returned HTTP 401; those candidates were not
added. The selected public repositories were verified and pinned successfully.

## Target-machine limitation

```bash
python3 Scripts/serve-qwen-agents.py --fleet --worker-precision 3bit --check > /tmp/agent-fleet-preflight.log 2>&1
```

Exit 1, complete output:

```text
error: Insufficient unified RAM for this model plus concurrent sessions; choose a smaller package.
```

The launcher correctly refused the 18 GiB development Mac. No fleet was launched,
no full checkpoint was downloaded, and no existing process was stopped. There
is no `~/.hermes/config.yaml` on this machine, so client configuration is provided
as a target-Mac example, not claimed to be applied to a running harness.

No Swift source or dependency changes were needed for this task, and the prior
Swift build was not repeated. No community benchmark was run; the old Gemma-pack
model-run preflight was not exercised because there was no model run. Live GPU
multi-model overlap, all-role inference, full-model tool calling, client behavior,
throughput, peak memory, and retrieval relevance remain unverified on the target
64 GB Mac. Package-size arithmetic and fixture concurrency are not performance
measurements or proof that every long-context workload fits.
