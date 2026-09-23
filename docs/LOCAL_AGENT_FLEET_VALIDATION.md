# Local agent fleet validation — September 23, 2026

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
