#!/usr/bin/env python3
"""Serve verified local packages with concurrent MLX inference.

Single-model mode exposes one installation; --fleet adds specialist autorouting
and memory-aware cross-model concurrency. The dedicated oMLX process runs
instead of the native app/service. This launcher never executes agent tools.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile

from agent_prefix_cache import prefix_cache_size, prefix_namespace, runtime_identity
from agent_admission import cache_profile
from agent_models import ROLES, catalog as role_catalog, model_path, route_config, context_budgets, CONTEXT_CHOICES

ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIRECTORY = ROOT / "Sources/TurboFieldfareApp/Core/Resources"
MODEL = ROOT / "scratch/qwen3.8-27b-8bit.mlx"
ENGINE = ROOT / ".build/qwen-agent-venv/bin/omlx"
PROCESS_PATTERN = (
    "TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|"
    "TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|"
    "mlx_lm|mlx-lm|omlx"
)
GIB = 1024 ** 3


def supported_catalogs():
    return [json.loads(path.read_text()) for path in sorted(CATALOG_DIRECTORY.glob("qwen-*.json"))]


def installed_catalog(directory):
    receipt = directory / "qwen-install.json"
    if directory.is_symlink() or receipt.is_symlink() or not receipt.is_file():
        raise ValueError("Complete the model download in the app first (missing or invalid receipt).")
    installed = json.loads(receipt.read_text())
    for catalog in supported_catalogs():
        if installed == catalog:
            return catalog
    raise ValueError("Installation does not match any supported pinned Qwen package.")


def validate_package(directory, catalog):
    """Bounded file hashing; no mmap/model allocation and no partial installs."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Model must be a completed installation directory, not a symlink.")
    receipt = directory / "qwen-install.json"
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError("Complete the model download in the app first (missing receipt).")
    if json.loads(receipt.read_text()) != catalog:
        raise ValueError("Installation does not match the pinned Qwen package.")
    names = {entry["name"] for entry in catalog["files"]}
    # oMLX discovers safetensors directly. Reject additional weights/models so
    # a modified directory cannot load anything outside the verified catalog.
    for path in directory.iterdir():
        if path.is_dir() or (path.suffix == ".safetensors" and path.name not in names):
            raise ValueError(f"Unexpected model content: {path.name}")
    weights = 0
    for entry in catalog["files"]:
        name = entry["name"]
        if Path(name).name != name or ".." in name:
            raise ValueError("Invalid catalog path.")
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or incomplete model file: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while data := handle.read(1024 * 1024):
                digest.update(data)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Model checksum mismatch: {name}")
        if name.endswith(".safetensors"):
            weights += entry["bytes"]
    return weights


def memory_ceiling(physical_bytes, weight_bytes):
    # This is an admission policy, not a predicted runtime footprint.
    ceiling = min(48, physical_bytes * 0.75 / GIB)
    if weight_bytes + 8 * GIB > ceiling * GIB:
        raise ValueError("Insufficient unified RAM for this model plus concurrent sessions; choose a smaller package.")
    return ceiling


def settings(model, concurrency, context, ceiling, cache_gb=0, cache_directory=None):
    if cache_gb and cache_directory is None:
        raise ValueError("Prefix cache requires a dedicated directory.")
    return {
        "server": {"host": "127.0.0.1", "cors_origins": [], "sse_keepalive_mode": "chunk"},
        "model": {"model_dirs": [str(model)], "model_fallback": False},
        "scheduler": {"max_concurrent_requests": concurrency, "chunked_prefill": True,
                      "decode_fairness": True, "prefill_priority": "context"},
        "memory": {"prefill_memory_guard": True, "memory_guard_tier": "custom",
                   "memory_guard_custom_ceiling_gb": ceiling,
                   "soft_threshold": 0.85, "hard_threshold": 0.95},
        "sampling": {"max_context_window": context, "max_context_window_policy": context,
                     "max_tokens": min(4096, context // 2), "temperature": 1.0,
                     "top_p": 0.95, "top_k": 20},
        "cache": {"enabled": bool(cache_gb), "hot_cache_max_size": "0",
                  "ssd_cache_dir": str(cache_directory) if cache_directory else None,
                  "ssd_cache_max_size": f"{max(1, cache_gb)}GB", "initial_cache_blocks": 4},
        "huggingface": {"hf_cache_enabled": False},
    }


def command(model, state, port, concurrency, ceiling, cache_gb=0, cache_directory=None):
    args = [str(ENGINE), "serve", "--model-dir", str(model), "--base-path", str(state),
            "--host", "127.0.0.1", "--port", str(port),
            "--max-concurrent-requests", str(concurrency),
            "--memory-guard-gb", str(ceiling), "--no-hf-cache",
            "--sse-keepalive-mode", "chunk"]
    if cache_gb:
        if cache_directory is None:
            raise ValueError("Prefix cache requires a dedicated directory.")
        args += ["--paged-ssd-cache-dir", str(cache_directory),
                 "--paged-ssd-cache-max-size", f"{cache_gb}GB",
                 "--hot-cache-max-size", "0", "--initial-cache-blocks", "4"]
    else:
        args += ["--no-cache"]
    return args


def check_processes():
    result = subprocess.run(["pgrep", "-fl", PROCESS_PATTERN], capture_output=True, text=True)
    if result.returncode == 0:
        raise ValueError("Close the existing model app/server before starting agent mode:\n" + result.stdout)
    if result.returncode != 1:
        raise ValueError("Could not inspect running model processes: " + result.stderr)


def configure_fleet(state, roles, model_root, configured, launch, concurrency, context, worker_precision='8bit',
                    stream_profiles=None, expert_chunk_rows=16, role_contexts=None, adaptive_cache=None, read_ahead_pool_bytes=0):
    configured['model']['model_dirs'] = [str(model_path(role, model_root, worker_precision)) for role in roles]
    configured['scheduler']['embedding_batch_size'] = 4
    configured['idle_timeout'] = {'idle_timeout_seconds': 300}
    routes = route_config(roles, model_root, worker_precision)
    budgets = context_budgets(roles, context, role_contexts)
    for role, entry in routes.items():
        entry['max_context_window'] = budgets[role]
        config_path = model_path(role, model_root, worker_precision) / 'config.json'
        entry['cache_profile'] = cache_profile(json.loads(config_path.read_text())) if config_path.is_file() else {'layout': 'fallback'}
    for role, profile in (stream_profiles or {}).items():
        routes[role]['resident_weight_bytes'] = profile['planned_weight_bytes']
        routes[role]['expert_streaming'] = {'directory': str(model_path(role, model_root, worker_precision)),
            'cache_bytes': profile['cache_bytes'], 'chunk_rows': expert_chunk_rows}
        if read_ahead_pool_bytes:
            routes[role]['expert_streaming']['read_ahead'] = True
            extra = min(2 * profile['largest_expert_bytes'], read_ahead_pool_bytes)
            routes[role]['resident_weight_bytes'] += extra
            routes[role]['expert_streaming']['read_ahead_reserved_bytes'] = extra
        if adaptive_cache:
            routes[role]['expert_streaming']['adaptive'] = adaptive_cache
            # Reserve the maximum up front: growth cannot consume session headroom.
            routes[role]['resident_weight_bytes'] += adaptive_cache['max_bytes'] - profile['cache_bytes']
    (state / 'routes.json').write_text(json.dumps({'routes': routes, 'concurrency': concurrency,
        'budget_bytes': configured['memory']['memory_guard_custom_ceiling_gb'] * 1024**3 * 0.85,
        'expert_read_ahead': {'pool_bytes': read_ahead_pool_bytes},
        'prefix_cache': {'enabled': configured['cache']['enabled'], 'storage': 'ssd',
            'limit': configured['cache']['ssd_cache_max_size'] if configured['cache']['enabled'] else '0',
            'hot_cache_bytes': 0, 'scope': 'matching-prefix-within-model',
            'namespace': Path(configured['cache']['ssd_cache_dir']).name if configured['cache']['ssd_cache_dir'] else None},
        **({'adaptive_expert_cache': {**adaptive_cache, 'memory_ceiling': configured['memory']['memory_guard_custom_ceiling_gb'] * 1024**3}} if adaptive_cache else {})}))
    preferences = {}
    idle_seconds = {'worker': 120, 'coder': 300, 'research': 300,
                    'extract': 180, 'embed': 600, 'rerank': 600}
    for role, entry in routes.items():
        preferences[entry['model']] = {
            'model_type_override': entry['kind'], 'is_pinned': False,
            'ttl_seconds': idle_seconds[role],
            'max_context_window': budgets[role],
            **{key: entry[key] for key in ('temperature', 'top_p', 'top_k') if key in entry}}
        if 'expert_streaming' in entry:
            preferences[entry['model']]['moe_gate_up_fusion_enabled'] = False
    (state / 'model_settings.json').write_text(json.dumps({'version': 1, 'models': preferences}))
    # Keep the directory allowlist from settings, without a CLI override.
    launch = list(launch)
    del launch[launch.index('--model-dir'):launch.index('--model-dir') + 2]
    return [str(ENGINE.parent / 'python'), str(ROOT / 'Scripts/agent_runtime.py'),
            str(state / 'routes.json'), *launch[1:]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--fleet", action="store_true", help="Enable specialist roles and prompt routing in one model process")
    parser.add_argument('--worker-precision', choices=['3bit', '8bit'], default='8bit',
                        help='Fleet worker precision; 3bit leaves more room for overlapping large models')
    parser.add_argument('--stream-experts', action='store_true',
                        help='Opt in to SSD expert streaming for fleet coder/research models (unbenchmarked)')
    parser.add_argument('--expert-cache-mb', type=int, choices=[64,128,256,512], default=256,
                        help='Shared expert-cache MiB per streamed model, not per subagent')
    parser.add_argument('--adaptive-expert-cache', action='store_true',
                        help='Grow/shrink streamed caches within reserved memory and a shared pool')
    parser.add_argument('--expert-cache-max-mb', type=int, choices=[256,512,1024,2048], default=1024,
                        help='Adaptive per-model ceiling (MiB); admission reserves this maximum')
    parser.add_argument('--expert-cache-pool-mb', type=int, choices=[256,512,1024,2048,4096], default=1024,
                        help='Combined adaptive cache budget across streamed models (MiB)')
    parser.add_argument('--expert-read-ahead', action='store_true',
                        help='Read one upcoming routed expert per streamed model in the background')
    parser.add_argument('--expert-read-ahead-mb', type=int, choices=[8,16,32], default=16,
                        help='Shared read-ahead staging budget in MiB, including read-copy headroom')
    parser.add_argument('--expert-chunk-rows', type=int, choices=[1,4,8,16,32], default=16,
                        help='Token rows processed together inside streamed expert layers')
    parser.add_argument("--roles", nargs="+", choices=list(ROLES), default=list(ROLES),
                        help="Roles to expose in fleet mode; omitted roles fail explicitly")
    parser.add_argument("--model-root", type=Path, default=ROOT / "scratch",
                        help="Installation root for fleet mode")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--context", type=int,
                        choices=CONTEXT_CHOICES, default=65536,
                        help="Single-model context; fleet ceiling for all role budgets")
    parser.add_argument("--role-context", action="append", default=[], metavar="ROLE=TOKENS",
                        help="Fleet context override, repeatable; cannot exceed --context")
    parser.add_argument("--prefix-cache-gb", type=int, choices=[0, 2, 4, 8, 16], default=None,
                        help="Shared SSD prefix reuse: fleet defaults to 4 GiB, single-model to 0; 0 disables")
    parser.add_argument("--list-models", action="store_true", help="List supported pinned packages without loading models")
    parser.add_argument("--check", action="store_true", help="Verify prerequisites and checksums without starting inference")
    args = parser.parse_args()
    args.prefix_cache_gb = prefix_cache_size(args.fleet, args.prefix_cache_gb)
    role_contexts = {}
    if args.role_context and not args.fleet:
        parser.error('--role-context requires --fleet')
    try:
        for override in args.role_context:
            role, value = override.split('=', 1)
            if role in role_contexts:
                raise ValueError(f'Duplicate context override: {role}')
            role_contexts[role] = int(value)
        budgets = context_budgets(args.roles, args.context, role_contexts)
    except ValueError as error:
        parser.error(f'Invalid --role-context: {error}')
    if args.stream_experts and not args.fleet:
        parser.error('--stream-experts requires --fleet')
    if args.stream_experts and not set(args.roles) & {'coder', 'research'}:
        parser.error('--stream-experts requires the coder or research role')
    if args.expert_read_ahead and not args.stream_experts:
        parser.error('--expert-read-ahead requires --stream-experts')
    read_ahead_pool_bytes = args.expert_read_ahead_mb * 1024**2 if args.expert_read_ahead else 0
    adaptive_cache = None
    if args.adaptive_expert_cache:
        if not args.stream_experts:
            parser.error('--adaptive-expert-cache requires --stream-experts')
        count = len(set(args.roles) & {'coder', 'research'})
        if args.expert_cache_max_mb < args.expert_cache_mb:
            parser.error('--expert-cache-max-mb must be at least --expert-cache-mb')
        if args.expert_cache_pool_mb < count * args.expert_cache_mb:
            parser.error('--expert-cache-pool-mb must fit all initial streamed caches')
        adaptive_cache = {'min_bytes':64*1024**2, 'max_bytes':args.expert_cache_max_mb*1024**2,
                          'pool_bytes':args.expert_cache_pool_mb*1024**2}
    if args.list_models:
        for catalog in supported_catalogs():
            size = sum(f["bytes"] for f in catalog["files"]) / GIB
            print(f'{catalog["repoID"]}: {size:.2f} GiB package, revision {catalog["revision"]}')
        return 0
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    try:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ValueError("Apple Silicon macOS is required.")
        macos = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
        if int(macos.split(".")[0]) < 26:
            raise ValueError("macOS 26 or later is required by this checkout.")
        swift = subprocess.check_output(["swift", "--version"], text=True, stderr=subprocess.STDOUT)
        version = re.search(r"Swift version (\d+)\.(\d+)", swift)
        if not version or tuple(map(int, version.groups())) < (6, 2):
            raise ValueError("Swift 6.2 or later is required by this checkout.")
        check_processes()
        pressure = subprocess.check_output(["memory_pressure", "-Q"], text=True)
        free = re.search(r"memory free percentage:\s*(\d+)%", pressure)
        if not free or int(free.group(1)) < 15:
            raise ValueError("Memory pressure is too high or could not be assessed:\n" + pressure)
        if shutil.disk_usage(ROOT).free < (2 + args.prefix_cache_gb) * GIB:
            raise ValueError("Insufficient free disk for the requested prefix cache plus 2 GiB for server state/logs.")
        if not ENGINE.is_file():
            raise ValueError("Run bash Scripts/setup-qwen-agents.sh first.")
        if args.fleet:
            roles = list(dict.fromkeys(args.roles))
            model_root = args.model_root.expanduser().absolute()
            packages = [(model_path(role, model_root, args.worker_precision), role_catalog(role, args.worker_precision)) for role in roles]
        else:
            model = args.model.expanduser().absolute()
            packages = [(model, installed_catalog(model))]
        model, catalog = packages[0]
        stream_profiles = {}
        if args.stream_experts:
            from expert_streaming import inspect_model
            for role, (directory, _) in zip(roles, packages):
                if role in ('coder', 'research'):
                    profile = inspect_model(directory, args.expert_cache_mb * 1024**2)
                    stream_profiles[role] = profile
                    print(f"{role}: planned core/cache/staging {profile['planned_weight_bytes'] / GIB:.2f} GiB; "
                          f"expert bank {profile['expert_bytes'] / GIB:.2f} GiB stays on disk", flush=True)
        physical = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        # Fleet admission reserves active weights and reclaims idle models.
        # Require room for each model individually; overlapping requests are
        # admitted only when their combined reservation fits the soft ceiling.
        planned = [sum(f["bytes"] for f in data["files"] if f["name"].endswith(".safetensors"))
                   for _, data in packages]
        if args.fleet:
            for index, role in enumerate(roles):
                if role in stream_profiles:
                    planned[index] = stream_profiles[role]['planned_weight_bytes']
                    if read_ahead_pool_bytes:
                        planned[index] += min(2 * stream_profiles[role]['largest_expert_bytes'], read_ahead_pool_bytes)
                    if adaptive_cache:
                        planned[index] += adaptive_cache['max_bytes'] - stream_profiles[role]['cache_bytes']
        weights = max(planned)
        ceiling = memory_ceiling(physical, weights)
        for directory, data in packages:
            print(f"Verifying {directory.name}…", flush=True)
            validate_package(directory, data)
        if args.fleet:
            print("Role context budgets: " + ", ".join(f"{role}={tokens}" for role, tokens in budgets.items()), flush=True)
        print(f"Shared SSD prefix reuse: {args.prefix_cache_gb} GiB; hot RAM cache: disabled", flush=True)
        if args.check:
            print("Preflight passed; no model loaded.")
            return 0
        state_root = ROOT / ".build/qwen-agent-server"
        state_root.mkdir(parents=True, exist_ok=True)
        # Keep this lock in the supervisor; child oMLX changes its process title.
        with (state_root / "server.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another agent-server launcher is already running.")
            check_processes()
            # Fresh per-run settings avoid inheriting global oMLX model dirs,
            # tool integrations, or experimental controls from another install.
            state = Path(tempfile.mkdtemp(prefix="run-", dir=state_root))
            cache_key = prefix_namespace(packages,
                runtime_identity(ENGINE.parent.parent, ROOT / 'Scripts/qwen-agent-requirements.txt'),
                streamed_models=[model_path(role, model_root, args.worker_precision).name for role in stream_profiles]
                    if args.fleet else ())
            cache_directory = state_root / "prefix-cache" / cache_key
            configured = settings(model, args.concurrency, args.context, ceiling,
                                  args.prefix_cache_gb, cache_directory)
            launch = command(model, state, args.port, args.concurrency, ceiling,
                             args.prefix_cache_gb, cache_directory)
            if args.fleet:
                launch = configure_fleet(state, roles, model_root, configured, launch,
                                         args.concurrency, args.context, args.worker_precision,
                                         stream_profiles, args.expert_chunk_rows, role_contexts, adaptive_cache, read_ahead_pool_bytes)
            (state / "settings.json").write_text(json.dumps(configured, indent=2) + "\n")
            environment = {k: v for k, v in os.environ.items() if not k.startswith("OMLX_")}
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            print(f"API: http://127.0.0.1:{args.port}/v1\nModel: {'auto (specialist routing)' if args.fleet else model.name}\n"
                  f"Concurrent generations: {args.concurrency}; context: {args.context}; "
                  f"memory guard ceiling: {ceiling:g} GiB\n"
                  "Keep this terminal open. Stop this server before reopening the native app.", flush=True)
            child = subprocess.Popen(launch, env=environment, start_new_session=True)
            def stop_child(signum, _frame):
                if child.poll() is None:
                    child.send_signal(signum)
            handlers = {sig: signal.signal(sig, stop_child) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                return child.wait()
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
