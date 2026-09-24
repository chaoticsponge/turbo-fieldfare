"""Persistent prefix-cache identity and read-only engine statistics."""
import hashlib
import importlib.metadata
import json


def prefix_cache_size(fleet, requested):
    return (4 if fleet else 0) if requested is None else requested


def runtime_identity(environment, requirements):
    versions = {}
    for location in environment.glob('lib/python*/site-packages'):
        for distribution in importlib.metadata.distributions(path=[str(location)]):
            name = (distribution.metadata.get('Name') or '').lower().replace('_', '-')
            if name in {'omlx', 'mlx', 'mlx-lm', 'mlx-vlm', 'safetensors', 'transformers', 'tokenizers'}:
                versions[name] = distribution.version
    return {'requirements_sha256': hashlib.sha256(requirements.read_bytes()).hexdigest(),
            'versions': versions}


def prefix_namespace(packages, runtime, streamed_models=()):
    streamed = set(streamed_models)
    models = [{'directory':path.name, 'repo':catalog['repoID'], 'revision':catalog['revision'],
               'expert_streaming':path.name in streamed} for path,catalog in packages]
    payload = {'format':1, 'runtime':runtime,
               'models':sorted(models,key=lambda item:(item['directory'],item['repo'],item['revision']))}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prefix_cache_snapshot(pool, routes, configured):
    result = {**configured, 'models': {}}
    if not configured.get('enabled'):
        return result
    for role, route in routes.items():
        if route['kind'] not in ('llm', 'vlm'):
            continue
        entry = pool.get_entry(route['model'])
        engine = entry.engine if entry else None
        if engine is None:
            result['models'][role] = {'state':'unloaded'}
            continue
        try:
            stats = engine.get_cache_stats()
            if stats is None:
                result['models'][role] = {'state':'unavailable'}
                continue
            names = ('hits','misses','tokens_saved','block_size','exact_prefix_hits',
                     'exact_prefix_tokens_restored','evictions')
            values = {key:stats.get(key) if isinstance(stats,dict) else getattr(stats,key,None)
                      for key in names}
            result['models'][role] = {'state':'ready', **{k:v for k,v in values.items() if type(v) in (int,float)}}
        except (AttributeError, RuntimeError):
            # An engine may be unloading while this read-only snapshot is taken.
            result['models'][role] = {'state':'unavailable'}
    return result
