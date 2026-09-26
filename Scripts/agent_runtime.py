#!/usr/bin/env python3
"""Attach routing to the pinned oMLX app in the SAME inference process."""
import json
from pathlib import Path
import sys

from agent_routing import AgentRouter, RouteError, OUTPUT_RESERVATION, TOKEN_RESERVATION
from agent_models import weight_reservation
from agent_prefix_cache import prefix_cache_snapshot


async def reclaim_idle(pool, protected, session_bytes, weights, budget):
    loaded = set(pool.get_loaded_model_ids())
    projected = lambda: sum(weights[m] for m in loaded | protected) + session_bytes
    # Retain idle weights when they fit; reclaim LRU idle models when needed.
    for previous in sorted(loaded - protected, key=lambda mid: pool.get_entry(mid).last_access):
        if projected() <= budget:
            break
        if await pool.unload_if_idle_unpinned(previous):
            loaded.remove(previous)
    if projected() > budget:
        raise RouteError('Idle model cleanup has not completed; retry shortly', 503)


def install_context_validation(server):
    """Use the engine's actual chat/template token count, never character estimates."""
    original = server.validate_context_window

    def validate(num_prompt_tokens, model_id=None):
        original(num_prompt_tokens, model_id)
        output = OUTPUT_RESERVATION.get()
        limit = server.get_max_context_window(model_id)
        if output and limit and num_prompt_tokens + output > limit:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=(
                f'Context budget exceeded for {model_id}: {num_prompt_tokens} prompt tokens + '
                f'{output} reserved output tokens exceeds {limit}. Shorten the history or output budget.'))
        reserved = TOKEN_RESERVATION.get()
        if reserved is not None and num_prompt_tokens + output > reserved:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=(
                'Rendered prompt exceeds its admission token estimate; simplify the prompt or tool template. '
                'No generation was started.'))

    server.validate_context_window = validate


def main():
    routes_file = Path(sys.argv[1])
    config = json.loads(routes_file.read_text())
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    from omlx import server
    from omlx.cli import main as omlx_main

    from agent_transport import install_transport_limits
    install_transport_limits()
    install_context_validation(server)

    streaming = {entry['expert_streaming']['directory']: entry['expert_streaming']
                 for entry in config['routes'].values() if 'expert_streaming' in entry}
    read_ahead = None
    read_ahead_bytes = config.get('expert_read_ahead', {}).get('pool_bytes', 0)
    if read_ahead_bytes:
        from expert_read_ahead import ReadAheadPool
        read_ahead = ReadAheadPool(read_ahead_bytes)
    adaptive = None
    if streaming:
        from expert_streaming_mlx import install_loader, sample_memory
        limits = config.get('adaptive_expert_cache')
        if limits:
            from adaptive_expert_cache import AdaptiveExpertCaches
            adaptive = AdaptiveExpertCaches(limits['pool_bytes'], limits['memory_ceiling'], sample_memory,
                initial_budgets={Path(path).name: options['cache_bytes']
                                 for path, options in streaming.items() if options.get('adaptive')})
        install_loader(streaming, adaptive, read_ahead)

    budget = config['budget_bytes']
    weights = {entry['model']: weight_reservation(entry) for entry in config['routes'].values()}

    async def switch(model, protected, session_bytes):
        pool = server.get_engine_pool()
        for route in config['routes'].values():
            if 'expert_streaming' in route:
                entry = pool.get_entry(route['model'])
                if entry is not None and entry.engine is None and not entry.is_loading:
                    # oMLX otherwise counts the entire on-disk expert bank.
                    # Update before it reserves memory for the first load.
                    entry.estimated_size = int(weights[route['model']])
        await reclaim_idle(pool, protected, session_bytes, weights, budget)

    def prefix_status():
        configured = config.get('prefix_cache', {'enabled': False})
        if not configured.get('enabled'):
            return prefix_cache_snapshot(None, config['routes'], configured)
        try:
            pool = server.get_engine_pool()
        except (RuntimeError, server.HTTPException):
            return {**configured, 'state': 'starting', 'models': {}}
        return prefix_cache_snapshot(pool, config['routes'], configured)

    server.app.add_middleware(AgentRouter, routes=config['routes'],
                              switch=switch, concurrency=config['concurrency'], budget_bytes=budget,
                              default_role=config.get('default_role'),
                              cache_status=adaptive.snapshot if adaptive else None, prefix_status=prefix_status,
                              read_ahead_status=read_ahead.snapshot if read_ahead else None)
    omlx_main()


if __name__ == '__main__':
    main()
