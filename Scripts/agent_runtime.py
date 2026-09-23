#!/usr/bin/env python3
"""Attach routing to the pinned oMLX app in the SAME inference process."""
import json
from pathlib import Path
import sys

from agent_routing import AgentRouter, RouteError
from agent_models import weight_reservation


async def reclaim_idle(pool, protected, active, weights, budget):
    loaded = set(pool.get_loaded_model_ids())
    projected = lambda: sum(weights[m] for m in loaded | protected) + active * 3 * 1024**3
    # Retain idle weights when they fit; reclaim LRU idle models when needed.
    for previous in sorted(loaded - protected, key=lambda mid: pool.get_entry(mid).last_access):
        if projected() <= budget:
            break
        if await pool.unload_if_idle_unpinned(previous):
            loaded.remove(previous)
    if projected() > budget:
        raise RouteError('Idle model cleanup has not completed; retry shortly', 503)


def main():
    routes_file = Path(sys.argv[1])
    config = json.loads(routes_file.read_text())
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    from omlx import server
    from omlx.cli import main as omlx_main

    streaming = {entry['expert_streaming']['directory']: entry['expert_streaming']
                 for entry in config['routes'].values() if 'expert_streaming' in entry}
    if streaming:
        from expert_streaming_mlx import install_loader
        install_loader(streaming)

    budget = config['budget_bytes']
    weights = {entry['model']: weight_reservation(entry) for entry in config['routes'].values()}

    async def switch(model, protected, active):
        pool = server.get_engine_pool()
        for route in config['routes'].values():
            if 'expert_streaming' in route:
                entry = pool.get_entry(route['model'])
                if entry is not None and entry.engine is None and not entry.is_loading:
                    # oMLX otherwise counts the entire on-disk expert bank.
                    # Update before it reserves memory for the first load.
                    entry.estimated_size = int(weights[route['model']])
        await reclaim_idle(pool, protected, active, weights, budget)

    server.app.add_middleware(AgentRouter, routes=config['routes'],
                              switch=switch, concurrency=config['concurrency'], budget_bytes=budget)
    omlx_main()


if __name__ == '__main__':
    main()
