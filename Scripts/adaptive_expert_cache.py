"""Bounded expert-cache policy, independent of MLX and checkpoint loading."""
import threading
import time
import weakref

MIB = 1024**2


class AdaptiveExpertCaches:
    """Resize only the caller's cache at its evaluated inference boundary.

    Lock order is store -> controller. Never acquire another store's lock.
    A shrink returns pool capacity only after entries have actually been evicted.
    """
    def __init__(self, pool_bytes, memory_ceiling, sample, clock=time.monotonic,
                 interval=2.0, step_bytes=64*MIB, initial_budgets=None):
        if pool_bytes <= 0 or memory_ceiling <= 0 or interval <= 0 or step_bytes <= 0:
            raise ValueError('Adaptive cache limits must be positive')
        self.initial_budgets = dict(initial_budgets or {})
        if any(v <= 0 for v in self.initial_budgets.values()) or sum(self.initial_budgets.values()) > pool_bytes:
            raise ValueError('Pool must fit initial budgets for all configured models')
        self.pool_bytes, self.memory_ceiling = pool_bytes, memory_ceiling
        self.sample, self.clock = sample, clock
        self.interval, self.step_bytes = interval, step_bytes
        self.lock = threading.RLock()
        self.entries = weakref.WeakKeyDictionary()
        self.last_sample = float('-inf')
        self.pressure = 'unknown'
        self.growth_since_sample = 0
        self.used_bytes = self.available_bytes = None

    def _committed(self):
        loaded = {s['name']:s['budget'] for s in self.entries.values()}
        return sum(max(loaded.get(name, 0), self.initial_budgets.get(name, 0))
                   for name in loaded.keys() | self.initial_budgets.keys())

    def register(self, store, name, minimum, maximum):
        with self.lock:
            if store in self.entries:
                raise ValueError('Expert cache is already registered')
            if not 0 < minimum <= store.cache.budget <= maximum:
                raise ValueError('Expected minimum <= initial expert cache <= maximum')
            if any(s['name'] == name for s in self.entries.values()):
                raise ValueError('Expert model is already registered')
            available = self.pool_bytes - self._committed() + self.initial_budgets.get(name, 0)
            if available < minimum:
                raise ValueError('Adaptive expert-cache pool is exhausted')
            store.cache.resize(min(store.cache.budget, available))
            self.entries[store] = dict(name=name, minimum=minimum, maximum=maximum,
                budget=store.cache.budget, bytes=store.cache.bytes, hits=store.cache.hits,
                misses=store.cache.misses, evictions=store.cache.evictions,
                previous_hits=store.cache.hits, previous_misses=store.cache.misses,
                last_adjustment=self.clock(), reason='initial')

    def unregister(self, store):
        with self.lock:
            self.entries.pop(store, None)

    def _pressure(self, now):
        if now - self.last_sample < self.interval:
            return
        self.last_sample = now
        self.growth_since_sample = 0
        try:
            used, available, total = self.sample()
            if used < 0 or available < 0 or total <= 0 or available > total:
                raise ValueError('Invalid memory sample')
            self.used_bytes, self.available_bytes = used, available
            if used >= self.memory_ceiling * .85 or available <= total * .10:
                self.pressure = 'high'
            elif used <= self.memory_ceiling * .70 and available >= total * .20:
                self.pressure = 'low'
            else:
                self.pressure = 'hold'
        except Exception:
            # Telemetry failure never grants more memory.
            self.pressure = 'unavailable'
            self.used_bytes = self.available_bytes = None

    def boundary(self, store):
        """Caller holds store.lock and has evaluated all expert-dependent work."""
        with self.lock:
            state = self.entries.get(store)
            if state is None:
                return
            now = self.clock()
            if now - state['last_adjustment'] < self.interval:
                return
            self._pressure(now)
            cache = store.cache
            hits = cache.hits - state['previous_hits']
            misses = cache.misses - state['previous_misses']
            target, reason = cache.budget, 'hold'
            if self.pressure in ('high', 'unavailable'):
                target = max(state['minimum'], cache.budget // 2)
                reason = 'memory-pressure' if self.pressure == 'high' else 'telemetry-unavailable'
            elif self.pressure == 'low' and hits + misses >= 32 and misses / (hits + misses) >= .10:
                free = self.pool_bytes - self._committed() + max(0,
                    self.initial_budgets.get(state['name'], 0) - cache.budget)
                # Leave space for the cache margin used by admission too.
                headroom = max(0, int((self.memory_ceiling * .70 - self.used_bytes) / 1.05) - self.growth_since_sample)
                growth = min(self.step_bytes, free, headroom, state['maximum'] - cache.budget)
                target += growth
                self.growth_since_sample += growth
                reason = 'cache-misses' if growth else 'pool-or-model-limit'
            if target != cache.budget:
                cache.resize(target)
            state.update(budget=cache.budget, previous_hits=cache.hits,
                         previous_misses=cache.misses, last_adjustment=now, reason=reason)
            state.update(bytes=cache.bytes, hits=cache.hits, misses=cache.misses, evictions=cache.evictions)

    def snapshot(self):
        with self.lock:
            return {'pool_bytes': self.pool_bytes, 'pressure': self.pressure,
                    'sampled_used_bytes': self.used_bytes, 'sampled_available_bytes': self.available_bytes,
                    'allocated_budget_bytes': sum(s['budget'] for s in self.entries.values()),
                    'committed_pool_bytes': self._committed(),
                    'models': {s['name']: {k:s[k] for k in ('minimum','maximum','budget','bytes','hits','misses','evictions','reason')}
                               for s in self.entries.values()}}
