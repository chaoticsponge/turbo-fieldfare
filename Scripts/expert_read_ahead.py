"""One-expert asynchronous file read, with a shared byte reservation.

Workers return ordinary bytes only. Decoding/MLX evaluation stays on the caller.
"""
from concurrent.futures import ThreadPoolExecutor
import threading


class ReadAheadPool:
    def __init__(self, limit):
        if type(limit) is not int or limit <= 0:
            raise ValueError('Read-ahead pool must have a positive byte limit')
        self.limit, self.reserved, self.peak = limit, 0, 0
        self.submitted = self.consumed = self.skipped = 0
        self.lock = threading.Lock()

    def acquire(self, size):
        with self.lock:
            if size <= 0 or self.reserved + size > self.limit:
                self.skipped += 1
                return False
            self.reserved += size
            self.peak = max(self.peak, self.reserved)
            self.submitted += 1
            return True

    def release(self, size, consumed=False):
        with self.lock:
            self.reserved -= size
            if consumed:
                self.consumed += 1

    def snapshot(self):
        with self.lock:
            return {'limit_bytes':self.limit, 'reserved_bytes':self.reserved,
                    'peak_reserved_bytes':self.peak, 'submitted':self.submitted,
                    'consumed':self.consumed, 'skipped_for_budget':self.skipped}


class ExpertReadAhead:
    """Methods are serialized by the owning ExpertStore lock.

    Account twice the payload: one expert plus transient chunks used by pread.
    At most one future exists, including completed-but-unconsumed data.
    """
    def __init__(self, pool):
        self.pool = pool
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='expert-pread')
        self.pending = None
        self.closed = False

    def schedule(self, key, payload_bytes, read):
        if self.closed or self.pending is not None:
            return False
        reserved = 2 * payload_bytes
        if not self.pool.acquire(reserved):
            return False
        try:
            self.pending = (key, self.executor.submit(read), reserved)
        except BaseException:
            self.pool.release(reserved)
            raise
        return True

    def consume(self, key, read, decode):
        if self.closed:
            raise RuntimeError('Read-ahead worker has been closed')
        if self.pending is None or self.pending[0] != key:
            self.discard()
            return decode(read())
        _, future, reserved = self.pending
        self.pending = None
        payload = None
        consumed = False
        try:
            payload = future.result()  # Failed pread is surfaced, never hidden.
            result = decode(payload)
            consumed = True
            return result
        finally:
            payload = future = None
            self.pool.release(reserved, consumed)

    def discard(self):
        if self.pending is None:
            return
        _, future, reserved = self.pending
        self.pending = None
        try:
            if not future.cancel():
                try:
                    future.result()
                except Exception:
                    pass  # Unused speculative data; no answer consumed it.
        finally:
            future = None
            self.pool.release(reserved)

    def close(self):
        if not self.closed:
            self.discard()
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.closed = True
