"""Ordered bounded CPU loading, no pixel/model policy.

Uses two CPU workers. The
consumer owns one current item; at most two further futures own decoded items.
Close the context on early exit so pending work is cancelled and workers joined.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import time


@contextmanager
def ordered_decode(rows, loader, *, enabled, stats=None):
    """Yield (row, loaded) in order; loader alone executes in CPU threads.

    Failures surface at their original ordinal. Already-running successor
    reads may finish during shutdown; no new work is submitted after failure.
    Worker startup, initial reads and join occur inside this context.
    """
    stats = stats if stats is not None else {}
    stats.update(submitted=0, consumed=0, maxPending=0, consumerWaitSeconds=0.,
                 joinSeconds=0., workersJoined=False)
    if not enabled:
        def serial():
            for row in rows:
                stats['submitted'] += 1
                value = loader(row)
                stats['consumed'] += 1
                yield row, value
                del value
        iterator = serial()
        try:
            yield iterator
        finally:
            iterator.close()
            stats['workersJoined'] = True
        return
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='flex-ordered-decode')
    pending = deque()
    source = iter(rows)
    def submit():
        try:
            row = next(source)
        except StopIteration:
            return
        pending.append((row, executor.submit(loader, row)))
        stats['submitted'] += 1
        stats['maxPending'] = max(stats['maxPending'], len(pending))
        if len(pending) > 2:
            raise RuntimeError('prefetch bound exceeded')
    def ordered():
        submit(); submit()
        while pending:
            row, future = pending.popleft()
            waited = time.perf_counter()
            value = future.result()
            stats['consumerWaitSeconds'] += time.perf_counter() - waited
            del future
            stats['consumed'] += 1
            submit()
            yield row, value
            del value
    iterator = ordered()
    try:
        yield iterator
    finally:
        iterator.close()
        for _, future in pending:
            future.cancel()
        started = time.perf_counter()
        executor.shutdown(wait=True, cancel_futures=True)
        stats['joinSeconds'] = time.perf_counter() - started
        pending.clear()
        stats['workersJoined'] = True
