"""Utility functions for recurrent policy gradients."""

import time

from contextlib import contextmanager


@contextmanager
def elapsed_timer():
    t_start = time.perf_counter()
    elapsed = lambda: time.perf_counter() - t_start
    yield lambda: elapsed()
    t_end = time.perf_counter()
    elapsed = lambda: t_end - t_start
