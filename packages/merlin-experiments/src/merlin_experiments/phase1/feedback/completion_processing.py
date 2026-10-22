"""Bounded request-owned processing of already durable capsule completions.

This queue owns no grading or CPU lease. The caller must nest it inside the
original compiler execution lifetime, so all accepted work drains before that
custody closes. Native source/model checks still happen before submission.
"""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy


class CompletionProcessing:
    def __init__(self, process, *, max_items=2048):
        if type(max_items) is not int or not 1 <= max_items <= 2048:
            raise ValueError("completion processing requires a bounded cohort")
        self.process = process
        self.max_items = max_items
        self._executor = None
        self._pending = []
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("completion processing cannot be reopened")
        self._entered = True
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grade-completions")
        return self

    def submit(self, row):
        if self._executor is None:
            raise RuntimeError("completion processing is not open")
        if len(self._pending) >= self.max_items:
            # Refuse instead of waiting while the native scheduler needs to
            # refill. The original durable row remains independently available.
            raise ValueError("completion processing exceeded its bounded cohort")
        frozen = deepcopy(row)
        context = copy_context()
        self._pending.append(self._executor.submit(context.run, self.process, frozen))

    def __exit__(self, kind, error, traceback):
        executor, self._executor = self._executor, None
        executor.shutdown(wait=True, cancel_futures=False)
        failures = []
        for future in self._pending:
            try:
                future.result()
            except BaseException as failure:
                failures.append(failure)
        if failures:
            if error is not None:
                error.add_note(f"{len(failures)} completion processing operation(s) also failed")
            else:
                raise RuntimeError(
                    f"{len(failures)} completion processing operation(s) failed after draining all accepted rows"
                ) from failures[0]
        return False
