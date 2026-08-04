"""One lock serializing in-process MLIR parse/lowering, because xDSL's parser is not thread-safe.

Two whole-model builds running in the same process — the obvious way to use a 48-core host for a set
of images that are otherwise independent — corrupt each other inside the parser. The failure is not a
crash in shared state you can see; it surfaces as a bogus *parse error* on perfectly valid IR::

    xdsl.utils.exceptions.ParseError: <unknown>:406:5
        %618 = linalg.matmul {prov.region_id = "matmul_0", ...} ins(...) outs(...)
        ^^^^
        Could not build linalg op

measured with three concurrent ``build_app`` calls: one succeeded and two died there, at three
different line numbers, on modules that parse fine on their own. So the hazard is per-parse global
state in the dialect/op construction path, not anything in this repo, and it cannot be fixed by
passing a fresh context.

The cost of serializing is small and known: lowering a whole model is tens of seconds, while the spike
run that follows it is minutes to tens of minutes (spectformer is 2.78 G cycles). Holding this lock
across parse+lower and releasing it before the compile/link/simulate subprocesses keeps the part that
actually dominates wall clock fully parallel.

Use it around *any* in-process MLIR work that could run concurrently::

    from ...common.ir_lock import IR_LOCK
    with IR_LOCK:
        prepared, features = prepare_for_lowering(...)
        res = lower_model_file(prepared, ...)

It is an ``RLock`` so a nested helper that also takes it does not deadlock.
"""
from __future__ import annotations

import threading

#: Held for the duration of parse/lowering. Module-level, so every importer shares one lock.
IR_LOCK = threading.RLock()
