"""Frozen public scalability inputs for developmental rolling Spike checks.

This adds no measurement authority and issues no full-suite qualification.
The existing public_scalability final gate remains the only suite issuer.
"""
from __future__ import annotations

from . import declared_bounds, public_scalability

#: Byte bound for a completed result this feedback pins. Large scalability cases write multi-MiB
#: results (7.7 MiB for the largest GEMM); the 2 MiB bound refused them, and the refusal failed
#: the whole grade. It is the floor: a case that embeds larger declared operands gets a bound
#: derived from them (:mod:`.declared_bounds`).
RESULT_MAX_BYTES = 32 << 20
#: Bound for this module's own small projection records.
RECORD_MAX_BYTES = 2 << 20


class Selection:
    def __init__(self, context):
        self.context = context
        self.selected = public_scalability.selected(context)
        self.cases, self.identity = {}, None
        if self.selected is not None:
            root, record = self.selected
            self.cases = {name: root / 'cases' / name for name in record['cases']}
            self.identity = {'root': str(root), 'record': record}

    def verify(self):
        if public_scalability.selected(self.context) != self.selected:
            raise ValueError('rolling public scalability selection changed')


class PerformanceFeedback:
    """Bounded observations of native products; never modifies numerical receipts.

    Each case's observation has its own budget (the selection's per-case
    seconds within the grade's timeout, :func:`.joint_performance.deadline`),
    counted from the case's own start, so compilation, grading and other cases'
    observations never consume it. The grade's completion callbacks are serialized, so a
    completed case's observation is handed to a few worker threads (each
    observation command still leases its CPU from the shared pool; a callback
    thread that holds a slot observes in place, so no slot is held while waiting)
    and :meth:`project` waits for them. One request-wide budget consumed serially
    under a lock left every later case of a large grade "budget exhausted"; the
    lock now guards only this object's bookkeeping. Feedback is projected
    only onto existing verdict rows after reopening its durable numerical
    receipt and the selected frozen suite.
    """
    def __init__(self, selection, *, compiler, runs_root, timeout, environment=None):
        from pathlib import Path
        import threading
        from . import joint_performance as P

        self.selection = selection
        self.compiler, self.runs_root = Path(compiler), Path(runs_root)
        self.environment = dict(P.environment() if environment is None else environment)
        self.timeout = timeout
        self.records, self.observed, self.bounds, self.started = {}, {}, {}, set()
        self.lock = threading.Lock()
        self.pool, self.pending = None, {}

    def completed(self, row):
        """Observe one completed case: in a worker thread when the caller holds no CPU slot."""
        import contextvars
        from concurrent.futures import ThreadPoolExecutor

        from merlin.common import cpu_pool

        from .joint_performance_batch import MAX_CONCURRENT

        name = row.get('capsule')
        if name not in self.selection.cases:
            return
        with self.lock:
            if name in self.records or name in self.started:
                raise ValueError('duplicate rolling performance observation')
            self.started.add(name)
            width = min(MAX_CONCURRENT, cpu_pool.compute_capacity())
            if width <= 1 or cpu_pool.current_thread_owns_cpu_lease():
                inline = True
            else:
                inline = False
                if self.pool is None:
                    self.pool = ThreadPoolExecutor(width, thread_name_prefix='rolling-observation')
                self.pending[name] = self.pool.submit(contextvars.copy_context().run, self._observe, name)
        if inline:
            self._observe(name)

    def wait(self):
        """Finish every dispatched observation; a failed one loses only its own case's projection."""
        with self.lock:
            pending, self.pending = dict(self.pending), {}
            pool, self.pool = self.pool, None
        for name, future in pending.items():
            try:
                future.result()
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                print(f'[first-grade] rolling performance observation unavailable for {name}: '
                      f'{type(error).__name__}: {str(error)[:160]}', flush=True)
        if pool is not None:
            pool.shutdown(wait=True)

    def _observe(self, name):
        import json
        from merlin.runtime import _native_files as F
        from merlin.targetgen.capsule_common import load_capsule
        from merlin.targetgen.capsule_runner import suite_for
        from .joint_performance import attach

        case_root = self.runs_root / 'runs' / suite_for(self.selection.context.target) / name
        result_path = case_root / 'capsule_result.json'
        capsule = load_capsule(self.selection.cases[name])
        bound = declared_bounds.result_bytes(capsule, RESULT_MAX_BYTES)
        with self.lock:
            self.bounds[name] = bound
        pin = F.pin(result_path, max_bytes=bound)
        result = json.loads(result_path.read_bytes())
        if result.get('capsule') != name:
            raise ValueError('rolling performance result belongs to another capsule')
        # The original ordinary grader and producer are the sole source of
        # execution/numerical evidence; callback metadata supplies none.
        projection = {}
        import time
        from .joint_performance import deadline, environment_scope
        with environment_scope(self.environment):
            attach(projection, case_root / 'generated', result=result,
                context=self.selection.context, compiler=self.compiler,
                capsule=capsule, remaining_s=max(0.0, deadline(self.timeout) - time.monotonic()))
        if F.pin(result_path, max_bytes=bound) != pin:
            raise ValueError('rolling numerical receipt changed during observation')
        # Existing attach emits these public projections. Do not copy native
        # consoles, private numerical data or qualification into feedback.
        feedback = {k: projection[k] for k in ('performance', 'compiler_diagnostics') if k in projection}
        if not feedback:
            return
        value = {'schema': 'merlin.rolling_scalability_performance.v1',
            'capsule': name, 'numerical_result': pin, 'feedback': feedback,
            'qualification_authority': False}
        output = self.runs_root / 'rolling_scalability_performance'
        output.mkdir(parents=True, exist_ok=True)
        path = output / (name + '.json')
        with path.open('x') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
        record = F.pin(path, max_bytes=RECORD_MAX_BYTES)
        with self.lock:
            self.records[name] = record
            self.observed[name] = result

    def project(self, verdict):
        """Attach feedback, leaving every original grade/count/tier untouched."""
        import json
        from pathlib import Path
        from merlin.runtime import _native_files as F

        self.wait()
        self.selection.verify()
        with self.lock:
            # Reopen all inputs before modifying any public verdict row.
            projections = {}
            for name, pin in self.records.items():
                if F.pin(pin['path'], max_bytes=RECORD_MAX_BYTES) != pin:
                    raise ValueError('rolling performance projection changed')
                value = json.loads(Path(pin['path']).read_bytes())
                result = value['numerical_result']
                if F.pin(result["path"], max_bytes=self.bounds.get(name, RESULT_MAX_BYTES)) != result:
                    # The score's host-compute veto rewrites a completed result after this
                    # observation (offload_census._record_veto): that row now fails, so it gets
                    # no performance feedback. Any other change still refuses the whole projection.
                    from merlin.targetgen.offload_census import is_recorded_veto
                    current = json.loads(Path(result['path']).read_bytes())
                    if not is_recorded_veto(self.observed.get(name), current):
                        raise ValueError('rolling numerical result changed before feedback publication')
                    continue
                projections[name] = value['feedback']
            for row in verdict.get('per_capsule', ()):
                row.update(projections.get(row.get('capsule'), {}))
