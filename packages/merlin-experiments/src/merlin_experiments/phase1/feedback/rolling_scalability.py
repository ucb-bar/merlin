"""Frozen public scalability inputs for developmental rolling Spike checks.

This adds no measurement authority and issues no full-suite qualification.
The existing public_scalability final gate remains the only suite issuer.

Timing a large case is its own lane (:class:`ObservationLane`): a grade publishes its numerical
verdict as soon as its numerical results are complete, projects only the observations already landed,
and the rest follow as a projection update of that same grade (:meth:`PerformanceFeedback.follow_up`).
Measured on a live run: once the compiler handled whole-model cases, each grade waited 3-4 minutes
past its last numerical result for advisory timing, and those minutes held a grade owner that the next
grade needed. Observations outlive the grade that dispatched them, lease their own CPU at most at the
:data:`~merlin.common.cpu_pool.PREPARATION` class, are superseded by a newer grade's observation of
the same case, and are reused while the case's executable and timing selection are unchanged.
"""
from __future__ import annotations

import threading

from . import declared_bounds, public_scalability

#: Byte bound for a completed result this feedback pins. Large scalability cases write multi-MiB
#: results (7.7 MiB for the largest GEMM); the 2 MiB bound refused them, and the refusal failed
#: the whole grade. It is the floor: a case that embeds larger declared operands gets a bound
#: derived from them (:mod:`.declared_bounds`).
RESULT_MAX_BYTES = 32 << 20
#: Bound for this module's own small projection records.
RECORD_MAX_BYTES = 2 << 20
SCHEMA = 'merlin.rolling_scalability_performance.v1'
_FAILURES = (OSError, ValueError, KeyError, TypeError, RuntimeError)


def _say(message):
    print('[first-grade] ' + message, flush=True)


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


class _Ticket:
    """One dispatched observation of one case; later grades with the same key follow it."""

    def __init__(self, feedback, name, key):
        self.feedback, self.name, self.key = feedback, name, key
        self.followers, self.finished, self.superseded = [], False, False
        self.future = None


class ObservationLane:
    """Advisory timing observations shared by every grade of one run.

    Holds no grade slot: a grade only dispatches here, and each observation command leases its
    own CPU from the shared pool. A newer grade's dispatch of a case with a different execution key
    supersedes the older one (cancelled if not started, its result discarded otherwise), so at most
    one observation per case is queued or running. A dispatch whose key equals the newest observation
    of that case (same executable bytes, declaration, timing selection and numerical screen) follows
    it instead of timing the case again.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.pool, self.futures = None, set()
        self.latest = {}  # case -> newest _Ticket
        self.done = {}  # case -> (key, record pin) of the newest landed observation
        self.busy = {}  # runs root -> observations still owed to it

    def dispatch(self, feedback, name, key):
        """Observe ``name`` for ``feedback``'s grade, follow an equal observation, or reuse a landed one."""
        import contextvars
        from concurrent.futures import ThreadPoolExecutor

        from merlin.common import cpu_pool

        from .joint_performance_batch import MAX_CONCURRENT

        with self.lock:
            held = self.done.get(name)
            if key is not None and held is not None and held[0] == key:
                reuse = held[1]
            else:
                reuse = None
                current = self.latest.get(name)
                if (key is not None and current is not None and current.key == key
                        and not current.superseded and not current.finished):
                    current.followers.append(feedback)
                    self._hold(feedback)
                    return 'following'
                if current is not None and not current.finished:
                    current.superseded = True
                    if current.future.cancel():
                        for owner in (current.feedback, *current.followers):
                            self._release(owner)
                            owner.dropped(name, 'superseded by a newer grade before it started')
                        current.followers = []
                ticket = self.latest[name] = _Ticket(feedback, name, key)
                if self.pool is None:
                    width = max(1, min(MAX_CONCURRENT, cpu_pool.compute_capacity() or 1))
                    self.pool = ThreadPoolExecutor(width, thread_name_prefix='rolling-observation')
                self._hold(feedback)
                ticket.future = self.pool.submit(contextvars.copy_context().run, self._run, ticket)
                self.futures = {future for future in self.futures if not future.done()} | {ticket.future}
        if reuse is not None:
            feedback.reuse(name, reuse)
            return 'reused'
        return 'observing'

    def _hold(self, feedback):
        root = str(feedback.runs_root)
        self.busy[root] = self.busy.get(root, 0) + 1

    def _release(self, feedback):
        root = str(feedback.runs_root)
        self.busy[root] -= 1
        if not self.busy[root]:
            del self.busy[root]

    def _run(self, ticket):
        from merlin.common.cpu_pool import PREPARATION, cpu_priority, lease_priority

        record = None
        try:
            # Never above a grade's numerical capsules: the less urgent of the dispatching grade's class
            # (a superseded grade already drains at PREPARATION) and PREPARATION.
            with cpu_priority(max(lease_priority(), PREPARATION)):
                record = ticket.feedback.observe(ticket.name)
        finally:
            self._finish(ticket, record)

    def _finish(self, ticket, record):
        with self.lock:
            ticket.finished = True
            followers, ticket.followers = ticket.followers, []
            superseded = ticket.superseded
            if record is not None and not superseded:
                self.done[ticket.name] = (ticket.key, record)
            self._release(ticket.feedback)
            for follower in followers:
                self._release(follower)
        ticket.feedback.landed(ticket.name, superseded=superseded)
        for follower in followers:
            if record is None or superseded:
                follower.dropped(ticket.name, 'the followed observation was unavailable or superseded')
            else:
                follower.reuse(ticket.name, record)

    def busy_runs(self):
        """Runs roots an observation still reads (their executables must stay in place)."""
        with self.lock:
            return frozenset(self.busy)

    def drain(self):
        """Finish every queued observation and its follow-up; the lane stays usable afterwards."""
        from concurrent.futures import wait

        while True:
            with self.lock:
                futures = set(self.futures)
                if not futures:
                    pool, self.pool = self.pool, None
                    break
            wait(futures)
            with self.lock:
                self.futures -= futures
        if pool is not None:
            pool.shutdown(wait=True)


_LANES = {}
_LANES_LOCK = threading.Lock()


def lane(run_dir):
    """The one observation lane of a run directory in this process."""
    from pathlib import Path

    key = str(Path(run_dir).resolve())
    with _LANES_LOCK:
        return _LANES.setdefault(key, ObservationLane())


def busy_runs(run_dir):
    """Runs roots of ``run_dir`` with advisory observations in flight (see :func:`lane`)."""
    from pathlib import Path

    with _LANES_LOCK:
        found = _LANES.get(str(Path(run_dir).resolve()))
    return found.busy_runs() if found is not None else frozenset()


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

    With a ``lane`` (:class:`ObservationLane`) the observations belong to the lane instead:
    :meth:`project` with ``wait=False`` projects what has landed, and :meth:`follow_up` hands every
    later observation of this grade to a publisher under the same checks. A follow-up that fails
    them is dropped with a log line; it never changes a count, status or tier.
    """
    lane = None  # without one, observations belong to this grade and project() waits for them

    def __init__(self, selection, *, compiler, runs_root, timeout, environment=None, lane=None):
        from pathlib import Path
        from . import joint_performance as P

        self.selection = selection
        self.compiler, self.runs_root = Path(compiler), Path(runs_root)
        self.environment = dict(P.environment() if environment is None else environment)
        self.timeout = timeout
        self.records, self.observed, self.bounds, self.started = {}, {}, {}, set()
        self.lock = threading.Lock()
        self.pool, self.pending = None, {}
        self.lane = lane
        self.published, self.refused, self.publisher = set(), None, None
        self.publish_lock = threading.Lock()

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
        if self.lane is not None:
            self.lane.dispatch(self, name, self.key(name))
            return
        with self.lock:
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
            except _FAILURES as error:
                _say(f'rolling performance observation unavailable for {name}: '
                     f'{type(error).__name__}: {str(error)[:160]}')
        if pool is not None:
            pool.shutdown(wait=True)

    def _case_root(self, name):
        from merlin.targetgen.capsule_runner import suite_for

        return self.runs_root / 'runs' / suite_for(self.selection.context.target) / name

    def key(self, name):
        """What an observation of ``name`` depends on, or None (never reused).

        The executable identity (:func:`.qa._execution_digest_from_result`: the bytes the case ran
        plus target and hardware revisions), the declaration, the timing selection and the original
        numerical screen the observation starts from. An owned host completion is timed through its
        own scope and is never reused.
        """
        import json
        from merlin.targetgen.capsule_common import load_capsule
        from . import joint_performance as P
        from .qa import _execution_digest_from_result

        try:
            with P.environment_scope(self.environment):
                choice = P.selected()
            path = self._case_root(name) / 'capsule_result.json'
            result = json.loads(path.read_bytes())
            digest = _execution_digest_from_result(path)
            if choice is None or digest is None or result.get('owned_host_native_execution') is not None:
                return None
            tier = (result.get('tiers') or {}).get('L2') or {}
            return (digest, P.capsule_identity(load_capsule(self.selection.cases[name])), choice[0]['sha256'],
                    (result.get('numeric') or {}).get('status'), tier.get('status'), tier.get('engine'))
        except _FAILURES:
            return None

    def observe(self, name):
        """Lane entry: observe and return this case's record pin, or None when unavailable."""
        try:
            self._observe(name)
        except _FAILURES as error:
            _say(f'rolling performance observation unavailable for {name}: '
                 f'{type(error).__name__}: {str(error)[:160]}')
            return None
        with self.lock:
            return self.records.get(name)

    def _pin_result(self, name):
        import json
        from merlin.runtime import _native_files as F
        from merlin.targetgen.capsule_common import load_capsule

        result_path = self._case_root(name) / 'capsule_result.json'
        capsule = load_capsule(self.selection.cases[name])
        bound = declared_bounds.result_bytes(capsule, RESULT_MAX_BYTES)
        with self.lock:
            self.bounds[name] = bound
        pin = F.pin(result_path, max_bytes=bound)
        result = json.loads(result_path.read_bytes())
        if result.get('capsule') != name:
            raise ValueError('rolling performance result belongs to another capsule')
        return capsule, bound, pin, result

    def _observe(self, name):
        from merlin.runtime import _native_files as F
        from .joint_performance import attach

        case_root = self._case_root(name)
        capsule, bound, pin, result = self._pin_result(name)
        # The original ordinary grader and producer are the sole source of
        # execution/numerical evidence; callback metadata supplies none.
        projection = {}
        import time
        from .joint_performance import deadline, environment_scope
        with environment_scope(self.environment):
            attach(projection, case_root / 'generated', result=result,
                context=self.selection.context, compiler=self.compiler,
                capsule=capsule, remaining_s=max(0.0, deadline(self.timeout) - time.monotonic()))
        if F.pin(case_root / 'capsule_result.json', max_bytes=bound) != pin:
            raise ValueError('rolling numerical receipt changed during observation')
        # Existing attach emits these public projections. Do not copy native
        # consoles, private numerical data or qualification into feedback.
        feedback = {k: projection[k] for k in ('performance', 'compiler_diagnostics') if k in projection}
        if feedback:
            self._record(name, pin, result, feedback)

    def _record(self, name, pin, result, feedback, **extra):
        import json
        from merlin.runtime import _native_files as F

        value = {'schema': SCHEMA, 'capsule': name, 'numerical_result': pin, 'feedback': feedback,
            'qualification_authority': False, **extra}
        output = self.runs_root / 'rolling_scalability_performance'
        output.mkdir(parents=True, exist_ok=True)
        path = output / (name + '.json')
        with path.open('x') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
        record = F.pin(path, max_bytes=RECORD_MAX_BYTES)
        with self.lock:
            self.records[name] = record
            self.observed[name] = result

    def reuse(self, name, record):
        """Project an earlier observation of the same executable onto this grade's own receipt."""
        import json
        from pathlib import Path
        from merlin.runtime import _native_files as F

        try:
            if F.pin(record['path'], max_bytes=RECORD_MAX_BYTES) != record:
                raise ValueError('reused rolling performance record changed')
            prior = json.loads(Path(record['path']).read_bytes())
            if prior.get('schema') != SCHEMA or prior.get('capsule') != name:
                raise ValueError('reused rolling performance record describes another case')
            _, _, pin, result = self._pin_result(name)
            self._record(name, pin, result, prior['feedback'],
                         reused_observation={'record': record, 'numerical_result': prior['numerical_result']})
        except _FAILURES as error:
            self.dropped(name, f'reuse unavailable: {type(error).__name__}: {str(error)[:120]}')
            return
        self.landed(name)

    def dropped(self, name, reason):
        _say(f'rolling performance observation for {name} dropped: {reason}')

    def landed(self, name, *, superseded=False):
        if superseded:
            with self.lock:
                self.records.pop(name, None)
            self.dropped(name, 'superseded by a newer grade')
            return
        self._publish_landed()

    def _projections(self, names):
        """Reopen every input of ``names``' feedback; a changed one refuses all of them."""
        import json
        from pathlib import Path
        from merlin.runtime import _native_files as F

        self.selection.verify()
        projections = {}
        with self.lock:
            for name in names:
                pin = self.records[name]
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
        return projections

    def project(self, verdict, *, wait=True):
        """Attach feedback, leaving every original grade/count/tier untouched.

        ``wait=False`` projects only the observations already landed; :meth:`follow_up` publishes the rest.
        """
        if wait:
            self.wait()
        with self.publish_lock:
            with self.lock:
                names = sorted(set(self.records) - self.published)
            # Reopen all inputs before modifying any public verdict row.
            projections = self._projections(names)
            for row in verdict.get('per_capsule', ()):
                row.update(projections.get(row.get('capsule'), {}))
            self.published |= set(names)

    def follow_up(self, publish):
        """Hand each later landed observation of this grade to ``publish(projections)``.

        ``publish`` runs in this caller's context (its snapshot publication owner), serialized per grade.
        """
        import contextvars

        self.publisher = (contextvars.copy_context(), publish)
        self._publish_landed()

    def _publish_landed(self):
        if self.publisher is None:
            return  # the grade has not published yet; :meth:`project` takes what landed by then
        context, publish = self.publisher
        with self.publish_lock:
            with self.lock:
                names = sorted(set(self.records) - self.published)
            if not names or self.refused is not None:
                return
            self.published |= set(names)
            try:
                projections = self._projections(names)
            except _FAILURES as error:
                # Custody failed: this grade publishes no further timing (as project refused before).
                self.refused = f'{type(error).__name__}: {str(error)[:160]}'
                _say(f'rolling performance follow-up refused: {self.refused}')
                return
            if not projections:
                return
            try:
                context.copy().run(publish, projections)
            except Exception as error:  # noqa: BLE001 -- advisory timing never fails a published grade
                _say(f'rolling performance follow-up unavailable: {type(error).__name__}: {str(error)[:160]}')
