"""In-run operator grades through a pool of helper interpreters (``MERLIN_GRADE_PROCS``).

One controller process serializes every operator capsule's parent-side custody preparation and
result verification behind its process-wide snapshot admission. This module keeps the scheduler,
streaming completion callbacks, deferral and tier pricing in the controller, and moves only the
per-capsule ``operator_child.run_document`` call (prepare -> fresh sealed child -> verify ->
publish) into N helper interpreters. Each helper:

* is a fresh interpreter (spawn, never fork: the controller owns threads, flocks and memfds);
* reopens the controller's original owners exactly as ``joint_grade_shards`` does (compiler
  library, shared-pass requirement, ``compiler_execution`` on the same candidate copy), rebuilds
  the same selected simulator adapters, and runs ``operator_child.validate`` -- so it verifies the
  bundle snapshot itself and keeps every per-capsule custody check of the in-process route;
* serves only after a handshake: its adapter selection, compiler source/snapshot/profile identity
  must equal the controller's, otherwise it is not used.

The controller passes the same child document it would have run itself, applies the returned tier
costs/spend to its own ``tier_policy``, and forwards its scheduler deferral reason through a
per-request cancel file. ``MERLIN_GRADE_PROCS`` 0/1 (or unset in a helper) keeps the original
single-process route; any pool failure falls back to it for that and all later capsules.
Helper logs live under the run's ``_qa_work`` (never the agent workspace); nothing here reads
hidden capsule data.
"""

from __future__ import annotations

import contextlib
import contextvars
import itertools
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ENV = "MERLIN_GRADE_PROCS"
DEFAULT_PROCS = 8
_SCHEMA = "merlin.operator_grade_pool.v1"
_STARTUP_S = float(os.environ.get("MERLIN_GRADE_PROCS_STARTUP_S", "900"))
_POLL_S = 5.0  # the original child wait polls its deferral probe at the same cadence
_KILL_WAIT_S = 10
_POOL: contextvars.ContextVar = contextvars.ContextVar("merlin_operator_grade_pool", default=None)


class PoolUnavailable(Exception):
    """The pool cannot serve this request; the caller runs the original in-process route."""


class RemoteGradeError(RuntimeError):
    """A capsule-level exception raised by the original route inside a helper."""


def configured(environ=None) -> int:
    raw = (os.environ if environ is None else environ).get(ENV, str(DEFAULT_PROCS)).strip()
    if not raw.isascii() or not raw.isdecimal():
        return 0
    count = int(raw)
    return count if count > 1 else 0


def _log(message):
    print(f"[grade-pool] {message}", file=sys.stderr, flush=True)


def _identity():
    """Original executor identity both sides must share (no candidate or hidden content)."""
    from .compiler_profile_runtime import active_original_execution

    executor = active_original_execution()
    if executor is None:
        return None
    owner = executor.owner
    profile = getattr(owner, "profile", None)
    return {
        "source_sha256": executor.source_sha256,
        "candidate": str(executor.candidate),
        "snapshot_digest": getattr(owner, "snapshot_digest", None),
        "profile_sha256": profile.get("sha256") if isinstance(profile, dict) else None,
    }


def _helper_command(spec_path):
    return [sys.executable, "-m", __name__, str(spec_path)]


def _rethrow(reply):
    error = reply.get("error") or {}
    kind, message = error.get("type"), error.get("message", "")
    if kind == "StageGateError":
        from merlin_experiments.phase2.contracts import StageGateError

        raise StageGateError(message)
    if kind == "TimeoutExpired":
        raise subprocess.TimeoutExpired(cmd="operator grade child", timeout=error.get("timeout") or 0)
    raise RemoteGradeError(f"{kind}: {message}")


class _Helper:
    def __init__(self, index, process):
        self.index, self.process = index, process
        self.write_lock = threading.Lock()
        self.ready = threading.Event()
        self.hello = None
        self.pending = {}
        self.inflight = 0
        self.dead = False
        self.reader = threading.Thread(target=self._read, name=f"grade-pool-h{index:02d}", daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if message.get("op") == "ready":
                    self.hello = message
                    self.ready.set()
                elif message.get("op") in ("done", "error"):
                    slot = self.pending.get(message.get("id"))
                    if slot is not None:
                        slot[1] = message
                        slot[0].set()
        finally:
            self.dead = True
            self.ready.set()
            for slot in list(self.pending.values()):
                slot[0].set()

    def send(self, message):
        data = (json.dumps(message) + "\n").encode()
        with self.write_lock:
            self.process.stdin.write(data)
            self.process.stdin.flush()

    def alive(self):
        return not self.dead and self.process.poll() is None


class GradePool:
    def __init__(self, *, inputs, candidate, run_dir, processes):
        self.inputs, self.candidate, self.run_dir = inputs, Path(candidate), Path(run_dir)
        self.processes = processes
        self.lock = threading.Lock()
        self.helpers = []
        self.started = False
        self.broken = None
        self.directory = None
        self.ids = itertools.count()
        self.served = 0

    # ---------------------------------------------------------------- lifecycle
    def _spec(self, document):
        context, inputs = self.inputs.context, self.inputs
        # compiler_execution reads only the profile service and descriptor; never resolve a callable root here.
        public_root = None if callable(inputs.public_root) else inputs.public_root
        return {
            "schema": _SCHEMA,
            "run_dir": str(self.run_dir),
            "candidate": str(self.candidate),
            "descriptor": str(context.descriptor),
            "repo": str(context.repo),
            "harness": None if context.harness is None else str(context.harness),
            "readback_policy": context.readback_policy,
            "candidate_policy": None if context.candidate_policy is None else str(context.candidate_policy),
            "arm": inputs.arm,
            "public_root": None if public_root is None else str(public_root),
            "contract": None if inputs.contract is None else str(inputs.contract),
            "additional_forbidden": list(inputs.additional_forbidden),
            "target": document["target"],
            "adapters": document["adapters"],
            "identity": _identity(),
        }

    def _start(self, document):
        spec = self._spec(document)
        if spec["identity"] is None:
            raise PoolUnavailable("no original executor in this grade")
        base = self.run_dir / "_qa_work" / "grade_pool"
        base.mkdir(parents=True, exist_ok=True)
        self.directory = base / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}-{id(self):x}"
        (self.directory / "cancel").mkdir(parents=True)
        spec["cancel_dir"] = str(self.directory / "cancel")
        spec_path = self.directory / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True))
        spec_path.chmod(0o600)
        environment = {k: v for k, v in os.environ.items() if k != "MERLIN_JOINT_GRADE_PROCESSES"}
        environment[ENV] = "0"  # a helper never pools again
        started = time.monotonic()
        for index in range(self.processes):
            log = (self.directory / f"h{index:02d}.log").open("ab")
            try:
                process = subprocess.Popen(
                    _helper_command(spec_path),
                    cwd=os.getcwd(),
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    start_new_session=True,
                )
            finally:
                log.close()
            self.helpers.append(_Helper(index, process))
        deadline = started + _STARTUP_S
        for helper in self.helpers:
            helper.ready.wait(max(0.0, deadline - time.monotonic()))
        live = [h for h in self.helpers if h.alive() and h.hello is not None]
        for helper in self.helpers:
            if helper not in live:
                _terminate(helper.process)
        self.helpers = live
        _log(f"pool up: {len(live)}/{self.processes} helpers in {time.monotonic() - started:.0f}s "
             f"dir={self.directory} identity={spec['identity']['source_sha256']}/"
             f"{str(spec['identity']['snapshot_digest'])[:12]} pids={[h.process.pid for h in live]}")
        if not live:
            raise PoolUnavailable("no helper completed its handshake")

    def close(self):
        with self.lock:
            helpers, self.helpers = self.helpers, []
        for helper in helpers:
            with contextlib.suppress(OSError, ValueError):
                helper.process.stdin.close()
        deadline = time.monotonic() + _KILL_WAIT_S
        for helper in helpers:
            with contextlib.suppress(subprocess.TimeoutExpired):
                helper.process.wait(timeout=max(0.1, deadline - time.monotonic()))
            if helper.process.poll() is None:
                _terminate(helper.process)
        if helpers:
            _log(f"pool closed: served={self.served} broken={self.broken}")

    def _break(self, reason):
        if self.broken is None:
            self.broken = reason
            _log(f"pool unavailable, in-process route from here: {reason}")

    def _choose(self, document):
        with self.lock:
            if self.broken is not None:
                raise PoolUnavailable(self.broken)
            if not self.started:
                self.started = True
                try:
                    self._start(document)
                except Exception as exc:  # noqa: BLE001 -- any failure keeps the original route
                    self._break(f"start failed: {type(exc).__name__}: {exc}")
                    raise PoolUnavailable(self.broken) from exc
            live = [h for h in self.helpers if h.alive()]
            if not live:
                self._break("no live helper")
                raise PoolUnavailable(self.broken)
            helper = min(live, key=lambda h: h.inflight)
            helper.inflight += 1
            return helper

    # ---------------------------------------------------------------- one request
    def run(self, document):
        from merlin.targetgen.capsule_scheduling import active_defer_probe

        helper = self._choose(document)
        request = f"{os.getpid()}-{next(self.ids)}"
        slot = [threading.Event(), None]
        helper.pending[request] = slot
        cancel = self.directory / "cancel" / request
        probe = active_defer_probe()
        try:
            try:
                helper.send({"op": "job", "id": request, "document": document, "deferrable": probe is not None})
            except (OSError, ValueError) as exc:
                self._break(f"helper {helper.index} send failed: {exc}")
                raise PoolUnavailable(self.broken) from exc
            while not slot[0].wait(_POLL_S):
                if probe is not None and not cancel.exists() and (reason := probe()):
                    temporary = cancel.with_name(f".{request}.tmp")
                    temporary.write_text(reason, encoding="utf-8")
                    os.replace(temporary, cancel)
                if not helper.alive():
                    break
            reply = slot[1]
            if reply is None:
                self._break(f"helper {helper.index} exited rc={helper.process.poll()} mid-capsule")
                raise PoolUnavailable(self.broken)
            if reply.get("op") == "error":
                if reply.get("pool"):
                    self._break(f"helper {helper.index}: {reply.get('error')}")
                    raise PoolUnavailable(self.broken)
                _rethrow(reply)
            with self.lock:
                self.served += 1
            return reply["row"], dict(reply.get("costs") or {}), _spend(reply)
        finally:
            helper.pending.pop(request, None)
            with self.lock:
                helper.inflight -= 1
            with contextlib.suppress(OSError):
                cancel.unlink()


def _spend(reply):
    from .operator_child import _NO_SPEND

    return _NO_SPEND if reply.get("no_spend") else reply.get("spend")


def _terminate(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, sig)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_KILL_WAIT_S)
            return


@contextlib.contextmanager
def scoped(inputs, *, candidate, run_dir):
    """Offer the helper pool to operator grades started inside this scope (lazily started)."""
    from .operator_child import enabled

    count = configured()
    if not count or not enabled() or getattr(inputs, "compiler_profile_service", None) is None:
        yield None
        return
    pool = GradePool(inputs=inputs, candidate=candidate, run_dir=run_dir, processes=count)
    token = _POOL.set(pool)
    try:
        yield pool
    finally:
        _POOL.reset(token)
        pool.close()


def dispatch(document):
    """(row, costs, spend) from a helper, or None when this grade has no usable pool."""
    pool = _POOL.get()
    if pool is None:
        return None
    try:
        return pool.run(document)
    except PoolUnavailable:
        return None


# ------------------------------------------------------------------------------------ the helper


def _rebuild_adapters(selection, target):
    from merlin.targetgen import capsule_runner as runner
    from merlin.targetgen.contract.readback_policy import ReadbackPolicy

    return {
        tier: runner.simulator_adapter(
            record["engine"],
            target,
            selection=record["selection"],
            readback_policy=(
                ReadbackPolicy.from_record(record["readback_policy"]) if record["readback_policy"] is not None else None
            ),
        )
        for tier, record in selection.items()
    }


def _serve(spec, proto):
    from merlin.targetgen import capsule_runner as runner
    from merlin.targetgen import capsule_scheduling as CS

    from ..context import load_context
    from ..joint_grade_shards import original_selection
    from . import operator_child as OC
    from .compiler_profile_runtime import functional_service
    from .loop_grading import GradingInputs, compiler_execution

    runner._die_with_parent()
    write_lock = threading.Lock()

    def emit(message):
        with write_lock:
            proto.write(json.dumps(message) + "\n")
            proto.flush()

    run_dir, candidate, target = Path(spec["run_dir"]), Path(spec["candidate"]), spec["target"]
    context = load_context(
        spec["descriptor"],
        repo=spec["repo"],
        harness=spec["harness"],
        readback_policy=spec["readback_policy"],
        candidate_policy=spec["candidate_policy"],
    )
    inputs = GradingInputs(
        context=context,
        arm=spec["arm"],
        public_root=None if spec["public_root"] is None else Path(spec["public_root"]),
        policy_roots=(),
        contract=None if spec["contract"] is None else Path(spec["contract"]),
        additional_forbidden=tuple(spec["additional_forbidden"]),
        compiler_profile_service=functional_service(run_dir),
    )
    cancel_dir = Path(spec["cancel_dir"])
    threads = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(original_selection(run_dir))
        stack.enter_context(compiler_execution(inputs, candidate=candidate, run_dir=run_dir))
        adapters = _rebuild_adapters(spec["adapters"], target)
        selection = OC.validate(target=target, adapters=adapters, config=None, perf_extractor=None)
        identity = _identity()
        if selection != spec["adapters"] or identity != spec["identity"]:
            raise RuntimeError(f"helper handshake mismatch: identity={identity} expected={spec['identity']} "
                               f"adapters_equal={selection == spec['adapters']}")
        base = contextvars.copy_context()
        emit({"op": "ready", "pid": os.getpid(), "identity": identity})
        print(f"[grade-pool helper {os.getpid()}] ready identity={identity}", file=sys.stderr, flush=True)

        def job(message):
            request, document = message["id"], message["document"]
            try:
                if document.get("adapters") != selection or document.get("target") != target:
                    emit({"op": "error", "id": request, "pool": True, "error": "document selection differs from helper"})
                    return
                cancel = cancel_dir / request

                def probe():
                    try:
                        return cancel.read_text(encoding="utf-8") or None
                    except OSError:
                        return None

                token = CS._DEFER_PROBE.set(probe if message.get("deferrable") else None)
                try:
                    with runner._original_capsule_worker_scope():
                        row, costs, spend = OC.run_document(document, adapters=adapters, selection=selection)
                finally:
                    CS._DEFER_PROBE.reset(token)
                no_spend = spend is OC._NO_SPEND
                emit({"op": "done", "id": request, "row": row, "costs": costs,
                      "spend": None if no_spend else spend, "no_spend": no_spend})
            except BaseException as exc:  # noqa: BLE001 -- the controller re-raises it for this capsule
                emit({"op": "error", "id": request, "error": {
                    "type": type(exc).__name__, "message": str(exc)[:4000],
                    "timeout": getattr(exc, "timeout", None)}})

        for line in sys.stdin:
            message = json.loads(line)
            if message.get("op") != "job":
                continue
            thread = threading.Thread(target=base.copy().run, args=(job, message), daemon=True)
            thread.start()
            threads.append(thread)
            threads = [t for t in threads if t.is_alive()]
        for thread in threads:
            thread.join()
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(f"usage: python -m {__name__} SPEC.json", file=sys.stderr)
        return 64
    spec = json.loads(Path(argv[0]).read_text())
    if spec.get("schema") != _SCHEMA:
        return 65
    # The protocol owns the original stdout; everything printed by graders goes to the helper log.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        return _serve(spec, proto)
    except BaseException:  # noqa: BLE001 -- the controller sees EOF and falls back
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
