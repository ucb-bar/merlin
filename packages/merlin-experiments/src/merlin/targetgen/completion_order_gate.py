"""The completion-order check as the capsule ladder runs it: L2 structural evidence.

A numerical pass on the functional model (the L2 tier) says the program computes the declared
operation when every accelerator command completes in program order. :mod:`merlin.targetgen.
completion_order` asks the question that pass cannot: does the program depend on that order where
the hardware does not provide it? It reads the command stream the functional run actually executed
and the target's RTL-derived ordering facts, so it decides at L2, before any RTL engine runs, and a
flagged pair fails the capsule there with the commands and rows that race.

* ``hazard``: the functional tier is failed with :func:`merlin.targetgen.completion_order.reason`.
* ``clear``: recorded; also tells the opt-in memory-perturbation sweep it has nothing to look for.
* ``unknown``: recorded with every reason (facts not derived, an opaque command, an unmodeled
  configuration); never a pass by omission, never a failure of a program the facts cannot judge.
* ``deferred``: a ladder that reaches no RTL tier outside :func:`certification`. The check certifies
  rather than steers development, so it runs where certification happens: any ladder that reaches an
  RTL tier, and :func:`after_coverage`, the final grade's best-effort pass over its published passes.
  The row says so instead of reading as unknown or failed.

The verdict is a property of the executable's bytes and the facts, so it is cached under both
digests and a re-grade of unchanged bytes does not re-run the commit-log trace.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
from pathlib import Path
from typing import Any

#: Set to ``off`` to skip the check; the result then records that it was skipped and why.
SWITCH_ENV = "MERLIN_COMPLETION_ORDER_CHECK"
TIMEOUT_ENV = "MERLIN_COMPLETION_ORDER_TIMEOUT_S"
#: Set to ``off`` to run the check in the calling process instead of a worker process.
WORKER_ENV = "MERLIN_COMPLETION_ORDER_WORKER"
_CACHE_NAMESPACE = "completion_order"
#: Set inside :func:`certification`; the grade's worker threads inherit it with their context.
_CERTIFYING: contextvars.ContextVar[bool] = contextvars.ContextVar("completion_order_certifying", default=False)
DEFERRED_DETAIL = "deferred to certification: this grade reaches no RTL tier; the final grade checks it"
#: Seconds a worker may run past the check's own engine deadline before it is stopped.
_WORKER_GRACE_S = 300.0
_WORKER_BOOTSTRAP = (
    "import json, sys; sys.path[:] = json.loads(sys.argv[1]); "
    "from merlin.targetgen.completion_order_gate import _worker_main; _worker_main()"
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checker_digest() -> str:
    from merlin.perf import command_recorder, executed_trace, streamed_executed_trace
    from merlin.targetgen import completion_order, rtl_checks_generic, semantic_families
    from merlin.targetgen.rocc import decode, semantics

    readers = (
        completion_order,
        executed_trace,
        streamed_executed_trace,
        command_recorder,
        rtl_checks_generic,
        semantic_families,
        semantics,
        decode,
    )
    pins = {m.__name__: _sha(Path(m.__file__)) for m in readers}
    pins["command_recorder_source"] = _sha(command_recorder.SOURCE)
    pins[__name__] = _sha(Path(__file__))
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _semantic_environment(environment: dict[str, str]) -> dict[str, Any]:
    """Keep unknown settings, normalize existing file operands, and omit known output locations.

    Environment settings can change extension execution, so excluding the whole environment is
    unsafe. Only the infrastructure's output/scheduling locations are nonsemantic here. Library
    search directories retain their order and sonames with actual bytes, rather than their location.
    """
    from merlin.perf.command_recorder import INNER_ENV, LIMIT_ENV, OUT_ENV, SELECTION_ENV

    outputs = {
        "PWD", "OLDPWD", "TMPDIR", "TMP", "TEMP", "MERLIN_OUT_ROOT", "MERLIN_BUNDLE_CAS",
        "MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_COMPUTE_CPUS", "MERLIN_GRADE_CPUS",
        "MERLIN_PREPARATION_CPUS", "MERLIN_CPU_PRIORITY", SELECTION_ENV, OUT_ENV, INNER_ENV, LIMIT_ENV,
    }
    identity = {}
    for key, value in sorted(environment.items()):
        if key in outputs:
            continue
        if key == "LD_LIBRARY_PATH":
            directories = []
            for name in value.split(os.pathsep):
                directory = Path(name or ".")
                if not directory.is_dir():
                    directories.append({"missing": True})
                    continue
                directories.append({p.name: _sha(p) for p in sorted(directory.glob("*.so*")) if p.is_file()})
            identity[key] = directories
        elif key in {"LD_PRELOAD", "LD_AUDIT"}:
            # Nonabsolute loader tokens require the loader's resolution rules; preserve them exactly.
            identity[key] = [
                {"file": _sha(Path(token))} if Path(token).is_absolute() and Path(token).is_file() else token
                for token in value.replace(os.pathsep, " ").split()
            ]
        else:
            try:
                file = Path(value)
                if value and file.is_absolute() and file.is_file():
                    identity[key] = {"file": _sha(file)}
                    continue
            except OSError:
                pass
            identity[key] = value
    return identity


def _observation_identity(target: str, elf: Path) -> dict[str, Any] | None:
    """Reopen the actual decoder and selected functional engine before reusing a trace verdict.

    No resolvable engine identity means no persistent cache, rather than an identity by target name.
    The normal invocation explicitly names its accelerator receiver via --extlib.

    Engine, explicit file arguments and library search entries use content digests. The ELF's build
    directory and known output/lease locations do not change execution. Unknown environment settings
    remain in the key, since they may affect the receiver's behavior.
    """
    import shutil

    from merlin.perf.executed_trace import trace_invocation
    from merlin.perf import command_recorder
    from merlin.targetgen.completion_order import trace_isa

    invocation = trace_invocation(target, elf)
    if invocation is None:
        return None
    argv, env = invocation
    if not argv:
        return None
    engine = shutil.which(str(argv[0]), path=env.get("PATH"))
    if engine is None:
        return None
    command = ["engine:" + _sha(Path(engine).resolve())]
    for i, arg in enumerate(argv[1:], 1):
        arg = str(arg)
        if arg.startswith("--extlib="):
            path = Path(arg.partition("=")[2])
            if not path.is_file():
                return None
            command.append("--extlib=" + _sha(path))
        elif argv[i - 1] == "--extlib":
            if not Path(arg).is_file():
                return None
            command.append("file:" + _sha(Path(arg)))
        elif Path(arg).is_file():
            command.append("executable" if Path(arg).resolve() == elf.resolve() else "file:" + _sha(Path(arg)))
        else:
            command.append(arg)
    recorder = command_recorder.selected()
    recorder_identity = None if recorder is None else {
        "library": recorder["library"]["sha256"],
        "engine": recorder["engine"]["sha256"],
        "source": recorder["source_sha256"],
    }
    return {
        "command": command, "isa": trace_isa(target), "environment": _semantic_environment(env),
        "recorder": recorder_identity,
    }


def check(target: str, generated: Path | str, *, timeout_s: float | None = None) -> dict[str, Any]:
    """The completion-order record for the executable built in ``generated`` (never raises).

    ``timeout_s`` replaces the worker's engine budget (``TIMEOUT_ENV``) for this one check.

    The check runs in a worker process: the same code and environment, so the same record. Reading
    the commit log is pure Python per retired instruction, and a grade checks up to its worker count
    of executables at once from threads of ONE process, where every reader shared one interpreter
    lock. Measured on one 20-case grade: checks that take 2-20 s alone took 24-1200 s there, nine
    cases hit the deadline (``unknown``, so never cached and re-run by every later grade) and each
    engine sat blocked writing its log at ~14% of a core.
    """
    from merlin.common.cpu_pool import shared_compute_pool

    off = ("off", "0", "false", "no")
    if os.environ.get(SWITCH_ENV, "").strip().lower() in off or os.environ.get(WORKER_ENV, "").strip().lower() in off:
        with shared_compute_pool():
            return _check(target, generated)
    try:
        return _in_worker(target, str(generated), timeout_s)
    except Exception as exc:  # noqa: BLE001 - a crashed check is recorded, never read as a pass
        from merlin.targetgen import completion_order as CO

        return {"schema": CO.SCHEMA, "verdict": "unknown", "detail": f"check crashed: {type(exc).__name__}: {exc}"}


def _worker_main() -> None:
    """The worker side of :func:`_in_worker`: ``(target, generated)`` pickled on stdin, the record on stdout.

    The worker runs the engine and reads its stream while holding one CPU of the shared compute pool,
    leased in the caller's class (carried in its environment), so the pool counts the check as busy."""
    import pickle
    import sys

    from merlin.common.cpu_pool import shared_compute_pool

    target, generated = pickle.loads(sys.stdin.buffer.read())
    with shared_compute_pool():
        record = _check(target, generated)
    sys.stdout.buffer.write(pickle.dumps(record))
    sys.stdout.buffer.flush()


def _in_worker(target: str, generated: str, timeout_s: float | None = None) -> dict[str, Any]:
    """:func:`_check` in a fresh interpreter with this process's import path and environment."""
    import pickle
    import signal
    import subprocess
    import sys

    from merlin.common.cpu_pool import child_environment

    environment = child_environment()  # the caller's lease class travels with the worker
    if timeout_s is not None:
        environment[TIMEOUT_ENV] = str(timeout_s)
    budget = float(environment.get(TIMEOUT_ENV) or 1200)
    timeout = budget + (_WORKER_GRACE_S if timeout_s is None else min(_WORKER_GRACE_S, 30.0))
    worker = subprocess.Popen(  # noqa: S603 - this interpreter running this module
        [sys.executable, "-B", "-c", _WORKER_BOOTSTRAP, json.dumps(sys.path)],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,  # its own process group: stopping the worker stops its engine too
    )
    try:
        out, err = worker.communicate(pickle.dumps((target, generated)), timeout=timeout)
    except BaseException:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except OSError:
            pass
        worker.communicate()
        raise
    if worker.returncode != 0:
        tail = err.decode("utf-8", errors="replace").strip()[-300:]
        raise RuntimeError(f"worker exited {worker.returncode}: {tail}")
    return pickle.loads(out)


def _check(target: str, generated: Path | str) -> dict[str, Any]:
    """:func:`check` in this process."""
    from merlin.targetgen import completion_order as CO

    from .elf_lanes import PACKAGE_ELF_NAME

    if os.environ.get(SWITCH_ENV, "").strip().lower() in ("off", "0", "false", "no"):
        return {"schema": CO.SCHEMA, "verdict": "skipped", "detail": f"{SWITCH_ENV} switched the check off"}
    elf = Path(generated) / PACKAGE_ELF_NAME
    if not elf.is_file():
        return {"schema": CO.SCHEMA, "verdict": "unknown", "detail": f"no executable at {elf}"}
    try:
        from merlin.targetgen.rtl.facts import load_facts

        document = load_facts(target, regenerate=False)
        body = document.get("facts") or {}
    except Exception as exc:  # noqa: BLE001 - no facts: recorded as the reason, never a pass
        return {
            "schema": CO.SCHEMA,
            "verdict": "unknown",
            "detail": f"no RTL facts for {target!r}: {type(exc).__name__}: {exc}",
        }
    from merlin.targetgen.rtl import command_ordering_selection as selection_owner

    selection = None
    if selected := os.environ.get(selection_owner.ENV):
        try:
            ordering, selection = selection_owner.read(json.loads(selected), target=target, document=document)
            body = {**body, "command_ordering": ordering}
        except Exception as exc:  # noqa: BLE001 - an invalid explicit selection never falls back
            return {
                "schema": CO.SCHEMA,
                "verdict": "unknown",
                "detail": f"ordering selection unavailable: {type(exc).__name__}: {exc}",
            }
    ordering = body.get("command_ordering")
    if not ordering:
        return {
            "schema": CO.SCHEMA,
            "verdict": "unknown",
            "detail": (
                "the RTL facts carry no command_ordering section (merlin.targetgen.rtl.command_ordering derives it)"
            ),
        }
    try:
        model = CO.OrderingModel.for_target(target, body=body)
    except CO.Unavailable as exc:
        return {"schema": CO.SCHEMA, "verdict": "unknown", "detail": str(exc)}
    try:
        observed = _observation_identity(target, elf)
        identity = {
            "target": target,
            "elf": _sha(elf),
            "facts": _digest_json(document.get("facts")),
            # The selection's content digests; its path names where this grade's copy lives.
            "ordering_selection": None if selection is None else _digest_json(
                {k: (v.get("sha256") if isinstance(v, dict) else v) for k, v in selection.items()}
            ),
            "row_masks": model.row_masks,
            "select_flag": model.select_flag,
            "sentinel": model.sentinel,
            "readers": _checker_digest(),
            "observation": observed,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        from merlin.common.artifacts import cache_dir

        cached = cache_dir(_CACHE_NAMESPACE) / f"{key}.json"
        timeout = float(os.environ.get(TIMEOUT_ENV) or 1200)
        # A run that exhausted its budget is an outcome of these bytes UNDER THAT BUDGET, kept under its
        # own key: measured on fresh7, the largest cases spent 700-960 s per grade re-learning the same
        # "unknown". It stays "unknown" when carried; a different budget or checker is a different key.
        exhausted_key = _digest_json({"identity": key, "budget_s": timeout})
        exhausted = cache_dir(_CACHE_NAMESPACE) / f"{exhausted_key}.budget.json"
        for path in (cached, exhausted):
            if observed is not None and path.is_file():
                record = json.loads(path.read_text(encoding="utf-8"))
                if (
                    observed == _observation_identity(target, elf)
                    and _sha(elf) == identity["elf"] and _checker_digest() == identity["readers"]
                ):
                    record["cache"] = "carried"
                    return record
        record = CO.check_elf(target, elf, model=model, timeout_s=timeout)
        if (
            _sha(elf) != identity["elf"] or _checker_digest() != identity["readers"]
            or observed != _observation_identity(target, elf)
        ):
            return {"schema": CO.SCHEMA, "verdict": "unknown", "detail": "observation inputs changed during the check"}
        if selection is not None:
            record["ordering_selection"] = selection
        if observed is not None:
            # Only a complete observation, or one that ran out of exactly this budget, with unchanged
            # selected engine/decoder identity is reusable; a crash or an unavailable engine is not.
            spent = (record.get("budget_exhausted") or {}).get("timeout_s") == timeout
            if "detail" not in record:
                _publish_cache(cached, record)
            elif record.get("verdict") == "unknown" and spent:
                _publish_cache(exhausted, record)
        record["cache"] = "measured"
        return record
    except Exception as exc:  # noqa: BLE001 - a crashed check is recorded, never read as a pass
        return {"schema": CO.SCHEMA, "verdict": "unknown", "detail": f"check crashed: {type(exc).__name__}: {exc}"}


def _publish_cache(path: Path, record: dict[str, Any]) -> None:
    """Concurrent checks of byte-identical executables may publish the same cache key."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.stem, suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(record, handle)
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def certification():
    """Run the check in every grade inside this block, whatever tiers its ladder reaches."""
    token = _CERTIFYING.set(True)
    try:
        yield
    finally:
        _CERTIFYING.reset(token)


def certifies(tier_sequence, rtl_tiers) -> bool:
    """Whether a ladder over ``tier_sequence`` runs the check: it reaches an RTL tier, or it is a
    certification grade (:func:`certification`). Development screens defer it."""
    return _CERTIFYING.get() or bool(set(tier_sequence or ()) & set(rtl_tiers or ()))


def stepper(target: str, tier_sequence, rtl_tiers, adapters):
    """:func:`ladder_step` for one capsule's ladder, certifying as :func:`certifies` decides.

    The ladder reaches only the tiers it holds an adapter for. ``tier_sequence`` also lists every tier
    the contract declares, so a Spike-only self-check or development grade (adapters for L0-L2 only)
    would otherwise count the contract's RTL tier it never runs and pay for the trace."""
    import functools

    reached = [tier for tier in tier_sequence or () if (adapters or {}).get(tier) is not None]
    return functools.partial(ladder_step, target, certify=certifies(reached, rtl_tiers))


def deferred(tier: str | None = None) -> dict[str, Any]:
    """The record of a development grade that deferred the check: never a pass, never a failure."""
    from merlin.targetgen import completion_order as CO

    record = {"schema": CO.SCHEMA, "verdict": "deferred", "detail": DEFERRED_DETAIL}
    if tier is not None:
        record["tier"] = tier
    return record


def after_coverage(target: str, results: dict[str, Path], *, remaining, workers: int = 1,
                   margin_s: float = 30.0) -> dict[str, Any]:
    """Best-effort check of already published passes, inside whatever time ``remaining()`` leaves.

    ``results`` maps a case to its ``capsule_result.json``. Runs after the final coverage is written
    and never changes it: each check's engine budget ends ``margin_s`` before the deadline, a case
    is not started once that time is gone, and an unchecked case is reported as such, not failed.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock, verdicts = threading.Lock(), {}

    def one(name):
        budget = remaining() - margin_s
        if budget < 1:
            return
        record = check(target, Path(results[name]).parent / "generated", timeout_s=budget)
        with lock:
            verdicts[name] = {k: record.get(k) for k in ("verdict", "detail", "cache", "hazard_pairs")
                              if record.get(k) is not None}

    names = sorted(results)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, names))
    unchecked = [name for name in names if name not in verdicts]
    return {
        "status": "not_run" if len(unchecked) == len(names) else "partial" if unchecked else "complete",
        "scope": "best-effort after coverage publication; advisory, never changes a coverage row",
        "hazards": sorted(name for name, row in verdicts.items() if row.get("verdict") == "hazard"),
        "verdicts": verdicts,
        "not_checked": unchecked,
    }


def failure_reason(record: dict[str, Any]) -> str | None:
    """The author-facing reason when ``record`` fails the capsule, else None."""
    if record.get("verdict") != "hazard":
        return None
    from merlin.targetgen import completion_order as CO

    return CO.reason(record)


def functional_tier(tiers: dict, cfg) -> str | None:
    """The mandatory functional (non-RTL) tier that passed: the L2 evidence this check belongs to."""
    rtl = set(getattr(cfg, "rtl_tiers", ()) or ())
    passed = [
        t
        for t in sorted(tiers)
        if t not in rtl and getattr(tiers[t], "status", None) == "pass" and getattr(tiers[t], "mandatory", False)
    ]
    return passed[0] if passed else None


def after_functional_pass(target: str, tier: str, tiers: dict, generated) -> tuple[dict[str, Any], str | None]:
    """Run the check for ``tier`` (a passed functional tier); returns ``(record, failure reason)``.

    On a hazard the tier's record is rewritten to ``fail`` with the reason, in place."""
    import dataclasses

    record = check(target, generated)
    record["tier"] = tier
    reason = failure_reason(record)
    if reason is not None and tier in tiers:
        tiers[tier] = dataclasses.replace(tiers[tier], status="fail", reason=reason)
    return record, reason


def ladder_step(target: str, tier: str | None, tiers: dict, generated, cfg=None, *, certify: bool = True):
    """``(record, CertFailure | None)`` for the ladder. ``tier`` None: find the passed functional tier
    (one carried from an earlier grade); ``(None, None)`` when there is none to qualify.
    ``certify`` False (see :func:`certifies`): the :func:`deferred` record, without running the check."""
    from merlin.targetgen.capsule_common import _cat
    from merlin.targetgen.package_runtime import CertFailure

    if tier is None:
        tier = functional_tier(tiers, cfg)
        if tier is None:
            return None, None
    if not certify:
        return deferred(tier), None
    from merlin_experiments.phase1.feedback import setup_timing

    # Its own host stage: on large programs this commit-log trace runs for minutes after the numeric
    # comparison (measured 520-960 s per case) and was otherwise invisible between two stages.
    with setup_timing.stage("completion_order"):
        record, reason = after_functional_pass(target, tier, tiers, generated)
    failure = None if reason is None else CertFailure("completion_order", _cat("PROTOCOL_VIOLATION"), reason)
    return record, failure


def suppress_later(tiers: dict, sequence, failed: str, required, rtl_tiers, factory) -> None:
    """Record every tier after ``failed`` the ladder will not run as skipped (never absent)."""
    for later in list(sequence)[list(sequence).index(failed) + 1 :]:
        if later not in tiers:
            tiers[later] = factory(later, later in required, failed, from_rtl=later in set(rtl_tiers or ()))


def sweep_trigger(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """The completion-order verdict as the memory-perturbation sweep's static trigger, or None."""
    if not record or record.get("verdict") not in ("clear", "hazard", "unknown"):
        return None
    verdict = {"clear": "clear", "hazard": "hazard", "unknown": "unresolved"}[record["verdict"]]
    return {
        "verdict": verdict,
        "source": "completion_order",
        "hazard_pairs": record.get("hazard_pairs"),
        "uncertain": record.get("uncertain") or ({"detail": record.get("detail")} if record.get("detail") else {}),
    }
