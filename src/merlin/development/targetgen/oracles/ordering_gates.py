"""The completion-order check as the capsule ladder runs it: L2 structural evidence.

A numerical pass on the functional model (the L2 tier) says the program computes the declared
operation when every accelerator command completes in program order. :mod:`merlin.development.targetgen.
completion_order` asks the question that pass cannot: does the program depend on that order where
the hardware does not provide it? It reads the command stream the functional run actually executed
and the target's RTL-derived ordering facts, so it decides at L2, before any RTL engine runs, and a
flagged pair fails the capsule there with the commands and rows that race.

* ``hazard``: the functional tier is failed with :func:`merlin.development.targetgen.oracles.completion_order.reason`.
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
import functools
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---- completion_order_gate ---------------------------------------------------------------------

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
    "from merlin.development.targetgen.oracles.ordering_gates import _worker_main; _worker_main()"
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checker_digest() -> str:
    from merlin.analysis.perf.measurement import (
        command_recorder,
    )
    from merlin.analysis.perf.measurement import (
        execution_traces as executed_trace,
    )
    from merlin.analysis.perf.measurement import (
        execution_traces as streamed_executed_trace,
    )
    from merlin.development.targetgen.oracles import completion_order
    from merlin.target.capabilities import semantic_families
    from merlin.target.isa.rocc import decode, rtl_checks_generic, semantics

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
    from merlin.analysis.perf.measurement.command_recorder import INNER_ENV, LIMIT_ENV, OUT_ENV, SELECTION_ENV

    outputs = {
        "PWD",
        "OLDPWD",
        "TMPDIR",
        "TMP",
        "TEMP",
        "MERLIN_OUT_ROOT",
        "MERLIN_BUNDLE_CAS",
        "MERLIN_COMPUTE_CPU_SLOTS",
        "MERLIN_COMPUTE_CPUS",
        "MERLIN_GRADE_CPUS",
        "MERLIN_PREPARATION_CPUS",
        "MERLIN_CPU_PRIORITY",
        SELECTION_ENV,
        OUT_ENV,
        INNER_ENV,
        LIMIT_ENV,
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

    from merlin.analysis.perf.measurement import command_recorder
    from merlin.analysis.perf.measurement.execution_traces import trace_invocation
    from merlin.development.targetgen.oracles.completion_order import trace_isa

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
    recorder_identity = (
        None
        if recorder is None
        else {
            "library": recorder["library"]["sha256"],
            "engine": recorder["engine"]["sha256"],
            "source": recorder["source_sha256"],
        }
    )
    return {
        "command": command,
        "isa": trace_isa(target),
        "environment": _semantic_environment(env),
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
    from merlin.common.process.cpu_pool import shared_compute_pool

    off = ("off", "0", "false", "no")
    if os.environ.get(SWITCH_ENV, "").strip().lower() in off or os.environ.get(WORKER_ENV, "").strip().lower() in off:
        with shared_compute_pool():
            return _check(target, generated)
    try:
        return _in_worker(target, str(generated), timeout_s)
    except Exception as exc:  # noqa: BLE001 - a crashed check is recorded, never read as a pass
        from merlin.development.targetgen.oracles import completion_order as CO

        return {"schema": CO.SCHEMA, "verdict": "unknown", "detail": f"check crashed: {type(exc).__name__}: {exc}"}


def _worker_main() -> None:
    """The worker side of :func:`_in_worker`: ``(target, generated)`` pickled on stdin, the record on stdout.

    The worker runs the engine and reads its stream while holding one CPU of the shared compute pool,
    leased in the caller's class (carried in its environment), so the pool counts the check as busy."""
    import pickle
    import sys

    from merlin.common.process.cpu_pool import shared_compute_pool

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

    from merlin.common.process.cpu_pool import child_environment

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
    from merlin.development.targetgen.grading.elf_lanes import PACKAGE_ELF_NAME
    from merlin.development.targetgen.oracles import completion_order as CO

    if os.environ.get(SWITCH_ENV, "").strip().lower() in ("off", "0", "false", "no"):
        return {"schema": CO.SCHEMA, "verdict": "skipped", "detail": f"{SWITCH_ENV} switched the check off"}
    elf = Path(generated) / PACKAGE_ELF_NAME
    if not elf.is_file():
        return {"schema": CO.SCHEMA, "verdict": "unknown", "detail": f"no executable at {elf}"}
    try:
        from merlin.target.rtl.facts import load_facts

        document = load_facts(target, regenerate=False)
        body = document.get("facts") or {}
    except Exception as exc:  # noqa: BLE001 - no facts: recorded as the reason, never a pass
        return {
            "schema": CO.SCHEMA,
            "verdict": "unknown",
            "detail": f"no RTL facts for {target!r}: {type(exc).__name__}: {exc}",
        }
    from merlin.target.rtl.hw import command_ordering_selection as selection_owner

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
                "the RTL facts carry no command_ordering section (merlin.target.rtl.hw.command_ordering derives it)"
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
            "ordering_selection": None
            if selection is None
            else _digest_json({k: (v.get("sha256") if isinstance(v, dict) else v) for k, v in selection.items()}),
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
                    and _sha(elf) == identity["elf"]
                    and _checker_digest() == identity["readers"]
                ):
                    record["cache"] = "carried"
                    return record
        record = CO.check_elf(target, elf, model=model, timeout_s=timeout)
        if (
            _sha(elf) != identity["elf"]
            or _checker_digest() != identity["readers"]
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

    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.stem, suffix=".tmp", delete=False
    ) as handle:
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
    from merlin.development.targetgen.oracles import completion_order as CO

    record = {"schema": CO.SCHEMA, "verdict": "deferred", "detail": DEFERRED_DETAIL}
    if tier is not None:
        record["tier"] = tier
    return record


def after_coverage(
    target: str, results: dict[str, Path], *, remaining, workers: int = 1, margin_s: float = 30.0
) -> dict[str, Any]:
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
            verdicts[name] = {
                k: record.get(k) for k in ("verdict", "detail", "cache", "hazard_pairs") if record.get(k) is not None
            }

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
    from merlin.development.targetgen.oracles import completion_order as CO

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
    from merlin.development.targetgen.corpus.capsule_io import _cat
    from merlin.development.targetgen.packaging.loading import CertFailure

    if tier is None:
        tier = functional_tier(tiers, cfg)
        if tier is None:
            return None, None
    if not certify:
        return deferred(tier), None
    from merlin.evaluation.capsule.operations import setup_timing

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


# ---- load_order --------------------------------------------------------------------------------
# The memory-order tier: does a certified program depend on memory answering in issue order?
#
# WHY. The cert tier runs the elaborated RTL against a memory model that answers every request in the
# order it was accepted. A program that writes the same accumulator rows with two loads and nothing
# ordering their COMPLETION (only their issue) is correct on that model and wrong on a board whose memory
# system reorders independent requests. Measured: an overwrite-then-accumulate residual add passed L2 and
# L3, then on FireSim returned the first operand alone in the failing elements. A fence between the two
# loads made it pass on the same board, and so did the vendor library's own residual add once fenced.
#
# OPT-IN. The tier costs one in-order run plus N seeded runs of the memory-perturbing engine per capsule it
# fires on, so it runs only when ``MERLIN_L3_ORDER_SEEDS`` names a positive seed count. Unset (or 0), a
# grade is byte-identical to one from before this tier existed and no record is written.
#
# WHAT IT DOES, both steps fail-closed:
#
# 1. A STATIC trigger. When the grade reached a completion-order verdict over the executed command
#    stream (:mod:`merlin.development.targetgen.oracles.completion_order`), that verdict is the trigger. Otherwise, the
#    accumulator-only check over the decoded instruction trace (:func:`static_check`): two inbound transfers whose
#    destinations may overlap in the accumulator, with no barrier between them, are a hazard. A transfer
#    whose direction or destination the decoder could not resolve, and a command whose memory effects
#    this check does not model (a hardware loop unroller issues its own loads), count as UNRESOLVED.
#    Unresolved triggers the sweep exactly as a proven hazard does: "we could not tell" is never "clear".
# 2. When the static check is not clear, the executable is re-run on the target's memory-perturbing RTL
#    engine (:mod:`merlin.development.targetgen.oracles.divergence`), in order and under N seeds, and every run is judged
#    against the capsule's own golden. Any run that answers wrong under a legal schedule FAILS the
#    certificate, and the record names the seeds that reproduce it.
#
# WHERE THE FACTS COME FROM. The accumulator and accumulate bits of a local address come from the target's
# derived ISA constants (:func:`merlin.target.isa.rocc.decode.isa_constants`, owned by the selected
# backend) and the block width of a transfer from the derived array geometry
# (:func:`merlin.analysis.perf.bounds.per_shape_ceiling.machine_from_facts`). Which commands move data, and in which direction,
# is read from the decoded payload's own fields (the shared decoded-ABI field vocabulary of
# :mod:`merlin.analysis.perf.dependencies.rocc`); which command is a barrier comes from the shared class-family table
# (:mod:`merlin.target.capabilities.semantic_families`). A target without those facts is recorded as not applicable,
# never checked against guessed ones.
#
# WHAT A VERDICT MEANS. ``order_sensitive`` is a statement about the PROGRAM under the ordering freedom the
# bus protocol grants, not a claim that a particular chip reorders those two requests. ``order_stable`` is
# evidence, not proof: N seeds sample the schedule space, they do not cover it.

#: Which decoded-ABI classes issue NO memory request of their own -- configuration, cache maintenance,
#: and the preload/compute steps whose operands are already on chip -- is read from the shared class
#: vocabulary every capability manifest maps its RTL funct codes onto
#: (:func:`merlin.target.capabilities.semantic_families.local_only_classes`), never re-listed here and never a fact
#: about one device. A class outside that set that carries no transfer evidence is OPAQUE -- it may issue
#: loads the trace cannot see -- and is treated as an unresolved load.

#: The shared class family that orders memory (a host fence drains the accelerator's queues).
BARRIER_FAMILY = "synchronization"
#: The shared class family of a data-motion command.
MOVEMENT_FAMILY = "movement"
#: The decoded payload field naming a transfer's off-chip operand.
OFF_CHIP_FIELD = "dram"

#: The pin-registry role of a memory-perturbing elaborated-RTL engine. Looked up per target.
ENGINE_ROLE = "verilator_binary_mem_perturb"

SEEDS_ENV = "MERLIN_L3_ORDER_SEEDS"  # how many seeds; unset or 0 leaves the tier off
LATENCY_ENV = "MERLIN_L3_ORDER_MAX_LATENCY"  # any of these four: one profile instead of the defaults
TAIL_ENV = "MERLIN_L3_ORDER_TAIL_PERMILLE"
REGION_ENV = "MERLIN_L3_ORDER_REGION_LOG2"
SLOW_ENV = "MERLIN_L3_ORDER_SLOW_PERMILLE"
WORKERS_ENV = "MERLIN_L3_ORDER_WORKERS"
#: The knob profiles the seeds are dealt across, round robin. Two, because the two races measured need
#: opposite settings. A back-to-back overwrite/accumulate pair is caught by SHORT uniform latency (5 of 8
#: seeds at 64 cycles, 0 of 8 at 512 cycles with a tail, where a long first load fills the DMA's
#: in-flight window and the second cannot even issue). A pair several loads apart (a hardware loop
#: unroller's residual add) was never flagged by uniform latency up to 2048 cycles, nor by per-region
#: latency, and was flagged only by a heavy per-response TAIL. The tail is bounded because the
#: accelerator's own stall watchdog fires past ~10000 cycles. The fenced control stayed clean at every
#: setting tried.
DEFAULT_PROFILES: tuple[dict, ...] = (
    {"max_latency": 64, "tail_permille": 0},
    {"max_latency": 512, "tail_permille": 100},
)


@dataclass(frozen=True)
class OrderFacts:
    """What the static check needs to know about a target's local addresses, all derived."""

    acc_bit: int  # set in a local address that names the accumulator
    accum_bit: int  # set when the write adds onto the rows instead of overwriting them
    flag_mask: int  # every mode bit a local address may carry beside the row
    block: int | None  # columns per transfer block; None leaves every multi-column span unbounded


@functools.cache
def facts_for(target: str) -> tuple[OrderFacts | None, str]:
    """``(facts, why)``: the derived facts, or None and the reason they could not be derived.

    Cached per target, the absence included: deriving them loads the backend's ISA constants and the RTL
    facts, and a target without them would otherwise pay that failed derivation on every capsule.
    """
    try:
        from merlin.target.isa.rocc import decode as RD

        isa = RD.isa_constants(target)
    except Exception as exc:  # noqa: BLE001 -- no facts is "not applicable", returned with its reason
        return None, f"no derived ISA constants for {target!r}: {type(exc).__name__}: {exc}"
    bits = {name: isa.get(name) for name in ("ACC_I8", "ACC_ACCUM", "FULL_C_BIT")}
    missing = sorted(name for name, value in bits.items() if not isinstance(value, int) or isinstance(value, bool))
    if missing:
        return None, f"{target!r} derives no {', '.join(missing)}: accumulator rows cannot be told apart"
    try:
        from merlin.analysis.perf.bounds.decomposition import is_unknown
        from merlin.analysis.perf.bounds.per_shape_ceiling import machine_from_facts

        cols = machine_from_facts(target, measure_fill=False).array_cols
        block = None if is_unknown(cols) else int(cols)
    except Exception:  # noqa: BLE001 -- an unknown block width only widens spans to unbounded
        block = None
    acc, accum, full = bits["ACC_I8"], bits["ACC_ACCUM"], bits["FULL_C_BIT"]
    return OrderFacts(acc, accum, acc | accum | full, block), "derived"


@dataclass
class _Load:
    index: int
    cls: str
    rows: tuple[int, int | None] | None  # [lo, hi) accumulator rows (hi None: unbounded), None: unresolved


def _family(cls: str) -> str | None:
    from merlin.target.capabilities.semantic_families import from_isa_class

    return from_isa_class(cls)


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _destination(payload: Mapping) -> str:
    """``in`` / ``out`` / ``unknown`` for a transfer, read from which local field its payload resolves."""
    from merlin.analysis.perf.dependencies.rocc import CONSUMES, DEFINES

    if any(_int(payload.get(name)) is not None for name in DEFINES):
        return "in"
    if any(_int(payload.get(name)) is not None for name in CONSUMES):
        return "out"
    return "unknown"


def _as_load(row: Mapping, facts: OrderFacts) -> _Load | None:
    """The accumulator rows a command may write by a memory transfer, None when it provably writes none."""
    from merlin.analysis.perf.dependencies.rocc import DEFINES

    cls = str(row.get("class"))
    index = row.get("index", -1)
    payload = row.get("decoded") if isinstance(row.get("decoded"), Mapping) else {}
    transfer = OFF_CHIP_FIELD in payload or _family(cls) == MOVEMENT_FAMILY
    if not transfer:
        # no transfer evidence: a local-only command issues no request; anything else is opaque
        from merlin.target.capabilities.semantic_families import local_only_classes

        return None if cls in local_only_classes() else _Load(index, cls, None)
    direction = _destination(payload)
    if direction == "out":
        return None
    if direction == "unknown":
        return _Load(index, cls, None)
    addr = next(_int(payload.get(name)) for name in DEFINES if _int(payload.get(name)) is not None)
    if not addr & facts.acc_bit:
        return None  # a scratchpad destination: not the accumulator
    base = addr & ~facts.flag_mask
    rows, cols = _int(payload.get("rows")), _int(payload.get("cols"))
    if rows is None or cols is None or facts.block is None:
        return _Load(index, cls, (base, None))
    blocks = max(1, -(-cols // facts.block))
    # A multi-block transfer's stride between blocks is configured elsewhere, so its span is unbounded.
    return _Load(index, cls, (base, base + rows) if blocks == 1 else (base, None))


def _overlap(a: _Load, b: _Load) -> bool | None:
    """True / False when both ranges are known, None when either is not."""
    if a.rows is None or b.rows is None:
        return None
    (alo, ahi), (blo, bhi) = a.rows, b.rows
    if not ((ahi is None or blo < ahi) and (bhi is None or alo < bhi)):
        return False
    return None if (ahi is None or bhi is None) else True


def static_check(trace: Mapping | None, facts: OrderFacts) -> dict:
    """Classify a decoded trace: ``hazard`` / ``unresolved`` / ``clear``, with the pairs that decided it."""
    hazards: list[dict] = []
    unresolved: list[dict] = []
    window: list[_Load] = []
    for row in (trace or {}).get("instructions") or []:
        cls = str(row.get("class"))
        if _family(cls) == BARRIER_FAMILY:
            window = []
            continue
        load = _as_load(row, facts)
        if load is None:
            continue
        for prev in window:
            overlap = _overlap(prev, load)
            pair = {"first": prev.index, "second": load.index, "classes": [prev.cls, load.cls]}
            if overlap is True:
                hazards.append(pair)
            elif overlap is None:
                unresolved.append(pair)
        window.append(load)
    verdict = "hazard" if hazards else ("unresolved" if unresolved else "clear")
    return {
        "verdict": verdict,
        "hazard_pairs": hazards[:16],
        "unresolved_pairs": unresolved[:16],
        "n_hazard_pairs": len(hazards),
        "n_unresolved_pairs": len(unresolved),
    }


def engine_for(target: str) -> tuple[Path | None, str]:
    """The target's declared memory-perturbing engine, verified by content, or (None, why)."""
    from merlin.common.provenance_claims import load_artifacts, verify_artifact

    try:
        arts = [a for a in load_artifacts().values() if a.target == target and a.role == ENGINE_ROLE]
    except Exception as exc:  # noqa: BLE001 -- recorded as the reason the engine is unavailable
        return None, f"pin registry unreadable: {type(exc).__name__}: {exc}"
    if not arts:
        return None, f"no artifact with role {ENGINE_ROLE!r} is declared for {target!r}"
    if len(arts) > 1:
        return None, f"{len(arts)} artifacts declare role {ENGINE_ROLE!r} for {target!r}; the engine is ambiguous"
    check = verify_artifact(arts[0].name)
    if not check.present or check.matches is not True:
        return None, f"{arts[0].name}: " + ("; ".join(check.gaps) or "not verified")
    return Path(check.path), f"{arts[0].name} ({check.digest[:12]})"


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def seeds_requested() -> int:
    """How many seeds the tier runs; 0 (the default) leaves it off."""
    return max(0, _int_env(SEEDS_ENV) or 0)


def _profiles() -> tuple[dict, ...]:
    """The declared profiles, or the single one the environment names when it names any knob."""
    named = {
        "max_latency": _int_env(LATENCY_ENV),
        "tail_permille": _int_env(TAIL_ENV),
        "region_log2": _int_env(REGION_ENV),
        "slow_permille": _int_env(SLOW_ENV),
    }
    if all(value is None for value in named.values()):
        return DEFAULT_PROFILES
    return ({k: v for k, v in named.items() if v is not None},)


@dataclass
class OrderOutcome:
    record: dict
    failed: bool = False
    seeds: list = field(default_factory=list)

    @property
    def reason(self) -> str:
        names = ", ".join("in_order" if s is None else f"seed {s}" for s in self.seeds)
        return (
            "the certified program answers wrong when independent memory responses complete out of "
            f"order ({names}; {self.record.get('knobs')}): two loads into overlapping accumulator rows "
            "are ordered by issue, not completion -- order them (a fence) or make them not overlap"
        )


def run_order_check(
    *,
    target: str,
    trace: Mapping | None,
    elf: Path,
    judge: Callable[[int, str], str],
    facts: OrderFacts | None = None,
    engine: tuple[Path | None, str] | None = None,
    sweep: Callable[..., dict] | None = None,
    static: Mapping | None = None,
) -> OrderOutcome | None:
    """Run the tier. None only when it is switched off; every other outcome is recorded.

    ``static`` is the trigger verdict (``clear`` / ``hazard`` / ``unresolved``) when the caller already
    has one -- the completion-order check over the executed stream
    (:mod:`merlin.development.targetgen.oracles.completion_order`), which covers every local memory and follows the
    dispatch's real dependency edges. Without it the accumulator-only :func:`static_check` decides."""
    seeds_n = seeds_requested()
    if seeds_n <= 0:
        return None
    if static is None:
        why = "supplied"
        if facts is None:
            facts, why = facts_for(target)
        if facts is None:
            return OrderOutcome({"status": "not_applicable", "detail": why})
        static = static_check(trace, facts)
    static = dict(static)
    if static["verdict"] == "clear":
        return OrderOutcome({"status": "static_clear", "static": static})
    path, engine_why = engine if engine is not None else engine_for(target)
    rec: dict[str, Any] = {"static": static, "engine": engine_why}
    if path is None:
        rec["status"] = "unavailable"
        return OrderOutcome(rec)
    if not Path(elf).is_file():
        rec["status"] = "unavailable"
        rec["engine"] = f"{engine_why}; no executable at {elf}"
        return OrderOutcome(rec)
    profiles = _profiles()
    from merlin.development.targetgen.oracles import divergence as MP

    if sweep is None:
        sweep = MP.sweep
    # the engine's recorded invocation: it preloads the image so operands reach the memory model
    extra = [a.replace("{elf}", str(elf)) for a in MP.invocation_for(path)]
    rec["invocation"] = extra
    runs: list[dict] = []
    elapsed = 0.0
    res: dict = {}
    for i, knobs in enumerate(profiles):
        seeds = [s for s in range(1, seeds_n + 1) if (s - 1) % len(profiles) == i]
        if not seeds:
            continue
        # through the environment, so the engine runs with exactly the command line its backend uses
        res = sweep(
            path, elf, seeds, judge, via="env", include_in_order=(i == 0), extra_args=extra,
            workers=_int_env(WORKERS_ENV) or len(seeds) + 1, **knobs,
        )  # fmt: skip
        runs += [{**{k: r[k] for k in ("seed", "rc", "wall_s", "outcome")}, "profile": i} for r in res["runs"]]
        elapsed += float(res["elapsed_s"])
    bad = sorted((r["seed"] for r in runs if r["outcome"] == "fail"), key=lambda s: -1 if s is None else s)
    unjudged = [r["seed"] for r in runs if r["outcome"] not in ("pass", "fail")]
    rec.update(
        {
            "status": "order_sensitive" if bad else ("unjudged" if unjudged else "order_stable"),
            "failing_seeds": bad,
            "unjudged_seeds": unjudged,
            "knobs": {"seeds": seeds_n, "profiles": list(profiles)},
            "runs": runs,
            "elapsed_s": round(elapsed, 2),
            "emulator_sha256": res.get("emulator_sha256"),
            "elf_sha256": res.get("elf_sha256"),
        }
    )
    return OrderOutcome(rec, failed=bool(bad), seeds=bad)


def after_certificate(
    *,
    target: str,
    tiers: Mapping[str, Any],
    rtl_tiers: Any,
    trace: Mapping | None,
    elf: Path,
    judge: Callable[[int, str], str],
    static: Mapping | None = None,
) -> OrderOutcome | None:
    """The tier as the capsule ladder runs it, after a ladder whose cert tier passed.

    The cert tier is the deepest elaborated-RTL tier that passed; with none, nothing was certified and
    there is nothing to qualify. Never raises: a check that crashed is recorded as ``error`` on the
    result, because a silent skip would read as a clean one.
    """
    if seeds_requested() <= 0:
        return None
    passed = [t for t in sorted(tiers) if t in set(rtl_tiers or ()) and getattr(tiers[t], "status", None) == "pass"]
    if not passed:
        return None
    try:
        out = run_order_check(target=target, trace=trace, elf=elf, judge=judge, static=static)
    except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed
        out = OrderOutcome({"status": "error", "detail": f"{type(exc).__name__}: {exc}"})
    if out is not None:
        out.record["cert_tier"] = passed[-1]
    return out


def console_judge(compare: Callable[[dict], bool], parse: Callable[[str], dict]) -> Callable[[int, str], str]:
    """``pass`` / ``fail`` against the golden, ``unjudged:<why>`` when the run produced nothing to compare."""

    def judge(rc: int, text: str) -> str:
        if rc != 0:
            return f"unjudged:rc={rc}"
        try:
            outputs = parse(text)
        except Exception as exc:  # noqa: BLE001 -- no parseable answer is an outcome, not a crash
            return f"unjudged:{type(exc).__name__}"
        return "pass" if compare(outputs) else "fail"

    return judge
