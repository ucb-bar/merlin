"""Catalog of the **Merlin-authored** MLIR passes, and the dialect-plane entry point.

The lowering pipeline reuses upstream MLIR passes for the mechanical descent
(bufferization, generic vectorization, ``convert-*-to-llvm``) — see
``llvmlower.pipeline.UPSTREAM_PIPELINE``. *This* module enumerates the transforms Merlin
**writes itself**, because they encode the research and have no upstream equivalent.

This file deliberately exposes separate registries instead of pretending they are one pipeline:

``catalog()``
    The production whole-model boundary passes. A real model capsule reaches these through
    :func:`run_dialect_plane`, and the capsule runner records that invocation.
``normalization_catalog()``
    Target-independent numeric/frontend normalizations. They are tested, but they do not discharge a
    target obligation and must never be laundered into one.
``prototype_catalog()``
    The staged contract→schedule→interface→target→runtime research pipeline. It is exercised by
    its own executable tests, not credited to a production capsule that does not call it.

The out-of-tree target backend has a fourth, external pass plane: its four mandatory manifest commands.
Those are subprocess entrypoints and are enforced by ``targetgen.capsule_common.run_entrypoints`` on
every capsule. They are not Merlin callables and therefore do not belong in this in-process catalog.

**Why each entry carries an obligation and a dialect pair.** A pass catalog that records only a
phase LABEL cannot answer the two questions that decide whether a generated backend is a compiler
or a pile: *what is this pass here to discharge?* and *does anything actually run it?* Both failure
modes are real — a pass whose only justification is that the score went up, and a declared pass no
capsule ever reaches — and a `stage` string expresses neither. So every entry now declares the
dialect it consumes, the dialect it produces, the ONE obligation it exists to discharge, and the
capsules that require it; :mod:`build_tools.scripts.check_pass_obligations` gates the first pair and
measures the second against a recorded invocation log (see :func:`install_pass_recorder`).

``UNKNOWN`` is a value, never a default to hide behind: it means "not determinable from this pass's
own code", it is reported by the gate, and a pass that is UNKNOWN on obligation AND declares no
requiring capsule is REJECTED. Do not make an entry green by guessing — a fabricated obligation is
worse than a reported gap, because a reported gap gets fixed.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .._common import HAS_XDSL

# --- the obligation vocabulary ---------------------------------------------------------------------
# A pass may exist ONLY to discharge one of these four. They are deliberately few: the point is that
# a reader can ask "which of these is this pass?" and get an answer that is either obvious or absent.
#
#   partition/eligibility   decides WHICH work is a unit and whether that unit is legal on the target
#                           (dispatch formation, hart placement, capability/capacity proofs).
#   target transformation   rewrites the program for the target's execution model while staying in
#                           the target-independent plane (residency, tiling, vector strategy).
#   target lowering         descends onto the target's own vocabulary — its dialect, its command
#                           buffer, its instruction encoding.
#   boundary materialization  materializes the seam between compiled accelerator work and the host /
#                           runtime: the interface ops, the call ABI, the dispatch table.
OBLIGATIONS: tuple[str, ...] = ("partition/eligibility", "target transformation",
                                "target lowering", "boundary materialization")

# Not determinable from the pass's own code. Reported, never silently accepted.
UNKNOWN = "UNKNOWN"

# The target's own dialect, named by ITS dialect plan — never a literal here (the cardinal
# derive-don't-hardcode rule: a target name in this shared table is exactly the overfit the gate
# rejects). ``lower_to_target``/``lower_to_runtime`` resolve the real name from the plan at run time.
TARGET_DIALECT = "<target-declared>"

# Not an MLIR dialect at all: the serializable runtime dispatch DAG (``dispatch_program.py``).
# Spelled out so a reader cannot mistake it for a dialect that simply has no ops registered.
DISPATCH_PROGRAM = "<dispatch-program>"


@dataclass(frozen=True)
class PassInfo:
    name: str            # the conceptual MLIR pass name
    stage: str           # phase the pass belongs to
    summary: str
    entry: str           # dotted path of the implementing callable
    input_dialect: str = UNKNOWN     # dialect consumed (or a sentinel above)
    output_dialect: str = UNKNOWN    # dialect produced (or a sentinel above)
    obligation: str = UNKNOWN        # one of OBLIGATIONS, or UNKNOWN
    # Requirement classes declared by capsules in ``pass_requirements``.  A class, rather than a
    # target-specific capsule name in this shared module, keeps the catalog target-independent while
    # still proving that a concrete capsule carrying that class caused the invocation.
    required_by: tuple[str, ...] = ()

    def discharges(self) -> bool:
        """True when this pass names one of the four allowed production obligations."""
        return self.obligation in OBLIGATIONS

    def is_required(self) -> bool:
        """True when at least one concrete capsule must traverse this pass."""
        return bool(self.required_by)


NORMALIZATION_CATALOG: tuple[PassInfo, ...] = (
    PassInfo("merlin-lower-quant-ext", "normalize",
             "dequantize_per_channel -> linalg.generic (i8 weights stay i8 in memory)",
             "merlin.llvmlower.passes_xdsl.lower_quant_ext",
             input_dialect="quant_ext", output_dialect="linalg", obligation=UNKNOWN),
    PassInfo("merlin-bf16-matmul-f32acc", "normalize",
             "bf16 linalg.matmul -> f32-accumulating generic + truncf (matches torch)",
             "merlin.llvmlower.passes_xdsl.lower_bf16_matmul_f32acc",
             input_dialect="linalg", output_dialect="linalg", obligation=UNKNOWN),
)


MODEL_BOUNDARY_CAPSTONE = "model-boundary-capstone"
_MODEL_CAPSTONES = (MODEL_BOUNDARY_CAPSTONE,)


CATALOG: tuple[PassInfo, ...] = (
    # Dispatch formation: the roots that become units of placeable work. Structure changes, dialects
    # do not — the kernels and the driver are the same linalg-on-tensors the input was.
    PassInfo("merlin-outline-dispatches", "outline",
             "split func @forward into per-dispatch kernel funcs + a driver",
             "merlin.xdsl_dialects.lowering.outline.outline_dispatches",
             input_dialect="linalg", output_dialect="linalg",
             obligation="partition/eligibility", required_by=_MODEL_CAPSTONES),
    # The driver stops being IR and becomes the table the runtime walks: buffer ids, model-arg
    # indices, result ids. That table IS the compiled-work/host seam.
    PassInfo("merlin-emit-dispatch-program", "runtime",
             "flatten the driver into a serializable dispatch DAG for the runtime",
             "merlin.xdsl_dialects.lowering.dispatch_program.build_dispatch_program",
             input_dialect="func", output_dialect=DISPATCH_PROGRAM,
             obligation="boundary materialization", required_by=_MODEL_CAPSTONES),
    PassInfo("merlin-partition-dispatches", "runtime",
             "level-synchronous multicore schedule of the dispatch DAG across harts",
             "merlin.xdsl_dialects.lowering.schedule_dispatch.partition_dispatches",
             input_dialect=DISPATCH_PROGRAM, output_dialect=DISPATCH_PROGRAM,
             obligation="partition/eligibility", required_by=_MODEL_CAPSTONES),
    # The `_mlir_ciface_*` wrapper is literally the host call boundary; nothing else materializes it.
    PassInfo("merlin-add-c-interface", "edge",
             "attach llvm.emit_c_interface so each public func gets a ciface wrapper",
             "merlin.llvmlower.passes_xdsl.add_c_interface",
             input_dialect="func", output_dialect="func",
             obligation="boundary materialization", required_by=_MODEL_CAPSTONES),
)


EDGE_CATALOG: tuple[PassInfo, ...] = (
    PassInfo("merlin-lower-inline-asm", "edge",
             "merlin.inline_asm -> llvm.inline_asm 1:1 (custom ISA, no LLVM fork)",
             "merlin.llvmlower.custom_isa.lower_inline_asm",
             input_dialect="merlin", output_dialect="llvm",
             obligation="target lowering"),
)


PROTOTYPE_CATALOG: tuple[PassInfo, ...] = (
    # staged core-dialect passes (synthetic-workload path; see pipeline.py)
    # Eligibility in the literal sense: it emits the capability, the requirement, and the proofs
    # (immutability, capacity fit) that decide whether the work is legal on this target at all.
    PassInfo("merlin-infer-contract-facts", "contract",
             "annotate linalg with reuse/immutability/quant/capacity facts",
             "merlin.xdsl_dialects.lowering.contract_facts.lower_to_contract",
             input_dialect="linalg", output_dialect="contract",
             obligation="partition/eligibility"),
    PassInfo("merlin-apply-schedule", "schedule",
             "residency/tiling/vector-strategy decisions over contract facts",
             "merlin.xdsl_dialects.lowering.schedule_decisions.lower_to_schedule",
             input_dialect="contract", output_dialect="schedule",
             obligation="target transformation"),
    # The interface op set (resident_pack/matmul/commit/evict) is accelerator work by construction —
    # see targetgen.boundary on why that grammar cannot express host computation. Materializing into
    # it, completely or not at all (`unaccounted_ops` fails closed), is the boundary obligation.
    PassInfo("merlin-materialize-interface", "interface",
             "schedule decisions -> interface ops (resident_pack/matmul/commit)",
             "merlin.xdsl_dialects.lowering.interface_lowering.lower_to_interface",
             input_dialect="schedule", output_dialect="interface",
             obligation="boundary materialization"),
    PassInfo("merlin-lower-to-target", "target",
             "interface ops -> a reference target dialect (toynpu/saturn)",
             "merlin.xdsl_dialects.lowering.target_lowering.lower_to_target",
             input_dialect="interface", output_dialect=TARGET_DIALECT,
             obligation="target lowering"),
    PassInfo("merlin-lower-to-runtime", "runtime",
             "target ops -> runtime command-buffer IR",
             "merlin.xdsl_dialects.lowering.runtime_lowering.lower_to_runtime",
             input_dialect=TARGET_DIALECT, output_dialect="runtime",
             obligation="target lowering"),
)


def catalog() -> tuple[PassInfo, ...]:
    """Production in-process passes whose exercise is certified by whole-model capsules."""
    return CATALOG


def normalization_catalog() -> tuple[PassInfo, ...]:
    """Target-independent frontend/numeric normalizations; not target-obligation candidates."""
    return NORMALIZATION_CATALOG


def prototype_catalog() -> tuple[PassInfo, ...]:
    """Executable research pipeline, independently tested and never credited to production."""
    return PROTOTYPE_CATALOG


def edge_catalog() -> tuple[PassInfo, ...]:
    """Target-family-specific edge passes, audited only by campaigns that select that edge."""
    return EDGE_CATALOG


def all_catalogs() -> tuple[PassInfo, ...]:
    """Every Merlin-authored transform, for documentation and entrypoint-resolution checks."""
    return CATALOG + NORMALIZATION_CATALOG + PROTOTYPE_CATALOG + EDGE_CATALOG


def by_stage(cat: Iterable[PassInfo] | None = None) -> dict[str, list[PassInfo]]:
    out: dict[str, list[PassInfo]] = {}
    for p in (cat if cat is not None else CATALOG):
        out.setdefault(p.stage, []).append(p)
    return out


def undischarged(cat: Iterable[PassInfo] | None = None) -> list[PassInfo]:
    """Production passes with no allowed obligation."""
    return [p for p in (cat if cat is not None else CATALOG) if not p.discharges()]


def unrequired(cat: Iterable[PassInfo] | None = None) -> list[PassInfo]:
    """Production passes that no concrete capsule requires."""
    return [p for p in (cat if cat is not None else CATALOG) if not p.is_required()]


# --- invocation recording: "exercised" as a measurement, not a declaration -------------------------
# The gate must be able to say "no capsule run reaches this pass" from EVIDENCE. A catalog cannot
# know that, and a check that infers it from the catalog would be the repeat of a failure this repo
# has hit three times: a check that could not run reporting success. So invocation is recorded to a
# file, and a gate with no file reports UNMEASURED — never "clean".
PASS_LOG_ENV = "MERLIN_PASS_LOG"          # path of the JSONL invocation log; unset = no recording
PASS_LOG_CAPSULE_ENV = "MERLIN_PASS_LOG_CAPSULE"   # capsule name to attribute invocations to
PASS_LOG_REQUIREMENTS_ENV = "MERLIN_PASS_LOG_REQUIREMENTS"  # JSON list of requirement classes

_LOG_INSTALL = "install"
_LOG_INVOKE = "invoke"
_RECORDER_MARK = "_merlin_pass_recorder"

# Set by :func:`pass_run_context`; a ContextVar so concurrent capsule grades in one process cannot
# cross-attribute each other's pass runs.
_CAPSULE: contextvars.ContextVar[str | None] = contextvars.ContextVar("merlin_pass_capsule",
                                                                     default=None)
_REQUIREMENTS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "merlin_pass_requirements", default=())


def pass_log_path() -> Path | None:
    """The invocation log, or None when recording is off. Off is a REPORTABLE state, not an error."""
    raw = (os.environ.get(PASS_LOG_ENV) or "").strip()
    return Path(raw) if raw else None


def current_capsule() -> str | None:
    """Capsule an invocation belongs to: the active context, else the env, else None (unattributed)."""
    cur = _CAPSULE.get()
    if cur:
        return cur
    env = (os.environ.get(PASS_LOG_CAPSULE_ENV) or "").strip()
    return env or None


def current_requirements() -> tuple[str, ...]:
    """Requirement classes of the active capsule, with a JSON environment fallback for children."""
    current = _REQUIREMENTS.get()
    if current:
        return current
    raw = (os.environ.get(PASS_LOG_REQUIREMENTS_ENV) or "").strip()
    if not raw:
        return ()
    try:
        values = json.loads(raw)
    except ValueError:
        return ()
    if not isinstance(values, list):
        return ()
    return tuple(str(value) for value in values if str(value).strip())


@contextlib.contextmanager
def pass_run_context(capsule: str, requirements: Iterable[str] = ()):
    """Attribute every pass invocation inside the block to ``capsule``.

    A capsule runner wraps its compile call in this so the log records WHICH capsule exercised the
    pass. Without it invocations are still recorded, but attributed to nothing — which the gate
    reports as `unattributed`, never as coverage.
    """
    tok = _CAPSULE.set(capsule)
    req_tok = _REQUIREMENTS.set(tuple(str(value) for value in requirements))
    try:
        yield
    finally:
        _REQUIREMENTS.reset(req_tok)
        _CAPSULE.reset(tok)


def _append(record: dict) -> None:
    p = pass_log_path()
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        # Never fail a compile because the audit log could not be written. The missing record is
        # itself the signal: a gate that sees no install record reports "not instrumented".
        pass


def record_invocation(name: str, *, capsule: str | None = None) -> None:
    """Record that pass ``name`` ran. No-op when recording is off."""
    if pass_log_path() is None:
        return
    _append({"kind": _LOG_INVOKE, "pass": name, "capsule": capsule or current_capsule(),
             "requirements": list(current_requirements()),
             "pid": os.getpid(), "t": round(time.time(), 3)})


def install_pass_recorder(cat: Iterable[PassInfo] | None = None) -> dict[str, str]:
    """Wrap every catalogued entry point in its defining module so calls are recorded.

    Returns ``{pass name: "instrumented" | "failed: <why>"}`` and writes the same map to the log as
    an ``install`` record. That record is what lets the gate tell "this pass never ran" apart from
    "this pass was never wrapped, so we did not look" — collapsing the two would report an
    uninstrumented pipeline as a fully exercised one.

    Idempotent: an already-wrapped callable is left alone.
    """
    status: dict[str, str] = {}
    for p in (cat if cat is not None else CATALOG):
        mod_name, _, fn_name = p.entry.rpartition(".")
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            if getattr(fn, _RECORDER_MARK, False):
                status[p.name] = "instrumented"
                continue
            wrapped = _wrap(fn, p.name)
            setattr(mod, fn_name, wrapped)
            _rebind_aliases(fn, wrapped)
            status[p.name] = "instrumented"
        except Exception as e:                       # import cycles, optional deps, renamed entries
            status[p.name] = f"failed: {type(e).__name__}: {str(e)[:120]}"
    _append({"kind": _LOG_INSTALL, "passes": status, "pid": os.getpid(),
             "t": round(time.time(), 3)})
    return status


def _rebind_aliases(original: Callable, wrapped: Callable) -> int:
    """Repoint every already-imported ``from X import fn`` alias at the wrapper; returns the count.

    Patching only the defining module is not enough: ``pipeline.py`` does
    ``from .contract_facts import lower_to_contract`` at import, so a module imported BEFORE the
    recorder keeps the unwrapped reference and its calls vanish from the log — a live pass would then
    be reported dead, which is the precise lie this instrumentation exists to prevent. Rebinding by
    object identity (``is``) makes installation order irrelevant.
    """
    n = 0
    for mod in list(sys.modules.values()):
        if mod is None or not getattr(mod, "__name__", "").startswith("merlin"):
            continue
        try:
            names = [k for k, v in vars(mod).items() if v is original]
        except Exception:                            # a module with an exotic __dict__ proxy
            continue
        for k in names:
            setattr(mod, k, wrapped)
            n += 1
    return n


def _wrap(fn: Callable, name: str) -> Callable:
    @functools.wraps(fn)
    def recorded(*a, **kw):
        record_invocation(name)
        return fn(*a, **kw)

    setattr(recorded, _RECORDER_MARK, True)
    return recorded


def read_pass_log(paths: Iterable[Path]) -> dict[str, Any]:
    """Parse invocation logs into the measured facts the gate reports.

    ``instrumented`` maps a pass to the install statuses seen (a pass that never appears in an
    install record is absent — the gate calls that `not_instrumented`, which is NOT `dead`).
    ``invocations`` maps a pass to the capsules it ran under (``None`` becomes ``"unattributed"``),
    while ``requirements`` records the requirement classes those concrete capsules declared.
    """
    instrumented: dict[str, set[str]] = {}
    invocations: dict[str, set[str]] = {}
    requirements: dict[str, set[str]] = {}
    read: list[str] = []
    unreadable: dict[str, str] = {}
    for path in paths:
        p = Path(path)
        if not p.is_file():
            unreadable[str(p)] = "no such file"
            continue
        read.append(str(p))
        for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                unreadable[f"{p}:{ln}"] = "not JSON"
                continue
            kind = rec.get("kind")
            if kind == _LOG_INSTALL:
                for name, st in (rec.get("passes") or {}).items():
                    instrumented.setdefault(name, set()).add(st)
            elif kind == _LOG_INVOKE and rec.get("pass"):
                invocations.setdefault(rec["pass"], set()).add(rec.get("capsule")
                                                               or "unattributed")
                values = rec.get("requirements") or []
                if isinstance(values, list):
                    requirements.setdefault(rec["pass"], set()).update(
                        str(value) for value in values if str(value).strip())
    return {"logs_read": read, "unreadable": unreadable,
            "instrumented": {k: sorted(v) for k, v in instrumented.items()},
            "invocations": {k: sorted(v) for k, v in invocations.items()},
            "requirements": {k: sorted(v) for k, v in requirements.items()}}


def exercise_report(cat: Iterable[PassInfo] | None = None,
                    logs: Iterable[Path] | None = None) -> dict[str, Any]:
    """Per-pass exercise status measured from the logs.

    Status is one of ``exercised`` (a capsule ran it), ``exercised_unattributed`` (it ran, but no
    capsule context was set), ``dead`` (instrumented and never invoked), ``not_instrumented``
    (wrapping failed or was never installed — we did not look), ``unmeasured`` (no log at all).
    """
    passes = list(cat if cat is not None else CATALOG)
    paths = list(logs) if logs is not None else ([pass_log_path()] if pass_log_path() else [])
    parsed = read_pass_log(paths) if paths else {"logs_read": [], "unreadable": {},
                                                 "instrumented": {}, "invocations": {},
                                                 "requirements": {}}
    out: dict[str, Any] = {}
    for p in passes:
        caps = parsed["invocations"].get(p.name)
        requirements = parsed["requirements"].get(p.name, [])
        inst = parsed["instrumented"].get(p.name)
        if not parsed["logs_read"]:
            status = "unmeasured"
        elif caps:
            attributed = [c for c in caps if c != "unattributed"]
            required_hits = [value for value in requirements if value in p.required_by]
            if not attributed:
                status = "exercised_unattributed"
            elif p.required_by and not required_hits:
                status = "exercised_wrong_capsule"
            else:
                status = "exercised"
        elif inst and all(s == "instrumented" for s in inst):
            status = "dead"
        else:
            status = "not_instrumented"
        out[p.name] = {"status": status, "capsules": caps or [], "install": inst or [],
                       "requirements": requirements, "required_hits": required_hits if caps else []}
    return {"per_pass": out, "logs_read": parsed["logs_read"],
            "unreadable": parsed["unreadable"]}


# Auto-install when the operator asked for a log. Importing this module is on the staged compile path
# (`compile_core` -> `xdsl_dialects.lowering`). Installation order does not matter: `_rebind_aliases`
# repoints callers that already did `from X import fn`.
if os.environ.get(PASS_LOG_ENV):
    install_pass_recorder()
@dataclass
class DialectPlaneResult:
    """Artifacts from running the authored passes on a whole model2MLIR module."""

    module: Any                    # outlined module (driver + kernel funcs)
    dispatches: list               # list[DispatchInfo]
    program: Any                   # DispatchProgram
    partition: Any                 # PartitionResult
    stats: dict[str, Any]


def run_dialect_plane(module, forward: str | None = None, prune: bool = True, n_harts: int = 1
                      ) -> DialectPlaneResult:
    """Run the production whole-model boundary plane on a real captured module.

    The returned module is outlined and carries C wrappers; the dispatch program is verified and
    partitioned. Numeric/frontend normalization remains the caller's responsibility because it is not a
    target obligation. ``n_harts`` is explicit so a capsule records which dispatch schedule it proved.
    """
    if not HAS_XDSL:
        raise RuntimeError("xDSL is required for the dialect plane")
    from ...llvmlower.passes_xdsl import add_c_interface
    from .dispatch_program import build_dispatch_program, prune_dead_nodes, verify_program
    from .outline import outline_dispatches
    from .schedule_dispatch import partition_dispatches

    # Phase 1-2 abstraction analysis on the REAL model (contract facts -> schedule
    # decisions). Value-preserving: run on a clone, report what the compiler recognizes
    # and selects — never mutate the module that lowers. This is the "which abstractions
    # are worth exposing" plane demonstrated on a real workload (not just synthetic).
    analysis: dict = {}
    try:
        from .contract_facts import lower_to_contract
        from .schedule_decisions import lower_to_schedule
        cm = lower_to_schedule(lower_to_contract(module))
        analysis = {
            "reusable_weight_facts": sum(1 for op in cm.walk()
                                         if op.name == "contract.fact"),
            "resident_pack_required": sum(1 for op in cm.walk()
                                          if op.name == "contract.require"),
            "scheduled_resident_packs": sum(1 for op in cm.walk()
                                            if op.name == "schedule.select_interface"),
        }
    except Exception as e:                       # analysis is advisory; never block lowering
        analysis = {"error": str(e)[:160]}

    outlined = outline_dispatches(module, forward=forward)
    program = build_dispatch_program(outlined, entry=forward or "forward")
    if prune:
        program = prune_dead_nodes(program)
    problems = verify_program(program)
    if problems:
        raise RuntimeError("invalid dispatch program: " + "; ".join(problems[:5]))
    partition = partition_dispatches(program, n_harts=n_harts)
    n_ciface = add_c_interface(outlined.module)
    outlined.module.verify()
    stats = {
        "kernels": outlined.n_kernels,
        "dispatch_nodes": len(program.nodes),
        "buffers": len(program.buffers),
        "c_interface_funcs": n_ciface,
        "partition": dict(partition.stats),
        "abstraction_analysis": analysis,
    }
    return DialectPlaneResult(module=outlined.module, dispatches=outlined.dispatches,
                              program=program, partition=partition, stats=stats)
