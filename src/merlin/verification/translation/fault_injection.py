"""A corpus of seeded compiler faults, and the mutation operators that inject them.

Why seeded faults: a verification layer is justified by what it DETECTS, not by how many tests it
contains. Each operator injects one defect into the ``interface`` module the real pass produced, so a
single mutation propagates consistently to every layer downstream — the structural checks, the SMT
refinement query, and the command buffer the simulator runs. That is the same trick the RTL
pre-screen's mutation demo uses, generalized from one screen to the whole ladder.

Each fault declares which layers *should* catch it. That declaration is an expectation the evaluation
tests, not a result it assumes: a fault nothing detects is reported as an uncovered class rather than
quietly dropped.

The operators mutate IR, never text, so they cannot accidentally produce something unparseable and
score a spurious "detection".
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from merlin.target.capabilities.families import DEFAULT_EXAMPLE_TARGET

# ---- faults ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Fault:
    """One seeded defect."""

    name: str
    summary: str
    mutate: Callable[[Any], None]
    #: Layers expected to catch it. Purely documentary — the matrix reports what actually happened.
    expected: tuple[str, ...] = field(default=())


def _func_ops(module, name: str) -> list:
    func = next(o for o in module.walk() if o.name == "func.func")
    return [o for o in func.body.block.ops if o.name == name]


def _block(module):
    return next(o for o in module.walk() if o.name == "func.func").body.block

    # --- numeric faults: the arithmetic changes, the op sequence does not -----------------------------


def _miswired_commit(module) -> None:
    commits = _func_ops(module, "interface.commit")
    matmuls = _func_ops(module, "interface.matmul")
    commits[1].operands[0] = matmuls[0].results[0]


def _swapped_operands(module) -> None:
    mm = _func_ops(module, "interface.matmul")[0]
    a, b = mm.operands[0], mm.operands[1]
    mm.operands[0], mm.operands[1] = b, a


def _dropped_activation(module) -> None:
    mm = _func_ops(module, "interface.matmul")
    mm[1].operands[0] = mm[0].operands[0]

    # --- structural faults: the op sequence changes, the arithmetic may not ---------------------------


def _dropped_evict(module) -> None:
    for op in _func_ops(module, "interface.resident_evict"):
        _block(module).detach_op(op)
        op.erase()
        return


def _evict_before_last_use(module) -> None:
    blk = _block(module)
    ev = _func_ops(module, "interface.resident_evict")
    mm = _func_ops(module, "interface.matmul")
    if not ev or len(mm) < 2:
        return
    op = ev[0]
    blk.detach_op(op)
    blk.insert_op_before(op, mm[-1])


def _duplicate_pack(module) -> None:
    """Pack the same weight twice: residency was never actually established."""
    packs = _func_ops(module, "interface.resident_pack")
    if not packs:
        return
    clone = packs[0].clone()
    _block(module).insert_op_after(clone, packs[0])


def _duplicate_commit(module) -> None:
    """Commit the same accumulator twice: the commit-once invariant is violated.

    Added 2026-09-04. Every structural fault in this corpus attacked RESIDENCY, so the commit half
    of the structural layer -- the obligation `must_commit_accumulator_before_reuse` compiles to --
    had never been falsified by anything. A check no fault can make fail is not evidence, and the
    detection matrix was reporting a static layer whose commit half was untested.
    """
    commits = _func_ops(module, "interface.commit")
    if not commits:
        return
    clone = commits[0].clone()
    _block(module).insert_op_after(clone, commits[0])


def _commit_after_reuse(module) -> None:
    """Reuse the accumulator in a later matmul BEFORE its commit -- the literal obligation text.

    The op sequence becomes matmul, matmul, commit, commit rather than matmul, commit, matmul,
    commit: the accumulator is read again while still live, which is the defect the obligation
    `must_commit_accumulator_before_reuse` is named for.
    """
    blk = _block(module)
    commits = _func_ops(module, "interface.commit")
    matmuls = _func_ops(module, "interface.matmul")
    if len(commits) < 2 or len(matmuls) < 2:
        return
        # Sink the first commit past the second matmul, so the second use precedes the first commit.
    op = commits[0]
    blk.detach_op(op)
    blk.insert_op_after(op, matmuls[-1])


CORPUS: tuple[Fault, ...] = (
    Fault(
        "miswired_commit",
        "the second commit reads the first accumulator (duplicated / mis-wired commit)",
        _miswired_commit,
        expected=("formal", "dynamic"),
    ),
    Fault("swapped_matmul_operands", "A @ W emitted as W @ A", _swapped_operands, expected=("formal", "dynamic")),
    Fault(
        "dropped_activation",
        "the second matmul reuses the first activation (an input is dropped)",
        _dropped_activation,
        expected=("formal", "dynamic"),
    ),
    Fault(
        "dropped_evict", "the resident weight is never evicted (leaked residency)", _dropped_evict, expected=("static",)
    ),
    Fault(
        "evict_before_last_use",
        "the weight is evicted while a later matmul still uses it (use-after-evict)",
        _evict_before_last_use,
        expected=("static",),
    ),
    Fault(
        "duplicate_pack",
        "the weight is packed twice, so the proven reuse never happens",
        _duplicate_pack,
        expected=("static",),
    ),
    Fault(
        "duplicate_commit",
        "the same accumulator is committed twice (commit-once violated)",
        _duplicate_commit,
        expected=("static",),
    ),
    Fault(
        "commit_after_reuse",
        "the accumulator is read by a later matmul before its commit (commit-after-reuse)",
        _commit_after_reuse,
        expected=("static",),
    ),
)


# --- command-buffer faults ------------------------------------------------------------------------
# These mutate the ARTIFACT A BACKEND EMITS, not our own IR, so they are the negative controls for the
# layer that checks a compiler we did not write. They are kept separate from CORPUS because they take
# a command-buffer dict rather than an xDSL module, and because the layers they exercise differ.


def _cb_swap_matmul_operands(cb) -> None:
    """A @ W emitted as W @ A, at the command-buffer level."""
    ops = next(c for c in cb["commands"] if str(c["opcode"]).startswith("MATMUL"))["operands"]
    ops["lhs"], ops["rhs"] = ops["rhs"], ops["lhs"]


def _cb_crosswire_commit(cb) -> None:
    """The second commit reads the first accumulator: one result is silently duplicated."""
    commits = [c for c in cb["commands"] if c["opcode"] == "COMMIT"]
    commits[1]["operands"]["src"] = commits[0]["operands"]["src"]


def _cb_narrow_output(cb) -> None:
    """Narrow the readout the program did not ask to narrow.

    Worth its own fault because the reference's default is `i8` when `output_dtype` is ABSENT, so an
    emitter that simply forgets the attribute silently saturates every result. A structural check
    cannot see this at all; it is a pure value defect.
    """
    next(c for c in cb["commands"] if c["opcode"] == "COMMIT")["attributes"]["output_dtype"] = "i8"


def _cb_drop_requant(cb) -> None:
    """Drop a requant stage from the epilogue, leaving the result unscaled."""
    for c in cb["commands"]:
        if c["opcode"] == "COMMIT":
            stages = list(c.get("attributes", {}).get("epilogue") or [])
            if "requant" in stages:
                stages.remove("requant")
                c["attributes"]["epilogue"] = stages
                return
                # nothing to drop at this shape; the fault is inapplicable rather than undetectable
    raise ValueError("no requant stage present to drop")

    #: Seeded faults over an emitted command buffer. Each takes the buffer dict and mutates it in place.


CB_CORPUS: tuple[Fault, ...] = (
    Fault(
        "cb_swapped_matmul_operands",
        "the backend emitted W @ A where the program specified A @ W",
        _cb_swap_matmul_operands,
        expected=("compilation",),
    ),
    Fault(
        "cb_crosswire_commit",
        "the second commit reads the first accumulator (a duplicated result)",
        _cb_crosswire_commit,
        expected=("compilation",),
    ),
    Fault(
        "cb_narrow_output",
        "the readout saturates to i8 where the program declared a wider output",
        _cb_narrow_output,
        expected=("compilation",),
    ),
)


# Preserve historical object lookup; implementation and completion state live here.


# ---- evaluate ----------------------------------------------------------------------------------
# Run every seeded fault past every verification layer, and record what caught it, and how fast.
#
# This produces the detection matrix — the number that justifies the layers existing. A test count
# proves nothing; what a verification layer is worth is measured by the defects it catches that the
# layer below it misses, and by what it charges to catch them.
#
# Three layers run here, all locally:
#
# * **static** — the structural checks (FileCheck over the pass's real output). Milliseconds.
# * **formal** — SMT translation validation: does the program still compute the declared contraction
#   for EVERY input at this shape. Seconds.
# * **dynamic** — the emitted command buffer simulated and compared against an independent golden.
#   This is the L0 question, and it is the cheapest rung of the existing ladder.
#
# The RTL tiers (L3+) are deliberately NOT modelled here. They need hardware/simulator access this
# harness does not have, so they are reported as ``not_measured`` rather than assumed — a layer whose
# detection rate we guessed would corrupt the whole comparison.
#
# One mutation propagates to all three layers because it is applied to the ``interface`` module and the
# remaining lowering is then re-run from it, exactly as the pipeline would.

LAYERS = ("static", "formal", "dynamic")


#: The outcome states a layer attempt may report. ``abstained`` exists because a solver timeout is
#: NOT a clean run: collapsing it into ``detected=False`` makes a layer that could not answer look
#: exactly like a layer that answered "no defect here", which is the one thing this package forbids.
OUTCOMES = ("detected", "clean", "abstained", "error")


@dataclass
class Detection:
    fault: str
    layer: str
    detected: bool
    seconds: float
    diagnostic: str = ""
    #: One of :data:`OUTCOMES`. ``detected`` iff ``detected`` is True; the false case SPLITS into
    #: ``clean`` (the layer ran and found nothing) and ``abstained`` (it could not decide -- solver
    #: timeout, missing tool). A consumer that reads ``detected`` alone cannot tell those apart, so
    #: every consumer that reports coverage must read this field instead.
    outcome: str = "clean"

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome {self.outcome!r} is not one of {OUTCOMES}")
        if self.detected and self.outcome != "detected":
            raise ValueError(f"detected=True must carry outcome 'detected', got {self.outcome!r}")
        if not self.detected and self.outcome == "detected":
            raise ValueError("outcome 'detected' must carry detected=True")


def _lower_to_interface(m: int, k: int, n: int, reuse: int, *, target: str):
    from merlin.compiler.dialects.lowering.analyses import lower_to_contract
    from merlin.compiler.dialects.lowering.interface_stage import lower_to_interface, lower_to_schedule
    from merlin.compiler.dialects.lowering.pipeline import build_input_module, load_curated_contract

    tc = load_curated_contract(target)
    if tc.get("name") != target:
        raise ValueError(f"selected contract names {tc.get('name')!r}, expected {target!r}")
    mod = build_input_module(reuse=reuse, m=m, k=k, n=n)
    return lower_to_interface(lower_to_schedule(lower_to_contract(mod, tc))), tc


def _finish_lowering(interface_module, tc):
    """Continue interface -> target -> runtime -> command buffer, as the pipeline does."""
    from merlin.compiler.dialects.lowering.runtime_stage import emit_command_buffer, lower_to_runtime
    from merlin.compiler.dialects.lowering.target_stage import lower_to_target
    from merlin.target.registry.identity import backend_for

    name = tc["name"]
    backend = backend_for(name)
    target_module = lower_to_target(interface_module, None, target=name, spec=None)
    runtime_module = lower_to_runtime(target_module, target=name, backend=backend, opcodes=None)
    return emit_command_buffer(runtime_module)

    # --- the three layers -----------------------------------------------------------------------------


def _static(interface_module) -> tuple[bool, str]:
    """FileCheck the structural obligations over the module's printed form."""
    from merlin.common.paths import repo_root
    from merlin.compiler.dialects import xdsl_context as _common
    from merlin.verification.translation.smt_encoding import find_filecheck

    fc = find_filecheck()
    checks = repo_root() / "tests" / "data" / "lit" / "core" / "materialize_interface_residency.mlir"
    if fc is None or not checks.is_file():
        # The layer could not RUN. That is an abstention, never a clean result.
        return False, "abstained", "FileCheck or check file unavailable"
    r = subprocess.run([fc, str(checks)], input=_common.text(interface_module), capture_output=True, text=True)
    # non-zero = FileCheck rejected the output = the fault was DETECTED
    if r.returncode:
        return True, "detected", (r.stderr or "").strip().splitlines()[0]
    return False, "clean", ""


def _formal(interface_module, timeout_ms: int) -> tuple[bool, str, str]:
    """Refutation is the detection. A solver ``unknown`` ABSTAINS -- it is not a miss.

    Measured 2026-09-04: at 16x16x16 the three numeric faults return ``unknown`` after 73-88 s
    against a 60 s bound, while the same faults are ``sat`` in 3-5 s at 4x4x4. Recording those as
    ``detected=False`` with no further state made a timeout indistinguishable from "the layer looked
    and found nothing", which would have let a coverage figure be drawn from three timeouts.
    """
    from merlin.verification.translation.refinement import UnsupportedSemantics, validate_interface_module

    try:
        v = validate_interface_module(interface_module, timeout_ms=timeout_ms)
    except UnsupportedSemantics as exc:
        # The ENCODER refused the program; the solver never ran. That is an abstention. Counting it
        # as a refutation would credit the formal layer for a defect it never reasoned about --
        # measured 2026-09-04 on `duplicate_commit`, where "3 commits but only 2 activation
        # arguments" was being reported as DETECTED in 2 ms, which is not a proof of anything.
        return False, "abstained", f"encoder refused the program: {exc}"[:160]
    if v.refuted:
        return True, "detected", "refuted with a counterexample"
    if v.status == "unknown":
        return False, "abstained", f"solver returned unknown within {timeout_ms} ms (NOT a miss)"
    return False, "clean", "unsat"


def _dynamic(interface_module, tc, golden) -> tuple[bool, str]:
    """Simulate the emitted command buffer and compare against the independent golden."""
    from merlin.runtime import simulate

    try:
        cb = _finish_lowering(interface_module, tc)
        got = simulate(cb)["outputs"]
    except Exception as exc:  # a mutation that breaks lowering IS a detection, and we say how
        return True, "detected", f"lowering/simulation failed: {type(exc).__name__}"
    if got != golden:
        return True, "detected", "outputs differ from the golden"
    return False, "clean", ""


def _compilation(interface_module, cb, timeout_ms: int) -> tuple[bool, str, str]:
    """Does the emitted command buffer compute what the interface program specified?

    The layer that covers a compiler we did not write. Like the formal layer above, an encoder that
    cannot model something the buffer uses ABSTAINS; reporting our own incompleteness as the
    backend's defect is the one outcome that would make this worse than not checking.
    """
    from merlin.verification.translation.refinement import UnsupportedSemantics, validate_compilation

    try:
        v = validate_compilation(interface_module, cb, timeout_ms=timeout_ms)
    except UnsupportedSemantics as exc:
        return False, "abstained", f"encoder cannot model this buffer: {exc}"[:160]
    if v.refuted:
        return True, "detected", "refuted with a counterexample"
    if v.status == "unknown":
        return False, "abstained", f"solver returned unknown within {timeout_ms} ms (NOT a miss)"
    return False, "clean", "unsat"


def _dynamic_cb(cb, golden) -> tuple[bool, str, str]:
    """Simulate a command buffer directly and compare against the independent golden."""
    from merlin.runtime import simulate

    try:
        got = simulate(cb)["outputs"]
    except Exception as exc:
        return True, "detected", f"simulation failed: {type(exc).__name__}"
    if got != golden:
        return True, "detected", "outputs differ from the golden"
    return False, "clean", ""


def run_cb_matrix(
    *,
    m: int = 4,
    k: int = 4,
    n: int = 4,
    reuse: int = 2,
    timeout_ms: int = 60_000,
    faults: tuple[Fault, ...] = CB_CORPUS,
) -> dict[str, Any]:
    """Run the COMMAND-BUFFER fault corpus past the layers that can see a command buffer.

    Separate from :func:`run_matrix` because the subject is different. ``run_matrix`` mutates the
    interface module — OUR compiler's output — and asks which layers notice. This mutates the
    command buffer, which is what a BACKEND emits, so it measures the layers that can see a compiler
    we did not write: the compilation check and the numeric golden.

    The static layer is absent on purpose rather than reported as a miss: its checks are FileCheck
    patterns over MLIR text, and a command buffer is JSON. A layer that cannot look at the artifact
    at all is not a layer that looked and found nothing.
    """
    import copy

    from merlin.runtime import simulate

    clean_iface, tc = _lower_to_interface(m, k, n, reuse, target=DEFAULT_EXAMPLE_TARGET)
    clean_cb = _finish_lowering(clean_iface, tc)
    golden = simulate(clean_cb)["outputs"]

    layers = ("compilation", "dynamic")

    def _run(cb):
        out = []
        t0 = time.time()
        hit, outcome, diag = _compilation(clean_iface, cb, timeout_ms)
        out.append((("compilation"), hit, outcome, diag, time.time() - t0))
        t0 = time.time()
        hit, outcome, diag = _dynamic_cb(cb, golden)
        out.append((("dynamic"), hit, outcome, diag, time.time() - t0))
        return out

    baseline = [
        Detection("<none: unmutated>", layer, hit, secs, diag, outcome)
        for layer, hit, outcome, diag, secs in _run(copy.deepcopy(clean_cb))
    ]

    rows: list[Detection] = []
    applicable: list[Fault] = []
    inapplicable: list[dict[str, str]] = []
    for fault in faults:
        cb = copy.deepcopy(clean_cb)
        try:
            fault.mutate(cb)
        except Exception as exc:
            # The mutation does not apply at this shape (e.g. no requant stage to drop). That is a
            # property of the program, not a detection result, so it is recorded and excluded from
            # the denominator rather than counted as a fault nothing caught.
            inapplicable.append({"fault": fault.name, "reason": f"{type(exc).__name__}: {exc}"[:160]})
            continue
        applicable.append(fault)
        for layer, hit, outcome, diag, secs in _run(cb):
            rows.append(Detection(fault.name, layer, hit, secs, diag, outcome))

    return {
        "schema": "verify_cb_detection_matrix/v1",
        "subject": "the command buffer a backend emits",
        "shape": {"m": m, "k": k, "n": n, "reuse": reuse},
        "timeout_ms": timeout_ms,
        "layers": list(layers),
        "layers_not_applicable": {
            "static": "FileCheck patterns match MLIR text; a command buffer is JSON, so this layer "
            "cannot look at the artifact at all — recorded as inapplicable, not as a miss",
            "rtl": "needs a simulator/hardware this harness does not have; reported, not assumed",
        },
        "false_positives": [asdict(d) for d in baseline],
        "detections": [asdict(d) for d in rows],
        "faults": [{"name": f.name, "summary": f.summary, "expected": list(f.expected)} for f in applicable],
        "faults_inapplicable": inapplicable,
    }


def run_matrix(
    *, m: int = 4, k: int = 4, n: int = 4, reuse: int = 2, timeout_ms: int = 60_000, faults: tuple[Fault, ...] = CORPUS
) -> dict[str, Any]:
    """Run the corpus past every layer. Returns a JSON-serializable record."""
    clean_iface, tc = _lower_to_interface(m, k, n, reuse, target=DEFAULT_EXAMPLE_TARGET)
    from merlin.runtime import simulate

    golden = simulate(_finish_lowering(clean_iface, tc))["outputs"]

    # A layer that flags the UNMUTATED program is broken; measure that first.
    baseline: list[Detection] = []
    for layer, fn in (
        ("static", lambda mod: _static(mod)),
        ("formal", lambda mod: _formal(mod, timeout_ms)),
        ("dynamic", lambda mod: _dynamic(mod, tc, golden)),
    ):
        mod, _ = _lower_to_interface(m, k, n, reuse, target=DEFAULT_EXAMPLE_TARGET)
        t0 = time.time()
        hit, outcome, diag = fn(mod)
        baseline.append(Detection("<none: unmutated>", layer, hit, time.time() - t0, diag, outcome))

    rows: list[Detection] = []
    for fault in faults:
        for layer, fn in (
            ("static", lambda mod: _static(mod)),
            ("formal", lambda mod: _formal(mod, timeout_ms)),
            ("dynamic", lambda mod: _dynamic(mod, tc, golden)),
        ):
            mod, _ = _lower_to_interface(m, k, n, reuse, target=DEFAULT_EXAMPLE_TARGET)
            fault.mutate(mod)
            t0 = time.time()
            try:
                hit, outcome, diag = fn(mod)
            except Exception as exc:
                hit, outcome, diag = True, "detected", f"{type(exc).__name__}: {exc}"[:160]
            rows.append(Detection(fault.name, layer, hit, time.time() - t0, diag, outcome))

    return {
        # v2 adds the per-attempt ``outcome`` and the solver bound. The bump is deliberate: a v1
        # record cannot distinguish an abstention from a clean miss, so a reader must be able to
        # tell which schema it is holding rather than silently treating the two as comparable.
        "schema": "verify_detection_matrix/v2",
        "shape": {"m": m, "k": k, "n": n, "reuse": reuse},
        "timeout_ms": timeout_ms,
        "layers": list(LAYERS),
        "layers_not_measured": {
            "rtl": "needs a simulator/hardware this harness does not have; reported, not assumed",
        },
        "false_positives": [asdict(d) for d in baseline],
        "detections": [asdict(d) for d in rows],
        "faults": [{"name": f.name, "summary": f.summary, "expected": list(f.expected)} for f in faults],
    }


def render(record: dict[str, Any]) -> str:
    """A readable matrix. Time is per detection attempt, in seconds."""
    layers = record["layers"]
    by: dict[tuple[str, str], dict] = {(d["fault"], d["layer"]): d for d in record["detections"]}
    names = [f["name"] for f in record["faults"]]
    w = max(len(n) for n in names) + 2
    bound = record.get("timeout_ms")
    head = f"detection matrix  (shape {record['shape']}"
    head += f", solver bound {bound / 1000:.0f}s)" if bound else ")"
    out = [head, ""]
    out.append("fault".ljust(w) + "".join(l.center(18) for l in layers))
    # An abstention is printed as ABSTAIN, never as `miss`: the layer did not answer, and a reader
    # counting misses would otherwise credit a timeout as evidence of absence.
    marks = {"detected": "DETECTED", "clean": "miss", "abstained": "ABSTAIN", "error": "ERROR"}
    for n in names:
        row = n.ljust(w)
        for l in layers:
            d = by.get((n, l))
            if not d:
                row += "-".center(18)
                continue
            mark = marks.get(d.get("outcome", "detected" if d["detected"] else "clean"), "?")
            row += f"{mark} {d['seconds']:.3f}s".center(18)
        out.append(row)
    out.append("")
    abstained = [d for d in record["detections"] if d.get("outcome") == "abstained"]
    if abstained:
        out.append(f"{len(abstained)} ABSTENTION(S) -- the layer could not decide; NOT evidence of absence:")
        for d in abstained:
            out.append(f"    {d['layer']:8s} {d['fault']:26s} {d['diagnostic']}")
        out.append("")
    for d in record["false_positives"]:
        if d["detected"]:
            out.append(f"!! FALSE POSITIVE: {d['layer']} flagged the unmutated program")
    for layer, why in (record.get("layers_not_measured") or {}).items():
        out.append(f"not measured: {layer} -- {why}")
        # A layer that CANNOT look at this artifact is a different statement from one that looked and
        # found nothing, and from one we simply did not run. All three are spelled out.
    for layer, why in (record.get("layers_not_applicable") or {}).items():
        out.append(f"not applicable: {layer} -- {why}")
    for item in record.get("faults_inapplicable") or []:
        out.append(f"fault not applicable at this shape: {item['fault']} -- {item['reason']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="run the fault corpus past every verification layer")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--reuse", type=int, default=2)
    ap.add_argument(
        "--timeout-ms",
        type=int,
        default=60_000,
        help="solver bound per formal attempt; recorded in the artifact, because a "
        "detection count is uninterpretable without the bound it was measured under",
    )
    ap.add_argument("--json", action="store_true", help="emit the record instead of the table")
    ap.add_argument("--write", action="store_true", help="write a versioned product under out/artifacts")
    args = ap.parse_args(argv)

    rec = run_matrix(m=args.m, k=args.k, n=args.n, reuse=args.reuse, timeout_ms=args.timeout_ms)
    print(json.dumps(rec, indent=1) if args.json else render(rec))
    if args.write:
        _write_product(rec)
    return 0


def _write_product(rec: dict[str, Any]) -> Path:
    """Persist the matrix as a versioned product, with provenance that says what it measured.

    Uses the ProductDir API rather than the bare path: ``new_product`` returns an object whose
    ``add_artifact``/``write_manifest`` write the manifest the layout gate requires. A product dir
    with no manifest, or one whose ``sources`` is empty, is worse than no artifact at all once
    somebody cites the numbers — there is then nothing recording WHAT they were measured against.
    """
    import subprocess

    from merlin.common.artifacts import new_product
    from merlin.verification.translation.smt_encoding import find_filecheck, find_mlir_tool

    def _ver(path: str | None, flag: str) -> str:
        if not path:
            return "unavailable"
        try:
            r = subprocess.run([path, flag], capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                return "unknown"
            return (r.stdout or r.stderr).strip().splitlines()[0]
        except Exception:
            return "unknown"

    z3_version = "unavailable"
    try:
        import z3

        z3_version = f"z3 {z3.get_version_string()}"
    except Exception:
        pass

    shape = rec["shape"]
    prod = new_product(
        "verification",
        version=1,
        sources=[
            f"fault corpus: merlin.verification.translation.fault_injection.CORPUS ({len(rec['faults'])} seeded faults)",
            f"program under test: merlin.compiler.dialects.lowering.pipeline.lower_repeated_rhs_matmul"
            f"(m={shape['m']}, k={shape['k']}, n={shape['n']}, reuse={shape['reuse']})",
            "static layer checks: tests/data/lit/core/materialize_interface_residency.mlir",
            f"solver: {z3_version} (bound {rec.get('timeout_ms', 'UNKNOWN')} ms per attempt)",
            f"exporter: {_ver(find_mlir_tool('mlir-translate'), '--version')}",
            f"matcher: {_ver(find_filecheck(), '--version')}",
        ],
        notes=(
            "Fault-detection matrix for the compiler-verification layers. Each seeded fault is injected "
            "into the interface module the real pass produced and the remaining lowering is re-run from "
            "it, so ONE mutation reaches all three layers consistently. The RTL tiers are recorded as "
            "not measured (no simulator access in this harness) rather than assumed — a guessed "
            "detection rate would corrupt the comparison. See docs/design/compiler_verification.md."
        ),
    )

    out = prod.add_artifact("detection_matrix.json")
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    prod.write_manifest()
    print(f"\nwrote {out}")
    return out

    # Preserve historical object lookup; implementation and completion state live here.


if __name__ == "__main__":
    raise SystemExit(main())
