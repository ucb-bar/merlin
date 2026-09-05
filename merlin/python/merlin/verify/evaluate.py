"""Run every seeded fault past every verification layer, and record what caught it, and how fast.

This produces the detection matrix — the number that justifies the layers existing. A test count
proves nothing; what a verification layer is worth is measured by the defects it catches that the
layer below it misses, and by what it charges to catch them.

Three layers run here, all locally:

* **static** — the structural checks (FileCheck over the pass's real output). Milliseconds.
* **formal** — SMT translation validation: does the program still compute the declared contraction
  for EVERY input at this shape. Seconds.
* **dynamic** — the emitted command buffer simulated and compared against an independent golden.
  This is the L0 question, and it is the cheapest rung of the existing ladder.

The RTL tiers (L3+) are deliberately NOT modelled here. They need hardware/simulator access this
harness does not have, so they are reported as ``not_measured`` rather than assumed — a layer whose
detection rate we guessed would corrupt the whole comparison.

One mutation propagates to all three layers because it is applied to the ``interface`` module and the
remaining lowering is then re-run from it, exactly as the pipeline would.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .faults import CORPUS, Fault

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


def _lower_to_interface(m: int, k: int, n: int, reuse: int):
    from merlin.xdsl_dialects.lowering import pipeline
    from merlin.xdsl_dialects.lowering.contract_facts import lower_to_contract
    from merlin.xdsl_dialects.lowering.interface_lowering import lower_to_interface
    from merlin.xdsl_dialects.lowering.pipeline import build_input_module, load_curated_contract
    from merlin.xdsl_dialects.lowering.schedule_decisions import lower_to_schedule

    tc = load_curated_contract("toy_npu")
    mod = build_input_module(reuse=reuse, m=m, k=k, n=n)
    return lower_to_interface(lower_to_schedule(lower_to_contract(mod, tc))), tc


def _finish_lowering(interface_module, tc):
    """Continue interface -> target -> runtime -> command buffer, as the pipeline does."""
    from merlin.targetgen.target_registry import backend_for
    from merlin.xdsl_dialects.lowering.emit_command_buffer import emit_command_buffer
    from merlin.xdsl_dialects.lowering.runtime_lowering import lower_to_runtime
    from merlin.xdsl_dialects.lowering.target_lowering import lower_to_target

    name = tc["name"]
    backend = backend_for(name)
    target_module = lower_to_target(interface_module, None, target=name, spec=None)
    runtime_module = lower_to_runtime(target_module, target=name, backend=backend, opcodes=None)
    return emit_command_buffer(runtime_module)


# --- the three layers -----------------------------------------------------------------------------

def _static(interface_module) -> tuple[bool, str]:
    """FileCheck the structural obligations over the module's printed form."""
    from merlin.common.paths import merlin_dir
    from merlin.verify.tools import find_filecheck
    from merlin.xdsl_dialects import _common

    fc = find_filecheck()
    checks = merlin_dir() / "tests" / "data" / "lit" / "core" / "materialize_interface_residency.mlir"
    if fc is None or not checks.is_file():
        # The layer could not RUN. That is an abstention, never a clean result.
        return False, "abstained", "FileCheck or check file unavailable"
    r = subprocess.run([fc, str(checks)], input=_common.text(interface_module),
                       capture_output=True, text=True)
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
    from .refine import UnsupportedSemantics, validate_interface_module

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


def run_matrix(*, m: int = 4, k: int = 4, n: int = 4, reuse: int = 2,
               timeout_ms: int = 60_000, faults: tuple[Fault, ...] = CORPUS) -> dict[str, Any]:
    """Run the corpus past every layer. Returns a JSON-serializable record."""
    clean_iface, tc = _lower_to_interface(m, k, n, reuse)
    from merlin.runtime import simulate

    golden = simulate(_finish_lowering(clean_iface, tc))["outputs"]

    # A layer that flags the UNMUTATED program is broken; measure that first.
    baseline: list[Detection] = []
    for layer, fn in (("static", lambda mod: _static(mod)),
                      ("formal", lambda mod: _formal(mod, timeout_ms)),
                      ("dynamic", lambda mod: _dynamic(mod, tc, golden))):
        mod, _ = _lower_to_interface(m, k, n, reuse)
        t0 = time.time()
        hit, outcome, diag = fn(mod)
        baseline.append(Detection("<none: unmutated>", layer, hit, time.time() - t0, diag, outcome))

    rows: list[Detection] = []
    for fault in faults:
        for layer, fn in (("static", lambda mod: _static(mod)),
                          ("formal", lambda mod: _formal(mod, timeout_ms)),
                          ("dynamic", lambda mod: _dynamic(mod, tc, golden))):
            mod, _ = _lower_to_interface(m, k, n, reuse)
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
        "faults": [{"name": f.name, "summary": f.summary, "expected": list(f.expected)}
                   for f in faults],
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
    for layer, why in record["layers_not_measured"].items():
        out.append(f"not measured: {layer} -- {why}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="run the fault corpus past every verification layer")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--reuse", type=int, default=2)
    ap.add_argument("--timeout-ms", type=int, default=60_000,
                    help="solver bound per formal attempt; recorded in the artifact, because a "
                         "detection count is uninterpretable without the bound it was measured under")
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
    from .tools import find_filecheck, find_mlir_tool

    def _ver(path: str | None, flag: str) -> str:
        if not path:
            return "unavailable"
        try:
            r = subprocess.run([path, flag], capture_output=True, text=True, timeout=30)
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
    prod = new_product("verification", version=1, sources=[
        f"fault corpus: merlin.verify.faults.CORPUS ({len(rec['faults'])} seeded faults)",
        f"program under test: merlin.xdsl_dialects.lowering.pipeline.lower_repeated_rhs_matmul"
        f"(m={shape['m']}, k={shape['k']}, n={shape['n']}, reuse={shape['reuse']})",
        f"static layer checks: merlin/tests/data/lit/core/materialize_interface_residency.mlir",
        f"solver: {z3_version} (bound {rec.get('timeout_ms', 'UNKNOWN')} ms per attempt)",
        f"exporter: {_ver(find_mlir_tool('mlir-translate'), '--version')}",
        f"matcher: {_ver(find_filecheck(), '--version')}",
    ], notes=(
        "Fault-detection matrix for the compiler-verification layers. Each seeded fault is injected "
        "into the interface module the real pass produced and the remaining lowering is re-run from "
        "it, so ONE mutation reaches all three layers consistently. The RTL tiers are recorded as "
        "not measured (no simulator access in this harness) rather than assumed — a guessed "
        "detection rate would corrupt the comparison. See docs/design/compiler_verification.md."))

    out = prod.add_artifact("detection_matrix.json")
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    prod.write_manifest()
    print(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
