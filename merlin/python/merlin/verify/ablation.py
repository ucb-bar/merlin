"""What does the formal layer buy over the dynamic grade we already run?

This is an ABLATION, not a coverage report. The seeded fault matrix in :mod:`merlin.verify.evaluate`
already answers "does the checker work". The question here is different and is the one a reviewer
asks: on artifacts this project has *already produced and already graded*, does proving equivalence
find anything the numeric check did not? It needs no agent, no capsule generation and no spend --
every input is an archived capsule-bench submission sitting under ``out/runs/``.

**The two things being compared.**

* The DYNAMIC layer (what we run today) executes the submitted buffer on one stimulus and compares it
  to the golden. ``Tensor.deterministic`` indexes its fill by ``(row, col)``, so rows and columns do
  differ -- but every value it produces is drawn from ``{0, 1, 2, 3}``: four of the 256 i8 values, all
  non-negative. Sign handling, i8 saturation, accumulator overflow and negative-operand paths are
  therefore not exercised, by construction rather than by oversight.
* The FORMAL layer (:func:`merlin.verify.refine.validate_equivalence`) proves the submitted buffer
  denotes the same function as the interface program it was handed, for *every* input at that shape.

The hypothesis is that the second finds defects in the gap left by the first. It is falsifiable: if
nothing is found there, the gap is real but empty, and that is the reportable answer.

**The eligibility split is the load-bearing part of this module.** Parsing an interface program back
into a command buffer and comparing it with the submission shows that most submissions *are* the
program they were handed, command for command. An equivalence theorem over two copies of one program
is ``X == X`` -- trivially ``unsat``, verifying nothing, and it is also the case where a bug in the
shared encoder cancels completely. Measured on the archived corpus:

    IDENTICAL                                   2500   60.8%   excluded
    different opcode sequence                    990   24.1%   eligible
    same opcodes, different operands/attributes   621   15.1%   eligible

Reporting all 4,111 as verified would have been 61% vacuous. That is the same shape as a gate wired to
nothing, and it is visible only because the split is measured here rather than assumed -- see the
``checks-that-skip-and-report-success`` family.

**What an abstention means.** The encoder models integer datapaths of rank <= 2. A rank-4 convolution,
a ``bf16`` or ``f8E4M3FN`` tensor, or an ``acc_scale`` epilogue (an IEEE-754 round trip) has no exact
bitvector encoding here, so the verdict is ``abstained`` -- never a pass, and never a refutation. The
report breaks abstentions out by reason so the coverage limit is a number rather than a footnote.

Run it::

    .venv/bin/python -m merlin.verify.ablation --timeout-ms 15000 --workers 6 --write
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

#: The three files a unit needs. The grade record sits one level above the generated dir.
SPEC_NAME = "input.interface.mlir"
BUFFER_NAME = "command_buffer.json"
GRADE_NAME = "capsule_result.json"

#: Only capsule-bench submissions. Other run families under ``out/runs`` (mesh_run, recipe-select)
#: carry command buffers produced by the IN-TREE lowering, not by a backend under evaluation, so
#: including them would answer a different question than the one this module asks.
POPULATION_GLOB = "capsule-bench/**/generated/" + BUFFER_NAME

#: The values ``Tensor.deterministic`` can produce (``lo=0, hi=3``). A counterexample outside this set
#: is an input the dynamic stimulus cannot construct, so no amount of re-running it would find that
#: defect. Read from the tensor module rather than repeated here, so the two cannot drift.
def stimulus_values() -> set[int]:
    """The distinct values the default stimulus emits. Derived, not assumed."""
    from merlin.runtime.tensor import Tensor

    vals: set[int] = set()
    for name in ("A0", "W", "X", "B"):
        for shape in ((8, 8), (16, 16)):
            vals |= set(Tensor.deterministic(name, shape, "i8").data)
    return vals


#: Largest symbolic product (sum of M*K*N over contractions) this module will hand to the encoder.
#:
#: The solver budget bounds SOLVING, not ENCODING: ``to_smtlib`` materialises the whole bit-blasted
#: term before z3's timeout can apply, so a large contraction consumes unbounded wall time BEFORE the
#: budget is consulted. Measured 2026-09-05: a first full run left six workers pegged for ten minutes
#: on its last ~70 units and never finished. A declared cap turns that into a stated, reported
#: abstention (``too_large``) instead of an accidental hang -- the limit is the same either way, but
#: this way it appears in the coverage table rather than in the wall clock.
MAX_ENCODED_MACS = 200_000


#: Largest total declared tensor footprint (elements) this module will encode. A MAC cap alone is
#: not enough: it prices only ``MATMUL``, and the first capped run still left six workers pegged for
#: twelve minutes on 70 buffers whose cost lived in ATTENTION/VECTOR_MAP chains instead. Every
#: encodable op emits terms proportional to the tensors it touches, so bounding the footprint bounds
#: them all -- including the next opcode someone adds.
MAX_ENCODED_ELEMENTS = 400_000


def declared_elements(cb: dict) -> int:
    """Total elements across the buffer's declared tensors, whatever the opcodes do with them."""
    total = 0
    for spec in (cb.get("tensors") or {}).values():
        shape = (spec or {}).get("shape") or []
        n = 1
        for extent in shape:
            try:
                n *= max(int(extent), 1)
            except (TypeError, ValueError):
                n = 0
                break
        total += n
    return total


def encoded_size(cb: dict) -> int:
    """Sum of M*K*N over the buffer's contractions -- the cost driver for bit-blasting."""
    tensors = cb.get("tensors") or {}
    total = 0
    for cmd in (cb.get("commands") or []):
        if str(cmd.get("opcode")) not in ("MATMUL", "MATMUL_RESIDENT"):
            continue
        operands = cmd.get("operands") or {}
        lhs = (tensors.get(str(operands.get("lhs"))) or {}).get("shape") or []
        rhs = (tensors.get(str(operands.get("rhs"))) or {}).get("shape") or []
        if len(lhs) == 2 and len(rhs) == 2:
            total += int(lhs[0]) * int(lhs[1]) * int(rhs[1])
        elif len(lhs) == 2:
            # rhs is a resident handle, not a declared tensor; price it as a square of the K extent,
            # which is the best bound available without resolving the pack source.
            total += int(lhs[0]) * int(lhs[1]) * int(lhs[1])
    return total


def units(root: Path | None = None) -> list[Path]:
    """Every archived submission carrying a spec, a buffer and a grade. Sorted, so runs repeat."""
    from merlin.common.paths import runs_dir

    base = Path(root) if root is not None else runs_dir()
    found: list[Path] = []
    for buf in base.rglob(POPULATION_GLOB):
        gen = buf.parent
        if (gen / SPEC_NAME).is_file() and (gen.parent / GRADE_NAME).is_file():
            found.append(gen.parent)
    return sorted(found)


def _norm(commands: Any) -> list[tuple]:
    """A command list reduced to what an equivalence question depends on, in order."""
    return [(c.get("opcode"),
             json.dumps(c.get("operands"), sort_keys=True),
             json.dumps(c.get("attributes"), sort_keys=True))
            for c in (commands or [])]


def classify(spec_cb: dict, agent_cb: dict) -> str:
    """``identical`` / ``operands`` / ``opcodes`` -- how far the submission moved from its spec.

    ``identical`` is the exclusion: the submission reproduces the interface program command for
    command, so proving them equivalent proves nothing about a compiler.
    """
    spec, agent = _norm(spec_cb.get("commands")), _norm(agent_cb.get("commands"))
    if spec == agent:
        return "identical"
    if [c[0] for c in spec] == [c[0] for c in agent]:
        return "operands"
    return "opcodes"


def read_unit(unit: Path) -> dict[str, Any]:
    """Load one unit's three artifacts. Returns a record with ``load_error`` set on failure."""
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    rec: dict[str, Any] = {"unit": str(unit), "capsule": unit.name}
    try:
        rec["grade"] = json.loads((unit / GRADE_NAME).read_text(encoding="utf-8"))
    except Exception as exc:
        rec["load_error"] = f"grade unreadable: {type(exc).__name__}"
        return rec
    try:
        rec["agent_cb"] = json.loads((unit / "generated" / BUFFER_NAME).read_text(encoding="utf-8"))
    except Exception as exc:
        rec["load_error"] = f"buffer unreadable: {type(exc).__name__}"
        return rec
    try:
        rec["spec_cb"] = parse_interface_mlir(
            (unit / "generated" / SPEC_NAME).read_text(encoding="utf-8"))
    except Exception as exc:
        # The contract grammar fails closed on an op it does not define; that is a parser limit on
        # OUR side, so it is an abstention reason, not a defect in the submission.
        rec["load_error"] = f"spec unparseable: {type(exc).__name__}: {exc}"[:220]
        return rec
    rec["shape"] = classify(rec["spec_cb"], rec["agent_cb"])
    return rec


def _grade_axes(grade: dict) -> dict[str, Any]:
    """The dynamic verdicts this ablation compares against.

    ``numeric`` is the axis, because it is the dynamic counterpart of the formal claim. ``status`` is
    the whole-capsule outcome and is recorded beside it, but a capsule that failed to BUILD says
    nothing about whether its command buffer computes the right function.
    """
    numeric = (grade.get("numeric") or {})
    tiers = (grade.get("tiers") or {})
    return {
        "numeric_status": str(numeric.get("status") or "absent"),
        "numeric_mismatches": numeric.get("mismatch_count"),
        "overall_status": str(grade.get("status") or "absent"),
        "l0_status": str((tiers.get("L0") or {}).get("status") or "absent"),
        "l0_reason": ((tiers.get("L0") or {}).get("reason") or None),
    }


class _WallTimeout(Exception):
    """A unit exceeded its hard wall bound."""


def _alarm(seconds: int):
    """Hard per-unit wall bound, as a context manager. Returns a no-op outside the main thread.

    Neither of the two budgets already in place actually bounds a unit. The solver timeout bounds
    SOLVING; the size caps bound the two cost drivers we thought of. Measured twice on 2026-09-05:
    both times the run reached 4,100 of 4,170 and then pegged six workers for over ten minutes on the
    remaining 70 -- first without a cap, then with a MAC cap, then with a tensor-footprint cap. Each
    cap removed the cause it named and the tail survived. A wall alarm stops asking WHICH cost driver
    is unbounded and bounds the unit itself, so the tail becomes a reported abstention instead of a
    hang. That is the same lesson as the polling-loop guard: bound the thing, do not enumerate the
    reasons it might not terminate.
    """
    import contextlib
    import signal
    import threading

    @contextlib.contextmanager
    def _ctx():
        if threading.current_thread() is not threading.main_thread():
            yield
            return

        def _fire(signum, frame):
            raise _WallTimeout(f"exceeded the {seconds}s hard wall bound")

        old_handler = signal.signal(signal.SIGALRM, _fire)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return _ctx()


def check_unit(unit_str: str, timeout_ms: int, wall_seconds: int = 90) -> dict[str, Any]:
    """Verdict for one unit. Never raises: a crash here must not take the population down."""
    from merlin.verify.refine import OutputContractViolation, validate_equivalence
    from merlin.verify.smt_semantics import UnsupportedSemantics

    unit = Path(unit_str)
    t0 = time.time()
    rec = read_unit(unit)
    if "load_error" in rec:
        return {"unit": unit_str, "capsule": unit.name, "verdict": "abstained",
                "reason": rec["load_error"], "reason_kind": "unreadable",
                "shape": "unknown", "seconds": round(time.time() - t0, 2)}

    out = {"unit": unit_str, "capsule": unit.name, "shape": rec["shape"],
           **_grade_axes(rec["grade"])}
    if rec["shape"] == "identical":
        # Excluded before the solver ever sees it, so the exclusion costs nothing and is auditable.
        out.update(verdict="excluded", reason="submission is its own specification, command for "
                                              "command; the query would be X == X",
                   reason_kind="vacuous", seconds=round(time.time() - t0, 2))
        return out
    size = max(encoded_size(rec["spec_cb"]), encoded_size(rec["agent_cb"]))
    elems = max(declared_elements(rec["spec_cb"]), declared_elements(rec["agent_cb"]))
    out["encoded_macs"], out["declared_elements"] = size, elems
    if elems > MAX_ENCODED_ELEMENTS:
        out.update(verdict="abstained", reason_kind="too_large",
                   reason=(f"the buffer declares {elems:,} tensor elements, over the cap of "
                           f"{MAX_ENCODED_ELEMENTS:,}; encoding is unbounded by the solver budget"),
                   seconds=round(time.time() - t0, 2))
        return out
    if size > MAX_ENCODED_MACS:
        out.update(verdict="abstained", reason_kind="too_large",
                   reason=(f"encoding this buffer would bit-blast {size:,} MAC terms, over the "
                           f"declared cap of {MAX_ENCODED_MACS:,}. The solver budget bounds solving, "
                           f"not encoding, so this would consume unbounded wall time before the "
                           f"timeout applied."),
                   seconds=round(time.time() - t0, 2))
        return out
    try:
        with _alarm(wall_seconds):
            v = validate_equivalence(rec["spec_cb"], rec["agent_cb"], timeout_ms=timeout_ms)
        verdict = {"unsat": "verified", "sat": "refuted"}.get(v.status, "abstained")
        out.update(verdict=verdict, reason="" if verdict != "abstained" else "solver returned unknown",
                   reason_kind="" if verdict != "abstained" else "solver_timeout")
        if verdict == "refuted":
            values = dict(v.model_values or {})
            out["counterexample"] = {k: values[k] for k in sorted(values)[:40]}
            stim = stimulus_values()
            outside = {k: x for k, x in values.items() if isinstance(x, int) and x not in stim}
            # The ablation's second headline: could the existing stimulus have reached this input?
            out["counterexample_outside_stimulus"] = bool(outside)
            out["counterexample_outside_examples"] = dict(sorted(outside.items())[:6])
    except OutputContractViolation as exc:
        # A definite negative, so it is a refutation -- but one with no counterexample, because no
        # input is needed to see it. Bucketed separately so it is never confused with a numeric
        # divergence in the report.
        out.update(verdict="refuted", reason=str(exc)[:300], reason_kind="output_contract",
                   counterexample_outside_stimulus=None)
    except _WallTimeout as exc:
        out.update(verdict="abstained", reason=str(exc), reason_kind="wall_timeout")
    except UnsupportedSemantics as exc:
        out.update(verdict="abstained", reason=str(exc)[:220], reason_kind=_reason_kind(str(exc)))
    except Exception as exc:
        out.update(verdict="error", reason=f"{type(exc).__name__}: {exc}"[:220],
                   reason_kind="error")
    out["seconds"] = round(time.time() - t0, 2)
    return out


def _reason_kind(reason: str) -> str:
    """Bucket an abstention so the coverage limit is a table rather than 1,600 strings."""
    low = reason.lower()
    if "rank" in low:
        return "rank_gt_2"
    if "dtype" in low or "float" in low:
        return "float_dtype"
    if "epilogue" in low:
        return "epilogue"
    if "commits" in low:
        return "output_count"
    if "is (" in low or "shape" in low:
        return "shape_mismatch"
    return "other"


def load_pin(path: Path) -> list[str]:
    """Read a pinned population list: one unit path per line, ``#`` comments ignored."""
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def write_pin(path: Path, pool: list[str]) -> None:
    """Freeze the population so a rerun draws the same units.

    This is not bookkeeping. The archive GROWS while other sessions run the bench -- enumerating twice
    twenty minutes apart gave 4,111 and then 4,170 units, and a population that moves under a sample
    is how the historical replay published three different numbers before it was pinned. Freezing the
    list makes a later, larger run an extension of this one rather than a different experiment
    wearing its name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# merlin.verify.ablation population pin\n"
                    f"# {len(pool)} units enumerated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                    + "\n".join(pool) + "\n", encoding="utf-8")


def run(*, timeout_ms: int = 15_000, workers: int = 6, limit: int | None = None,
        seed: int = 11, root: Path | None = None, pin: Path | None = None,
        wall_seconds: int = 90, progress: bool = True) -> dict[str, Any]:
    """Check every eligible submission. Returns the record the report is rendered from.

    ``limit`` draws a seeded shuffle-then-take rather than a prefix, so raising it EXTENDS the sample
    instead of replacing it -- a larger run is then a superset and cannot be a quiet reroll after an
    unwelcome result. The default is no limit: the whole population is affordable, which removes the
    sampling question entirely.

    ``pin`` reads the population from a frozen list instead of enumerating, so a rerun is over the
    same units even though the archive keeps growing underneath.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if pin is not None and pin.is_file():
        pool = [u for u in load_pin(pin) if Path(u).is_dir()]
        pinned_from = str(pin)
    else:
        pool = [str(u) for u in units(root)]
        pinned_from = None
        if pin is not None:
            write_pin(pin, pool)
            pinned_from = f"{pin} (written now)"
    total_population = len(pool)
    if limit is not None:
        ordered = list(pool)
        random.Random(seed).shuffle(ordered)
        pool = ordered[:limit]

    records: list[dict[str, Any]] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(check_unit, u, timeout_ms, wall_seconds): u for u in pool}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                records.append(fut.result())
            except Exception as exc:  # a worker died; record it rather than losing the unit
                records.append({"unit": futures[fut], "capsule": Path(futures[fut]).name,
                                "verdict": "error", "shape": "unknown",
                                "reason": f"worker died: {type(exc).__name__}: {exc}"[:200],
                                "reason_kind": "error", "seconds": 0.0})
            if progress and i % 100 == 0:
                print(f"  {i}/{len(pool)}  ({time.time() - started:.0f}s)", file=sys.stderr)

    return {
        "schema": "verify_ablation/v1",
        "question": "does proving equivalence find defects the numeric grade passed?",
        "population_total": total_population,
        "population_pin": pinned_from,
        "sampled": len(pool),
        "seed": seed if limit is not None else None,
        "timeout_ms": timeout_ms,
        "stimulus_values": sorted(stimulus_values()),
        "wall_seconds": round(time.time() - started, 1),
        "records": sorted(records, key=lambda r: (r.get("capsule", ""), r.get("unit", ""))),
    }


def render(record: dict[str, Any]) -> str:
    """The report. Every pre-declared cell is printed, including the empty ones.

    Printing a zero cell matters: "no refutation survived adjudication" and "we never looked" are
    different claims, and a table that omits its empty rows cannot distinguish them.
    """
    recs = record["records"]
    out: list[str] = []
    add = out.append

    add(f"# formal-vs-dynamic ablation over archived submissions")
    add(f"question: {record['question']}")
    if record.get("population_pin"):
        add(f"population pinned from {record['population_pin']}")
    add(f"population {record['population_total']} archived submissions"
        + (f"; sampled {record['sampled']} (seed {record['seed']})"
           if record.get("seed") is not None else "; all checked"))
    add(f"solver budget {record['timeout_ms']} ms per unit; wall {record['wall_seconds']}s")
    add(f"dynamic stimulus draws its values from {record['stimulus_values']}")
    add("")

    by_shape: dict[str, int] = {}
    for r in recs:
        by_shape[r.get("shape", "unknown")] = by_shape.get(r.get("shape", "unknown"), 0) + 1
    total = max(len(recs), 1)
    add("## eligibility (measured before any verdict was drawn)")
    labels = {"identical": "IDENTICAL to its own spec -- EXCLUDED, the query would be X == X",
              "opcodes": "different opcode sequence -- eligible",
              "operands": "same opcodes, different operands/attrs -- eligible",
              "unknown": "unreadable -- reported, not dropped"}
    for shape in ("identical", "opcodes", "operands", "unknown"):
        n = by_shape.get(shape, 0)
        add(f"  {n:6d}  {100 * n / total:5.1f}%  {labels[shape]}")
    eligible = [r for r in recs if r.get("verdict") != "excluded"]
    add(f"\neligible population: {len(eligible)}")
    add("")

    add("## verdict x numeric grade (eligible only)")
    grades = sorted({str(r.get("numeric_status", "absent")) for r in eligible})
    verdicts = ["verified", "refuted", "abstained", "error"]
    width = max((len(g) for g in grades), default=6) + 2
    add("  " + "verdict".ljust(12) + "".join(g.ljust(width) for g in grades) + "total")
    for v in verdicts:
        row = [r for r in eligible if r.get("verdict") == v]
        cells = "".join(str(sum(1 for r in row if str(r.get("numeric_status", "absent")) == g))
                        .ljust(width) for g in grades)
        add("  " + v.ljust(12) + cells + str(len(row)))
    add("")

    refuted = [r for r in eligible if r.get("verdict") == "refuted"]
    verified = [r for r in eligible if r.get("verdict") == "verified"]
    abstained = [r for r in eligible if r.get("verdict") == "abstained"]

    add("## headline 1 -- refuted, but the numeric grade PASSED")
    add("   Each of these is EITHER an encoder bug OR a real escape. Not reportable as a win until")
    add("   adjudicated by hand: a refutation of a correct backend is our bug until proven otherwise.")
    escapes = [r for r in refuted if str(r.get("numeric_status")) == "pass"]
    if not escapes:
        add(f"   {len(escapes)} -- none. On this population the formal layer found nothing the")
        add("   numeric check passed. That is the answer, not a gap in the run.")
    for r in escapes:
        outside = r.get("counterexample_outside_stimulus")
        add(f"   {r['capsule']:34s} outside-stimulus={outside}  {r['unit']}")
    add("")

    add("## headline 2 -- could the existing stimulus have reached the counterexample?")
    if refuted:
        outside = [r for r in refuted if r.get("counterexample_outside_stimulus")]
        add(f"   {len(outside)} of {len(refuted)} refutations need an input outside "
            f"{record['stimulus_values']},")
        add("   so re-running the dynamic check on its own stimulus could not have found them.")
    else:
        add("   no refutations, so this metric has no denominator on this population")
    add("")

    add("## headline 3 -- abstentions (coverage limit, never a pass)")
    kinds: dict[str, int] = {}
    for r in abstained:
        kinds[str(r.get("reason_kind") or "other")] = kinds.get(str(r.get("reason_kind") or "other"), 0) + 1
    add(f"   {len(abstained)} of {len(eligible)} eligible "
        f"({100 * len(abstained) / max(len(eligible), 1):.1f}%)")
    for kind, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        add(f"     {n:6d}  {kind}")
    add("")

    add("## the formal layer's blind spot -- verified, but the capsule FAILED")
    blind = [r for r in verified if str(r.get("numeric_status")) not in ("pass", "absent")]
    add(f"   {len(blind)} submissions compute the specified function and still failed numerically.")
    add("   A disagreement here is worth reading closely: it means the golden and this encoder")
    add("   disagree about the same buffer, and one of them is wrong.")
    for r in blind[:12]:
        add(f"     {r['capsule']:34s} numeric={r.get('numeric_status')} "
            f"mismatches={r.get('numeric_mismatches')}")
    if len(blind) > 12:
        add(f"     ... and {len(blind) - 12} more")
    add("")

    errors = [r for r in eligible if r.get("verdict") == "error"]
    if errors:
        add(f"## errors -- {len(errors)} unit(s) crashed the checker; these are OUR defects")
        seen: dict[str, int] = {}
        for r in errors:
            seen[str(r.get("reason"))[:90]] = seen.get(str(r.get("reason"))[:90], 0) + 1
        for reason, n in sorted(seen.items(), key=lambda kv: -kv[1])[:8]:
            add(f"     {n:4d}  {reason}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeout-ms", type=int, default=15_000, help="solver budget per unit")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel workers; this is a SHARED host, keep it modest")
    ap.add_argument("--limit", type=int, default=None,
                    help="seeded shuffle-then-take; omit to check the whole population")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--wall-seconds", type=int, default=90,
                    help="hard per-unit wall bound; the solver budget does not cover encoding")
    ap.add_argument("--root", default=None, help="run root (default: merlin.common.paths.runs_dir())")
    ap.add_argument("--pin", default=None,
                    help="population pin file; read if it exists, written on first use. The archive "
                         "grows while other sessions run the bench, so an unpinned rerun is over a "
                         "different population.")
    ap.add_argument("--json", action="store_true", help="print the record instead of the report")
    ap.add_argument("--write", action="store_true", help="write the record under out/artifacts/")
    a = ap.parse_args(argv)

    rec = run(timeout_ms=a.timeout_ms, workers=a.workers, limit=a.limit, seed=a.seed,
              root=Path(a.root) if a.root else None, pin=Path(a.pin) if a.pin else None,
              wall_seconds=a.wall_seconds)
    print(json.dumps(rec, indent=1) if a.json else render(rec))

    if a.write:
        from merlin.common.artifacts import new_product

        prod = new_product("verification", version=1, target="all", sources=[
            f"{rec['population_total']} archived capsule-bench submissions under out/runs",
            "formal: merlin.verify.refine.validate_equivalence",
            "dynamic: capsule_result.json numeric tier",
        ], notes=("Ablation: does proving equivalence find defects the numeric grade passed? "
                  "Structurally identical submissions are EXCLUDED -- a query over two copies of "
                  "one program is X == X and verifies nothing."))
        prod.add_artifact("ablation.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        prod.add_artifact("ablation.txt").write_text(render(rec), encoding="utf-8")
        prod.write_manifest()
        print(f"\nwrote {prod.path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
