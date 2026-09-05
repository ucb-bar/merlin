#!/usr/bin/env python
"""K1 WHOLE-MODEL PER-OP PROFILE: attribute model wall time to the ops the model executes.

WHY. The only whole-model instrumentation this repo had was a TWO-WAY split — a matmul bucket
(``-DMERLIN_DISPATCH_TIMING`` inside the routed GEMM shim) vs "everything else". Measured on the
K1, matmul is 1.3-6 % of a model once the kernel is fast (dispatch_breakdown_measured.json), so
94-97 % of model time had never been attributed to anything. This driver produces the missing
attribution and ranks what is left on the table by MEASURED milliseconds.

METHOD (default-OFF; the un-instrumented build is byte-identical).
``merlin.llvmlower.op_profile`` interleaves a one-instruction-cost ``@merlin_prof_mark(id)`` call
between the top-level ops of ``func.func @forward`` (plus a sentinel before ``func.return``); the
shim ``runtime/c/merlin_op_prof.c`` samples ``rdtime`` at each mark and credits the elapsed ticks
to the previous op's id. ``build_k1_binary(..., op_profile=True)`` splices this in AFTER any
kernel-backend rewrite and links the shim; the harness prints ``PROF <id> <ticks> <hits>``.
``run_on_k1`` joins those against the id->op table (``prov.*`` provenance) it emitted at build.

Each op record carries the cross-compiler JOIN KEY (:func:`op_profile.join_key`): ``prov.fqn`` when
present (the deepest ``nn.Module`` path that aligns a Merlin region with the SAME layer in an
ExecuTorch/GGUF/ONNX export — see ``baselines/contract.py`` / ``_et_export.py``), else
``prov.region_id``, else the MLIR op name. So this breakdown is a framework-agnostic
graph-node -> IR -> emitted-asm -> measured-cycles chain, not a Merlin-only scheme.

PERTURBATION GUARD (the profiler must not move what it measures). Every model is ALSO built and run
WITHOUT instrumentation (byte-identical baseline path). We report the profiled-vs-unprofiled median
wall delta and FAIL the perturbation check if it exceeds the board noise floor (--noise, default
1.9 %). A breakdown whose own instrumentation moved the wall past the floor is reported as
``perturbation_ok=false`` and must not be trusted.

COVERAGE GATE (the profiler must not attribute more time than it measured). The perturbation guard
and the coverage gate are INDEPENDENT, and the first does not imply the second: this tool once
reported 6.0 ms of attributed op time against a 4.2 ms wall — 144 % — while the perturbation guard
was clean at 0.11 %. The instrumentation had not moved the wall; the ticks were divided by the wrong
number of executions. So every bucket now carries ``share_of_attributed``, and carries
``share_of_runtime`` ONLY when ``op_profile.coverage_report`` puts the coverage inside
``op_profile.COVERAGE_BAND``. Outside the band the tool prints the ranking, refuses the percentages,
and says why.

THE DENOMINATOR IS PER EXECUTION, NOT PER ITERATION. ``ms_avg`` is the cost of ONE execution of
``@forward``; the harness's wall is per TIMED ITERATION. Those are the same window only for a plain
model. A version-1 session bundle declares ``steps: N`` and its harness runs ``merlin_run`` once per
step inside a single timed iteration, so the wall is divided by the executions-per-iteration count
(derived from the shim's own hit count by ``op_profile.executions_per_iteration``, never assumed)
before anything is expressed as a share. Getting this wrong in one direction produced the 144 %
artifact; getting it wrong in the other would refuse a sound 256-step profile at coverage ~0.004.

FAIL-CLOSED, AND GRADED BY THE RIGHT RULES. Every run is scored by ``zephyr_model._gate`` against
TIER-KEYED references (``fp32`` -> golden.npy, ``w8a8`` -> golden_w8a8.npy for an int8 package); a
run that does not gate is ``not_run`` and its wall is never recorded. The tier KEYS choose the
thresholds, so handing the W8A8 golden over as ``fp32`` — which this driver used to do, on top of an
fp32-strict cos of 0.9999 — grades a quantized build by fp32 rules and rejects a correct one. The
gate verdicts (tiers, tier_ok, comparison_complete, n_compared/n_reference) are recorded per launch.
An instrumented run that comes back with an EMPTY op table, or with a table whose hits are all zero,
is a blocker too — nothing was measured, which is not the same as a model with nothing to measure.

WHAT IT CANNOT ATTRIBUTE. A mark interval is one whole top-level op, so allocator cost (hoisted),
fork/join (below one op) and anything intra-op are outside its resolution; the artifact carries
``breakdown.categories_not_attributable`` naming each and how to measure it instead.

Run (board serialized internally — do NOT wrap in board_lock):
  MERLIN_COMPILE_TIMEOUT_S=3600 .venv/bin/python build_tools/scripts/k1_op_profile.py \
      --model out/artifacts/recaptures/bitvla_fp32_consistent -n 3
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from merlin.common.paths import artifacts_dir
from merlin.llvmlower import op_profile as opf
from merlin.mining import k1
from merlin.mining.registry import load_rvv_package
from merlin.runtime.backends import zephyr_model as zm

TIMEBASE_HZ = k1.K1_TIMEBASE_HZ  # rdtime tick rate (24 MHz)
NS_PER_TICK = 1e9 / TIMEBASE_HZ

# "is this op vectorized?" is answered by its RESOLVED family (op_profile.resolve_family): the
# default K1 RVV pipeline (llvmlower.pipeline.build_rvv_pipeline) vectorizes only the contraction
# family; every other family goes through convert-linalg-to-loops -> scalar. So the sum of the
# non-contraction ticks IS the scalar work left on the table.


def _ticks_to_ms(ticks: float) -> float:
    return ticks * NS_PER_TICK / 1e6


def _rvv_coverage(binary: Path) -> dict | None:
    """Disassemble the built binary and report the STATIC vector coverage of the compiler-emitted
    ``forward`` symbol (cross-check for the family-based ``vectorized`` intent flag). This is a
    static instruction count, not a dynamic one — but a low coverage on ``forward`` proves the
    emitted compute is scalar-dominated regardless of what family a op claims. Best-effort: returns
    None if objdump/the binary is unreadable (honest, never a silent 'fully vectorized')."""
    try:
        from merlin.baselines.rvv_audit import audit_binary
        rep = audit_binary(binary)
        fwd = rep.by_symbol.get("forward")
        if fwd is None:
            return None
        return {"symbol": "forward", "vector_insns": fwd.vector,
                "scalar_compute_insns": fwd.scalar_compute,
                "static_vector_coverage": fwd.coverage,
                "note": ("STATIC instruction coverage of the emitted `forward` (one inlined "
                         "function). Low coverage => the compute is scalar-dominated (addressing, "
                         "memrefCopy, tails) even for ops whose family is 'contraction'.")}
    except Exception:  # noqa: BLE001
        return None


def gate_run(prefix, golden_refs: dict, *, cos_th: float | None, rel_th: float | None,
             tag: str) -> tuple[bool, str | None, dict]:
    """Score one board run against the TIER-KEYED references and say whether it counts.

    WHY TIER-KEYED. ``zephyr_model._gate`` decides which tier carries the verdict from the KEYS of
    the references dict: ``w8a8`` is graded by W8A8 rules (T1: cos > 0.999 + rel < 1e-2 + no
    per-element blow-up) and ``fp32`` by fp32 rules (T2: cos > 0.99 + argmax + no blow-up). Handing a
    W8A8 golden over under the ``fp32`` key — which this driver did — grades int8 output by fp32
    rules AND hides that the w8a8 tier was never in play. That is the failure that cost a multi-hour
    hunt for a TinyLlama int8 "board bug" that did not exist (see ``k1_int8_fair_compare``), and it
    is why a correct W8A8 build could not get past this tool's own gate. The tiers here are built by
    the same rule the fair-compare arm uses, so a model that passes the study's gate passes this one.

    ``cos_th``/``rel_th`` are OPTIONAL EXTRA strictness on top of the tier verdict, not a replacement
    for it: the old default (cos >= 0.9999, rel <= 1e-2, applied to whichever single reference was
    loaded) is an fp32 threshold and rejects a correct quantized build outright. Default None = the
    tier gate's own verdict decides.

    Returns ``(ok, blocker, gate_dict)``. ``gate_dict`` is recorded verbatim in the artifact,
    including ``comparison_complete`` — the board prints at most ``dump_cap`` output elements and
    the gate scores that PREFIX, so a partial comparison must never read as a whole-output verdict.
    """
    g = zm._gate(prefix, golden_refs)
    ok = bool(g.get("ok"))
    detail = (f"tiers={g.get('tiers')} tier_ok={g.get('tier_ok')} "
              f"cos={g.get('cos')} rel={g.get('rel')}")
    blocker = None if ok else f"gate failed: {detail}"
    if ok and cos_th is not None and (g.get("cos") is None or g["cos"] < cos_th):
        ok, blocker = False, f"gate failed (--cos {cos_th}): {detail}"
    if ok and rel_th is not None and g.get("rel") is not None and g["rel"] > rel_th:
        ok, blocker = False, f"gate failed (--rel {rel_th}): {detail}"
    if not g.get("comparison_complete", True):
        # NOT a blocker -- but it is a different claim, and it must travel with the number.
        print(f"  [{tag}] PARTIAL OUTPUT: the gate compared {g.get('n_compared')} of "
              f"{g.get('n_reference')} elements ({g.get('compared_fraction', 0):.1%}) — this cos/rel "
              f"is a PREFIX score, not the model's accuracy. Raise --dump-cap to close it.",
              flush=True)
    print(f"  [{tag}] {'GATED' if ok else 'NOT_GATED'} {detail}", flush=True)
    return ok, blocker, {k: v for k, v in g.items() if isinstance(v, (int, float, bool, str))}


def _median(xs: list[int]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def run_profiled(model_dir: Path, pkg, golden_refs: dict, n: int, work_root: Path,
                 kernel_backend: str | None = None, cos_th: float | None = None,
                 rel_th: float | None = None, iters: int = 1, warmup: int = 0,
                 max_session_steps: int | None = None,
                 dump_cap: int | None = 4096) -> dict:
    """Build+run the instrumented binary n times; gate cos; accumulate per-op ticks.

    The per-op ticks are SUMMED across the gated runs then divided by the number of executions the
    shim COUNTED (`hits`), so a single op faster than one rdtime tick (~41.7 ns) still accumulates
    signal across runs. Returns the per-op table with per-execution ticks/ms plus the run-level
    walls and profiler-coverage metrics.
    """
    walls: list[int] = []
    time_ticks: list[int] = []
    prof_totals: list[int] = []
    per_op_ticks: dict[int, int] = defaultdict(int)
    per_op_hits: dict[int, int] = defaultdict(int)
    launch_execs: list[int] = []
    table: list[dict] | None = None
    gates: list[dict] = []
    gated = 0
    blocker = None
    for i in range(n):
        work = work_root / f"on_{i}"
        try:
            res = k1.run_on_k1(model_dir, work, pkg, timeout=1800, op_profile=True,
                               kernel_backend=kernel_backend, iters=iters, warmup=warmup,
                               max_session_steps=max_session_steps, dump_cap=dump_cap)
            ok, why, gate = gate_run(res["prefix"], golden_refs, cos_th=cos_th, rel_th=rel_th,
                                     tag=f"on {i}")
            gates.append(gate)
            if not ok:
                blocker = why
                continue
            m = res["metrics"]
            walls.append(m["wall_ns"])
            time_ticks.append(m.get("time_ticks", 0))
            tbl = res.get("op_profile") or []
            if table is None:
                table = tbl
            # HOW MANY TIMES did @forward actually run inside this launch? Ask the shim, do not
            # assume `--iters`. The accumulators run from process start, so they also cover the
            # UNTIMED warmup passes -- and a session model runs several steps per timed iteration.
            # Dividing by `iters` when the shim counted `warmup + iters` is what made this tool
            # report 6.0 ms of op time against a 4.2 ms wall (coverage 1.4407 = exactly 7/5 x 1.029).
            execs = max((int(r.get("hits") or 0) for r in tbl), default=0)
            launch_execs.append(execs)
            if execs:
                prof_totals.append(m.get("prof_total_ticks", 0) / execs)
            for rec in tbl:
                per_op_ticks[rec["id"]] += rec.get("ticks", 0)
                # SUM, not assign: ticks are summed across launches, so hits must be too or the
                # divisor stays at one launch's count while the numerator grows with n.
                per_op_hits[rec["id"]] += rec.get("hits", 0)
            gated += 1
            # `prof_cov` here is the SHIM's own raw ratio (its accumulated total against the
            # harness's timed ticks), printed live as a smoke signal. It is NOT the coverage the
            # artifact reports: that one divides by the measured executions per iteration. They
            # differ by exactly the window ratio, which is the point of the whole gate.
            print(f"  [on {i}] wall_ns={m['wall_ns']} execs={execs} "
                  f"raw_prof_ratio={m.get('prof_total_ticks',0)/max(m.get('time_ticks',1),1):.3f} "
                  f"marks={m.get('prof_marks')}")
        except Exception as e:  # noqa: BLE001
            blocker = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  [on {i}] BLOCKED — {blocker}")
            break
    if table is None or gated == 0:
        return {"ok": False, "blocker": blocker or "no gated run", "n_gated": 0,
                "gates": gates}
    # FAIL CLOSED on an EMPTY table. A build path that never instrumented the IR (or a console the
    # PROF lines never reached) leaves `res["op_profile"]` absent, `tbl` empty, and everything below
    # divides zero by zero: the tool then reported `ok: true` with a breakdown of nothing, which
    # reads as "this model has no attributable work" rather than as "the profiler was not linked in".
    empty = opf.table_blocker(table)
    if empty:
        return {"ok": False, "n_gated": gated, "blocker": empty, "gates": gates}
    for rec in table:
        rec["ticks"] = per_op_ticks[rec["id"]]
        rec["hits"] = per_op_hits[rec["id"]]
    # Per-execution ticks/ms, resolved family, join key. The divisor is the op's own measured hit
    # count -- see op_profile.per_execution_ticks and the module comment above it.
    opf.annotate_table(table, timebase_hz=TIMEBASE_HZ)
    execs_per_launch = int(_median(launch_execs)) if launch_execs else None
    return {"ok": True, "n_gated": gated, "table": table, "gates": gates,
            "walls": walls, "median_wall_ns": _median(walls),
            "median_time_ticks": _median(time_ticks),
            "median_prof_total_ticks": _median(prof_totals),
            "executions_per_launch": execs_per_launch,
            "timed_iterations_per_launch": max(1, iters),
            "warmup_per_launch": warmup,
            # How many @forward executions ONE TIMED ITERATION contains. 1 for a plain model; the
            # session bundle's declared `steps` (e.g. 256) for a v1 session, whose harness runs
            # merlin_run once per step inside a single timed iteration. Derived from the shim's own
            # count, never assumed -- see op_profile.executions_per_iteration.
            "executions_per_timed_iteration": opf.executions_per_iteration(
                execs_per_launch, max(1, iters), warmup),
            "rvv_coverage": _rvv_coverage(work_root / "on_0" / "v" / "merlin_k1"),
            "blocker": None}


def run_unprofiled(model_dir: Path, pkg, golden_refs: dict, n: int, work_root: Path,
                   kernel_backend: str | None = None, cos_th: float | None = None,
                   rel_th: float | None = None, iters: int = 1, warmup: int = 0,
                   max_session_steps: int | None = None,
                   dump_cap: int | None = 4096) -> dict:
    """Build+run the byte-identical UN-instrumented binary n times (the perturbation control)."""
    walls: list[int] = []
    gates: list[dict] = []
    gated = 0
    blocker = None
    for i in range(n):
        work = work_root / f"off_{i}"
        try:
            res = k1.run_on_k1(model_dir, work, pkg, timeout=1800, op_profile=False,
                               kernel_backend=kernel_backend, iters=iters, warmup=warmup,
                               max_session_steps=max_session_steps, dump_cap=dump_cap)
            ok, why, gate = gate_run(res["prefix"], golden_refs, cos_th=cos_th, rel_th=rel_th,
                                     tag=f"off {i}")
            gates.append(gate)
            if not ok:
                blocker = why
                continue
            walls.append(res["metrics"]["wall_ns"])
            gated += 1
            print(f"  [off {i}] wall_ns={res['metrics']['wall_ns']}")
        except Exception as e:  # noqa: BLE001
            blocker = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  [off {i}] BLOCKED — {blocker}")
            break
    return {"ok": gated > 0, "n_gated": gated, "walls": walls, "gates": gates,
            "median_wall_ns": _median(walls), "blocker": blocker}


def build_breakdown(model_dir: Path, prof: dict, unprof: dict, noise_pct: float) -> dict:
    """Roll the per-op table up, and REFUSE to express it as runtime percentages when it cannot be.

    Two rules this function exists to enforce, both of which it once broke:

    1. THE DENOMINATOR IS DECLARED. Every bucket carries ``share_of_attributed``; it carries
       ``share_of_runtime`` only when :func:`op_profile.coverage_report` says the profile's coverage
       is inside the sane band. A 144 %-coverage profile still gets a ranking (the ranking was
       right -- the transposes really were the prize) but no percentage-of-runtime, because that
       number would be inflated by an unstated factor.
    2. NO FALSE ZERO. Buckets resolve their family through :func:`op_profile.resolve_family`, which
       falls back to the MLIR op name. Bucketing on ``prov.family`` alone put 776 of 996 ops in
       ``(none)`` and printed ``contraction = 0.0 ms`` for a model whose matmuls measured 1.36 ms.
    """
    table = prof["table"]
    attributed_ticks = opf.sum_attributed_ticks(table)
    total_ms = _ticks_to_ms(attributed_ticks)
    wall_ticks = prof.get("median_time_ticks")
    per_iter = prof.get("executions_per_timed_iteration")
    coverage = opf.coverage_report(
        attributed_ticks, wall_ticks,
        executions=prof.get("executions_per_launch"),
        timed_iterations=prof.get("timed_iterations_per_launch"),
        executions_per_timed_iteration=per_iter)
    # The runtime denominator is handed to the rollups ONLY when the coverage check passed -- and it
    # is the wall of ONE @forward EXECUTION, which is what `ms_avg` is per. For a plain model that is
    # the timed-iteration wall; for a v1 session bundle one timed iteration runs @forward once per
    # declared step, so the iteration wall must be divided by that count first (coverage_report does
    # the same division, and both must use the same one or the shares and the gate disagree).
    wall_ticks_per_exec = coverage.get("wall_ticks_per_execution")
    wall_ms = (_ticks_to_ms(wall_ticks_per_exec)
               if wall_ticks_per_exec and coverage["runtime_shares_reportable"] else None)
    cov_scalar = coverage["profiler_coverage"]

    by_family = opf.rollup(table, lambda r: r["family_resolved"], "family", wall_ms=wall_ms,
                           coverage=cov_scalar)
    # THE DECISION BUCKETS. `family` is the frontend's vocabulary; `category` is the one a lever is
    # aimed at (op_profile.resolve_category) -- the quantize/requant chain pulled out of the three
    # places it hides, softmax separated from layer_norm, a materializing layout copy separated from
    # a metadata-only view. This is the rollup the ranking question is actually asked in.
    by_category = opf.rollup(table, lambda r: r["category"], "category", wall_ms=wall_ms,
                             coverage=cov_scalar)
    by_op = opf.rollup(table, lambda r: r.get("mlir_op") or "(none)", "mlir_op", wall_ms=wall_ms,
                       coverage=cov_scalar)
    by_key = opf.rollup(table, lambda r: r["join_key"], "join_key", wall_ms=wall_ms,
                        coverage=cov_scalar)
    by_aten = opf.rollup(table, lambda r: r.get("aten") or "(none)", "aten", wall_ms=wall_ms,
                         coverage=cov_scalar)

    # scalar (non-contraction) vs vectorized (contraction) split — the headline "left on table".
    measured = [r for r in table if r.get("ms_avg") is not None]
    scalar_ms = sum(r["ms_avg"] for r in measured if not r["vectorized"])
    vector_ms = sum(r["ms_avg"] for r in measured if r["vectorized"])
    # The one honest hole: an untagged `linalg.generic` may BE a contraction (the integer datapath
    # rewrites captured contractions into generics with an i8xi8->i32 body and no prov.family), and
    # the profiler sees only the printed op line. Surfaced as its own quantity so the contraction
    # bucket reads as a LOWER BOUND rather than as a complete measurement.
    unclassified_ms = sum(r["ms_avg"] for r in measured if opf.is_unclassified_generic(r))

    # perturbation check: profiled vs unprofiled median wall.
    pon = prof.get("median_wall_ns")
    poff = unprof.get("median_wall_ns")
    pert = None
    if pon and poff:
        delta_pct = 100.0 * (pon - poff) / poff
        # ONE-SIDED test: instrumentation can only ADD cost, so a perturbation problem is the
        # profiled wall being SLOWER than the control by MORE than the noise floor. A delta at or
        # below the floor (incl. negative — profiled faster) is noise, not a measurable perturbation.
        #
        # A CLEAN perturbation result says nothing about attribution. The 144 % profile passed this
        # guard at delta=0.11 %: the instrumentation genuinely did not move the wall, and the ticks
        # were still divided by the wrong number of executions. The two checks are independent and
        # both must pass — see `coverage`.
        pert = {"profiled_median_wall_ns": pon, "unprofiled_median_wall_ns": poff,
                "delta_pct": round(delta_pct, 3), "noise_floor_pct": noise_pct,
                "perturbation_ok": delta_pct <= noise_pct,
                "note": ("delta = profiled/unprofiled-1. Instrumentation can only add cost, so the "
                         "one-sided guard fails only when profiled is SLOWER than the un-instrumented "
                         "control by more than the board noise floor. |delta|<=floor (or negative) "
                         "means the profiler did not measurably move the wall. It does NOT mean the "
                         "attribution is sound — check `coverage` for that.")}

    # top ops by ms (individual dispatches).
    top = sorted(measured, key=lambda r: r["ms_avg"], reverse=True)[:40]
    top_ops = [{k: r.get(k) for k in ("id", "mlir_op", "family", "family_resolved",
                                      "family_source", "category", "category_source", "op", "aten",
                                      "role", "region_id", "fqn",
                                      "join_key", "result_type", "elems", "hits", "executions",
                                      "ms_avg", "vectorized")} for r in top]
    for r in top_ops:
        r["share_of_attributed"] = (r["ms_avg"] / total_ms) if total_ms else None
        r["share_of_runtime"] = (r["ms_avg"] / wall_ms) if wall_ms else None
        r["share_denominator"] = "wall" if wall_ms else "attributed"
        r["profiler_coverage"] = cov_scalar

    # THE HEADLINE COMPARISON, computed rather than left to a reader to assemble from two rows: is
    # the quantize chain really more expensive than every contraction combined? A static census says
    # its `roundeven` chain emits ~2.4x the instructions of every matmul; this is the same question
    # asked in MEASURED milliseconds. Both sides are stated with their op counts and the unknown
    # that could still move them, so the ratio is never quoted without its error bars.
    _by_cat = {r["category"]: r for r in by_category}
    _chain_ms = sum(_by_cat[c]["ms"] for c in opf.QUANTIZE_CHAIN_CATEGORIES if c in _by_cat)
    _contr_ms = _by_cat.get("contraction", {}).get("ms", 0.0)
    _unknown_ms = _by_cat.get("unclassified_generic", {}).get("ms", 0.0)
    quantize_vs_contraction = {
        "quantize_chain_ms": round(_chain_ms, 3),
        "quantize_chain_categories": list(opf.QUANTIZE_CHAIN_CATEGORIES),
        "contraction_ms": round(_contr_ms, 3),
        "ratio": (_chain_ms / _contr_ms) if _contr_ms else None,
        "unclassified_generic_ms": round(_unknown_ms, 3),
        "note": ("Measured ms, not instructions. Both sides are LOWER BOUNDS in opposite ways: an "
                 "untagged linalg.generic the body rules could not read sits in "
                 "unclassified_generic and may belong to either, and a contraction the int8 rewrite "
                 "left untagged sits there too. Quote the ratio with unclassified_generic_ms beside "
                 "it, and only when coverage says the shares are of runtime."),
    }

    return {
        "quantize_vs_contraction": quantize_vs_contraction,
        "total_attributed_ms": round(total_ms, 3),
        "measured_wall_ms": round(_ticks_to_ms(wall_ticks), 3) if wall_ticks else None,
        # The denominator every `share_of_runtime` below is a share OF: the wall of one @forward
        # execution. Recorded separately from `measured_wall_ms` (one TIMED ITERATION) because for a
        # session bundle they differ by the step count, and conflating them is the window mismatch.
        "wall_ms_per_execution": (round(_ticks_to_ms(wall_ticks_per_exec), 6)
                                  if wall_ticks_per_exec else None),
        "executions_per_timed_iteration": per_iter,
        "coverage": coverage,
        # Back-compat scalar for existing readers; `coverage` is the authoritative block.
        "profiler_coverage": coverage["profiler_coverage"],
        "runtime_shares_reportable": coverage["runtime_shares_reportable"],
        "share_denominator": coverage["share_denominator"],
        "vectorized_flag_meaning": ("`vectorized`/`contraction_intended_ms` reflect the pipeline's "
                                    "INTENT: the K1 RVV schedule vectorizes only the contraction "
                                    "family; every other family lowers to scalar loops. It is NOT "
                                    "proof the emitted asm is vector — see `rvv_coverage` for the "
                                    "measured static coverage of `forward`. The family is resolved "
                                    "from prov.family when tagged and from the MLIR op name when "
                                    "not (`family_source` says which)."),
        "rvv_coverage": prof.get("rvv_coverage"),
        "contraction_intended_ms": round(vector_ms, 3), "scalar_ms": round(scalar_ms, 3),
        "contraction_is_a_lower_bound": unclassified_ms > 0,
        "unclassified_generic_ms": round(unclassified_ms, 3),
        "unclassified_generic_note": ("untagged `linalg.generic` — family genuinely UNKNOWN. Some "
                                      "of these are contractions the integer datapath rewrote "
                                      "without re-stamping provenance, so `contraction_intended_ms` "
                                      "is a lower bound, not a measurement of all contraction time."),
        "scalar_share_of_attributed": round(scalar_ms / total_ms, 4) if total_ms else None,
        "scalar_share_of_runtime": (round(scalar_ms / wall_ms, 4) if wall_ms else None),
        "perturbation": pert,
        "by_category": by_category,
        "category_order": list(opf.CATEGORY_ORDER),
        "categories_not_attributable": opf.CATEGORIES_NOT_ATTRIBUTABLE,
        "by_family": by_family, "by_aten": by_aten, "by_mlir_op": by_op,
        "top_ops": top_ops,
        "by_join_key_top": by_key[:40],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/artifacts/recaptures/bitvla_fp32_consistent")
    ap.add_argument("--baseline", default="out/artifacts/targets/rvv/hand_v0")
    ap.add_argument("-n", type=int, default=3,
                    help="separate build+deploy launches; the min across them is reported")
    ap.add_argument("--iters", type=int, default=1,
                    help="TIMED inferences inside one launch; the harness reports the PER-ITERATION "
                         "cost. Default 1 = a single COLD inference, which is what every number this "
                         "script produced before this flag existed.")
    ap.add_argument("--warmup", type=int, default=0,
                    help="UNTIMED inferences before the timed ones. Required for a sustained "
                         "comparison: ExecuTorch's runner has no warmup and averages its cold first "
                         "execution into --num_executions, so measuring ours cold against that is "
                         "comparing a cold number to a mostly-warm one -- and it penalises US.")
    ap.add_argument("--cos", type=float, default=None,
                    help="OPTIONAL extra strictness on top of the tier gate. The verdict is the "
                         "tier gate's own (T1 w8a8: cos>0.999 + rel<1e-2 + per-element; T2 fp32: "
                         "cos>0.99 + argmax + per-element). The old default of 0.9999 was an fp32 "
                         "threshold applied to a QUANTIZED run and rejected correct W8A8 builds.")
    ap.add_argument("--rel", type=float, default=None,
                    help="OPTIONAL extra per-element strictness on top of the tier gate (which "
                         "already carries its own per-element blow-up veto).")
    ap.add_argument("--dump-cap", type=int, default=4096,
                    help="output elements the board prints, and therefore the CEILING on how much "
                         "of the answer the accuracy verdict covers -- the gate scores the printed "
                         "PREFIX. 0 means print the model's whole output (comparison_complete). "
                         "Default 4096 keeps the binary byte-identical to every previous run; a "
                         "partial comparison is reported, never silent.")
    ap.add_argument("--noise", type=float, default=1.9, help="board noise floor %% for perturbation")
    ap.add_argument("--max-session-steps", type=int, default=None,
                    help="cap how many steps of a SESSION bundle's corpus are embedded and run. A "
                         "version-1 session runs @forward once per step inside ONE timed iteration, "
                         "so this is also the executions-per-iteration count the coverage gate "
                         "divides the wall by (derived from the shim, not from this flag). Needed "
                         "for the large corpora: a 256-step, 154 MB session_inputs.npz becomes a "
                         "770 MB model_io.h that costs ~7 GB RSS to compile. Recorded in the "
                         "artifact, because it changes WHAT was measured (fewer steps = a colder "
                         "average).")
    ap.add_argument("--kernel-backend", default=None,
                    help="route matmuls to a fast expert kernel (xnnpack/openblas/ours) so the "
                         "per-op breakdown isolates the NON-matmul 94%% when the GEMM is fast")
    ap.add_argument("--features", default=None,
                    help="comma-separated default-off compiler features to enable on BOTH the "
                         "profiled and control builds (e.g. fuse_transpose_b). OMITTED -> the "
                         "package's own compiler_features, i.e. the lowering it was certified with. "
                         "Pass an empty string for the frozen baseline lowering. Run once with and "
                         "once without a feature to get its before/after whole-model profile.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.dump_cap = None if a.dump_cap == 0 else a.dump_cap

    md = Path(a.model)
    base = load_rvv_package(a.baseline)
    # The reference has to match the DATAPATH, not just the bundle. An int8 bundle's golden.npy is
    # weight-only (int8 weights dequantized into an f32 contraction); a W8A8 build also quantizes the
    # activations and is graded against golden_w8a8.npy. Gating a W8A8 run on the weight-only golden
    # fails cos for a correct build, which reads as "the model regressed" rather than "the wrong
    # reference was loaded". Fail closed when the package is int8 and no W8A8 reference exists.
    # TIER-KEYED, because `_gate` reads the verdict rules off these KEYS: `w8a8` is graded by W8A8
    # rules and `fp32` by fp32 rules. Loading the W8A8 golden and handing it over as `fp32` -- which
    # is what this driver did -- grades a quantized build by fp32 thresholds and hides that the w8a8
    # tier was never in play. Pass BOTH when both exist, so T1 (w8a8) and T2 (fp32 + argmax) are
    # live and the derived quantization-excess veto is measurable. Same construction as the
    # fair-compare arm, so a build that passes the study's gate passes this one.
    golden_refs: dict = {}
    if (md / "golden.npy").is_file():
        golden_refs["fp32"] = np.load(md / "golden.npy")
    if base.is_int8:
        w8a8 = md / "golden_w8a8.npy"
        if not w8a8.is_file():
            raise SystemExit(f"{a.baseline} is an int8 package but {md.name} ships no "
                             f"golden_w8a8.npy — recapture it rather than grading W8A8 output "
                             f"against the weight-only golden (a missing w8a8 tier reads as a "
                             f"codegen defect)")
        golden_refs["w8a8"] = np.load(w8a8)
    if not golden_refs:
        raise SystemExit(f"{md} ships no golden to gate against")
    print(f"[golden] tiers={sorted(golden_refs)} (package is_int8={base.is_int8})")
    # A package's compiler_features ARE part of what it is: the champion int8 package carries
    # `accumulator_resident_wholemodel_vf`, and replacing that with [] compiles the BASELINE lowering
    # while still calling itself that package. Measured on deepjscc W8A8, the discarded feature is the
    # difference between gating and cos=0.9176 -- which reads as a model regression rather than as the
    # profiler having built something else. So an omitted --features means "whatever this package was
    # certified with"; an explicitly empty one still means the frozen baseline.
    feats = ([f.strip() for f in a.features.split(",") if f.strip()] if a.features is not None
             else list(base.compiler_features or []))
    print(f"[features] {feats or '<baseline>'} "
          f"({'from the package' if a.features is None else 'from --features'})")
    # distinct run_ids so the two builds get distinct /tmp deploy paths on the shared board
    # (tr_ prefix per the shared-board tag convention; backend + feature-tag suffix keeps native vs
    # routed and baseline vs feature-on builds apart so before/after never collide on /tmp).
    bk = (a.kernel_backend or "native")
    ftag = "base" if not feats else "_".join(sorted(feats))[:40]
    pkg_on = replace(base, run_id=f"tr_opon_{bk}_{ftag}", compiler_features=feats)
    pkg_off = replace(base, run_id=f"tr_opoff_{bk}_{ftag}", compiler_features=feats)

    # NOT /tmp: this host's root filesystem is small and holds /tmp, while the build tree for a
    # whole model is gigabytes (a session model's model_io.h alone reaches hundreds of MB). Honor
    # TMPDIR when it is set to a big-enough volume, and fall back to the build root under out/.
    import os as _os

    from merlin.common.paths import build_dir
    _tmp = _os.environ.get("MERLIN_OPPROF_WORK") or _os.environ.get("TMPDIR")
    _root = Path(_tmp) if _tmp else (build_dir() / "opprofile")
    work_root = _root / f"tr_opprofile_{md.name}_{bk}_{ftag}"
    work_root.mkdir(parents=True, exist_ok=True)

    kb = a.kernel_backend
    print(f"=== {md.name}: profiled (instrumented, kernel_backend={kb}) x{a.n} ===")
    prof = run_profiled(md, pkg_on, golden_refs, a.n, work_root, kernel_backend=kb,
                        iters=a.iters, warmup=a.warmup,
                        max_session_steps=a.max_session_steps, dump_cap=a.dump_cap,
                        cos_th=a.cos, rel_th=a.rel)
    print(f"=== {md.name}: unprofiled control x{a.n} ===")
    unprof = run_unprofiled(md, pkg_off, golden_refs, a.n, work_root, kernel_backend=kb,
                            iters=a.iters, warmup=a.warmup,
                            max_session_steps=a.max_session_steps, dump_cap=a.dump_cap,
                            cos_th=a.cos, rel_th=a.rel)

    summary: dict = {
        "model": str(md), "board": "k1_spacemit", "vlen": k1.VLEN, "kernel_backend": kb,
        "compiler_features": feats,
        "timer": "rdtime per-op marks + CLOCK_MONOTONIC wall_ns; cycle_accurate=false",
        "timebase_hz": TIMEBASE_HZ, "n": a.n, "cos_threshold": a.cos, "rel_threshold": a.rel,
        "iters": a.iters, "warmup": a.warmup, "max_session_steps": a.max_session_steps,
        "dump_cap": a.dump_cap, "golden_tiers": sorted(golden_refs),
        "method": ("Per-op rdtime marks interleaved between the top-level ops of @forward "
                   "(default-off; un-instrumented build byte-identical). Ticks credited to the "
                   "previous op; keyed on prov.fqn||region_id (the cross-compiler join key). "
                   "Perturbation-gated vs an un-instrumented control within the board noise floor."),
        "profiled": {k: v for k, v in prof.items() if k != "table"},
        "unprofiled": unprof,
    }
    if prof.get("ok"):
        summary["breakdown"] = build_breakdown(md, prof, unprof, a.noise)
        summary["op_table"] = prof["table"]
    else:
        summary["breakdown"] = None
        print(f"!! profiled run did not gate: {prof.get('blocker')}")

    out = Path(a.out) if a.out else (artifacts_dir() / "measurements" / "k1_spacemit"
                                     / md.name / f"op_profile_{ftag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote -> {out}")

    b = summary.get("breakdown")
    if b:
        c = b["coverage"]
        ok = c["runtime_shares_reportable"]
        denom = "of RUNTIME" if ok else "of ATTRIBUTED"
        print(f"\n=== WHERE THE WALL GOES (measured ms, share {denom}, "
              f"profiler_coverage={c['profiler_coverage']}) ===")
        for row in b["by_category"]:
            share = row["share_of_runtime"] if ok else row["share_of_attributed"]
            print(f"  {row['category']:<24} {row['ms']:8.2f} ms  {100 * (share or 0):5.1f}% "
                  f"n_ops={row['n_ops']}")
        print("  (categories this profiler CANNOT attribute: "
              + ", ".join(sorted(b["categories_not_attributable"])) + " — see the artifact)")
        q = b["quantize_vs_contraction"]
        ratio = f"{q['ratio']:.2f}x" if q["ratio"] else "n/a"
        print(f"\nQUANTIZE CHAIN vs CONTRACTION: {q['quantize_chain_ms']:.2f} ms vs "
              f"{q['contraction_ms']:.2f} ms = {ratio}  "
              f"(unread untagged generics: {q['unclassified_generic_ms']:.2f} ms, which could "
              f"belong to either side)")
        print(f"\n=== TOP FAMILIES (measured ms, share {denom}) ===")
        for row in b["by_family"][:12]:
            share = row["share_of_runtime"] if ok else row["share_of_attributed"]
            src = "+".join(row["family_sources"])
            print(f"  {row['family']:<20} {row['ms']:8.2f} ms  {100 * (share or 0):5.1f}% "
                  f"n_ops={row['n_ops']:<5} vec={row['vectorized']} from={src}")
        pt = b["perturbation"]
        if pt:
            print(f"\nperturbation: profiled={pt['profiled_median_wall_ns']/1e6:.2f}ms "
                  f"unprofiled={pt['unprofiled_median_wall_ns']/1e6:.2f}ms "
                  f"delta={pt['delta_pct']}% ok={pt['perturbation_ok']}")
        cov = b.get("rvv_coverage")
        covs = (f"{cov['static_vector_coverage']:.3f}"
                if cov and cov.get("static_vector_coverage") else "n/a")
        print(f"attributed={b['total_attributed_ms']:.2f}ms/execution over "
              f"wall={b['wall_ms_per_execution']}ms/execution "
              f"({b['measured_wall_ms']}ms per timed iteration / "
              f"{b['executions_per_timed_iteration']} executions)  "
              f"profiler_cov={c['profiler_coverage']} "
              f"band={c['band']}  executions/launch={c['executions_per_launch']} "
              f"timed_iters={c['timed_iterations_per_launch']}")
        # FAIL CLOSED. A percentage of runtime is printed only when the coverage check passed; an
        # out-of-band profile prints the ranking and says, in the same breath, why the percentages
        # it is NOT printing would have been wrong.
        if ok:
            print(f"scalar={b['scalar_ms']:.1f}ms ({b['scalar_share_of_runtime']*100:.1f}% of "
                  f"runtime)  contraction(intended-vec)={b['contraction_intended_ms']:.1f}ms  "
                  f"forward_static_rvv_cov={covs}")
        else:
            print(f"!! REFUSING to report shares as a percentage of runtime: {c['refusal']}")
            print(f"   shares above are of ATTRIBUTED op time ({b['total_attributed_ms']:.2f} ms), "
                  f"NOT of the {b['measured_wall_ms']} ms wall — do not quote them as runtime %.")
            print(f"scalar={b['scalar_ms']:.1f}ms "
                  f"({b['scalar_share_of_attributed']*100:.1f}% of attributed)  "
                  f"contraction(intended-vec)={b['contraction_intended_ms']:.1f}ms  "
                  f"forward_static_rvv_cov={covs}")
        if c["executions_exceed_timed_iterations"]:
            # In band, but the two windows still differ: say so, because this is the mechanism that
            # produced the 144 % profile and it is invisible in the headline numbers.
            print(f"   note: the shim counted {c['executions_per_launch']} @forward executions "
                  f"per launch against {c['timed_iterations_per_launch']} TIMED iterations "
                  f"({c['executions_per_timed_iteration']} executions per iteration x "
                  f"{c['timed_iterations_per_launch']} timed + the untimed warmup). The per-op mean "
                  f"therefore averages over the warmup passes too, which are colder — that is what "
                  f"any residual above 1.0 in profiler_cov is.")
        if b["contraction_is_a_lower_bound"]:
            print(f"   contraction is a LOWER BOUND: {b['unclassified_generic_ms']:.2f} ms sits in "
                  f"untagged linalg.generic, whose family is UNKNOWN (some are contractions).")


if __name__ == "__main__":
    main()
