"""Grade a submitted mlir_oot_target_backend package through the capsule_bench_v0 fidelity suite.

This turns capsule_bench from a self-test into a reusable GRADER: given any contract-satisfying
package directory, run the public/dev capsule suite (and, post-freeze, the hidden suite) through the
shared command-buffer/reference/oracle ladder, run the integrity scan, decode + check the RoCC trace,
and emit a single machine-readable ``score_capsule.json``.

Run this OUTSIDE the agent sandbox (it needs spike/verilator + the hidden capsules). The package is
still only invoked via its 4 CLI entrypoints (subprocess) — never imported.

Usage:
    python -m merlin.targetgen.capsule_grade --package <pkg_dir> --target <target> \
        --runs-root <out> [--hidden] [--labels public,dev] [--score <out.json>] [--no-oracle]

Do NOT pass ``--capsules merlin/contract/capsules``: that is the common parent of every target's corpus
and grades the package against all of them. Omit the flag and the target's own graded roots are used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import capsule_runner as CR
from .capsule_common import tier_field as _tier_field
from .capsule_common import tier_status as _tier_status
from . import coverage_report as CV
from .corpora import source_experiment_env
from .oot_runner import CertFailure, build_package, integrity_scan, load_package


def cycles_by_tier(tiers: dict | None, *, ladder: list[str] | tuple[str, ...] = ()) -> dict[str, int]:
    """Every tier record that REPORTS a cycle count, keyed by the tier that reported it.

    Which tier holds a capsule's cycle count is a property of the run, not a constant. The ladder runs
    a target's cheapest oracle first and that ordering differs per target, so a capsule can carry a
    count under one tier and nothing under another. Harvesting a single fixed tier therefore drops
    counts silently and asymmetrically: measured on one graded run, 11 of 12 FAILING capsules carried
    an elaborated-RTL record with no entry at the tier the diagnostic read, so the cycle count of
    nearly every failure — exactly the capsules a diagnostic exists to explain — never reached the
    summary, while the passes (which cleared every tier) were all present. That reads as "failures
    have no cycles", which is the opposite of true.

    Keyed by tier rather than flattened, so two tiers' counts for one capsule sit side by side and can
    be compared. ``ladder`` orders the result (tiers outside it are appended in encounter order); a
    tier record in the bare-string form (``"pass"``) reports no fields and contributes nothing.
    """
    found: dict[str, int] = {}
    for name, record in (tiers or {}).items():
        cycles = _tier_field(record, "cycles")
        if cycles is not None:
            found[name] = cycles
    ordered = {t: found[t] for t in ladder if t in found}
    ordered.update({t: c for t, c in found.items() if t not in ordered})
    return ordered


def default_grade_workers(n_capsules: int | None = None) -> int:
    """How many per-capsule oracle instances (verilator/VCS/cyclotron) to fan out in parallel — the
    per-capsule sim runs are independent, so grading N capsules serially wastes wall-clock. Derived from
    the host (leave 2 cores headroom, cap at 16), never more than the number of capsules; overridable via
    ``MERLIN_GRADE_WORKERS``. Applies to every arm's grade (the shared suite grader)."""
    env = os.environ.get("MERLIN_GRADE_WORKERS")
    if env and env.isdigit() and int(env) > 0:
        w = int(env)
    else:
        w = max(1, min((os.cpu_count() or 4) - 2, 16))
    return max(1, min(w, n_capsules)) if n_capsules else w


def grade(package_dir: str | Path, *, capsules_root: str | Path, runs_root: str | Path,
          labels: set[str] | None = None, contract: str | Path | None = None,
          oracle_adapters: dict | None = None, timeout: int = 900,
          max_workers: int = 0, target: str, no_oracle: bool = False) -> dict:
    """Run the capsule suite over a submitted package; return a score dict (also schema-checkable).

    ``max_workers`` fans the per-capsule oracle runs out in parallel (verilator/VCS/cyclotron instances);
    ``<= 0`` (the default) derives a host-scaled count via :func:`default_grade_workers` so grading a
    multi-capsule suite never serializes the RTL sims. Any positive value is honored as an explicit cap.

    ``no_oracle`` marks an EXPLICIT structure-only smoke (typically paired with ``oracle_adapters={}``):
    a capsule that clears the structural tiers but whose mandatory numeric tier was deliberately not run
    is recorded ``not_gradeable_no_oracle`` (a withheld numeric verdict), NOT the fixable
    ``oracle_unavailable`` plane — so the report is honest and never claims a numeric pass. Graded runs
    (``no_oracle=False``) keep the ``not_run_is_not_pass`` behavior byte-for-byte."""
    labels = labels or {"public", "dev"}
    # Source the target's own tooling paths before resolving any adapter, exactly as the harness does
    # for an arm. Without this a grade run outside the harness can lose a target's certifying-tier sim
    # and report the whole suite `incomplete` -- fail-closed, but describing the environment rather than
    # the submission. Process env wins, so an exported var is untouched.
    _sourced = source_experiment_env(target)
    if _sourced:
        print(f"  sourced {len(_sourced)} tooling path(s) from the target's experiment.env: "
              f"{', '.join(sorted(_sourced))}", flush=True)
    # absolute before any threading, for the same reason run_suite does it: this function also reads the
    # per-capsule traces back out of this root, and a root that moved under a sibling thread's chdir
    # silently yields an empty coverage dict rather than an error.
    runs_root = str(Path(runs_root).resolve())
    pkg_dir = Path(package_dir).resolve()
    if contract is not None and Path(contract).exists():
        contract = str(Path(contract).resolve())
    capsules_root = ([str(Path(r).resolve()) for r in capsules_root]
                     if isinstance(capsules_root, (list, tuple))
                     else str(Path(capsules_root).resolve()))

    score: dict = {
        "task": f"{target}-mlir-oot-capsule", "package": str(pkg_dir),
        "integrity_exempt": None, "integrity_status": None,
        "labels_graded": sorted(labels),
        "functional_pass": 0, "n_capsules": 0, "n_passed": 0,
        "public_passed": None, "hidden_passed": None, "headline": None,
        "per_capsule": [], "tier_reached": {}, "first_failure_planes": {},
        "numeric_all_exact": None, "trace_all_pass": None,
        "cycles_diagnostic": {}, "highest_tier": None,
        "timing_diagnostic": {}, "timing_rollup": {},
    }

    # K0/K1: load + integrity scan + build (fail-closed, recorded honestly)
    try:
        pkg = load_package(pkg_dir, contract=contract)
        score["integrity_exempt"] = pkg.integrity_exempt
        integrity_scan(pkg)
        score["integrity_status"] = "clean" if not pkg.integrity_exempt else "exempt"
        build_package(pkg)
    except CertFailure as cf:
        score["integrity_status"] = f"FAIL[{cf.plane}]: {cf.detail[:200]}"
        score["functional_pass"] = 0
        score["failure"] = {"plane": cf.plane, "category": str(cf.category), "detail": cf.detail}
        return score

    caps = CR.discover_capsules(capsules_root, labels=labels, contract=contract)
    workers = max_workers if max_workers and max_workers > 0 else default_grade_workers(len(caps))
    import time as _time
    _suite_t0 = _time.perf_counter()
    results = CR.run_suite(caps, pkg_dir, runs_root=runs_root, contract=contract,
                           oracle_adapters=oracle_adapters, timeout=timeout,
                           max_workers=workers, target=target, no_oracle=no_oracle)
    _suite_wall = _time.perf_counter() - _suite_t0

    # collect decoded traces for coverage — read from the TARGET's own suite dir (run_capsule writes
    # under cfg.suite, e.g. atlas-capsule-bench), not the gemmini SUITE literal (which left the atlas
    # coverage dict silently empty; same root cause as the self-check n_capsules:0 blind loop).
    traces: dict[str, dict] = {}
    rr = Path(runs_root) / "runs" / CR.suite_for(target)
    for cap in caps:
        tp = rr / cap["name"] / "generated" / "instruction_trace.json"
        if tp.exists():
            try:
                traces[cap["name"]] = json.loads(tp.read_text())
            except Exception:
                pass

    # A capsule withheld as outside the target's declared capability is in NEITHER the numerator nor the
    # denominator. Counting it as a failure is what made all_pass unreachable and disabled the loop's
    # early exit; counting it as a pass would be a phantom certification. It stays in `results` so the
    # skip is auditable, and its count is reported separately.
    # A capsule DEFERRED by its own gate (a whole-model capstone waiting on the op suite) is likewise in
    # neither bucket. It never ran a tier, so scoring it as a failure is not a measurement of the
    # submission -- and because the loop's ONLY early exit is a genuine all_pass, a permanently-deferred
    # capsule makes all_pass unreachable and forces every run to buy its entire round budget. Measured:
    # a 28-capsule grade of {pass 14, fail 12, not_graded 1, gated 2} could never reach all_pass no
    # matter what the agent did. The gate fraction it is waiting on is reported separately, as OP
    # COVERAGE -- it is not a verdict on the model.
    ungraded = [r for r in results if r.get("status") == "not_graded"]
    deferred = [r for r in results if r.get("status") == "gated"]
    screened = [r for r in results if r.get("status") == "screened_only"]
    graded = [r for r in results
              if r.get("status") not in ("not_graded", "gated", "screened_only")]
    n_pass = sum(1 for r in graded if r["status"] == "pass")
    score["n_capsules"] = len(graded)
    score["n_passed"] = n_pass
    score["n_not_graded_ineligible"] = len(ungraded)
    score["n_gated_deferred"] = len(deferred)
    if ungraded:
        score["not_graded_ineligible"] = sorted(r.get("capsule") for r in ungraded)
    if deferred:
        score["gated_deferred"] = sorted(r.get("capsule") for r in deferred)
    # SCREENED BUT NOT CERTIFIED, listed by name. A headline of "14/14" over a suite where nine more
    # capsules were screened and never certified is only honest if the nine are visible next to it.
    score["n_screened_only"] = len(screened)
    if screened:
        score["screened_only"] = sorted(r.get("capsule") for r in screened)
        score["screened_only_note"] = (
            "these capsules PASSED the cheap screen tier and were deliberately not measured against the "
            "certifying tier: they fall outside the derived covering set and the certify budget was "
            "exhausted. They are in neither the numerator nor the denominator. The covering set "
            "guarantees every axis they exercise was certified by some capsule -- it does NOT certify "
            "these capsules.")
    # DECLINED, listed by name and shape. Unlike `screened_only` these ARE in the denominator: the
    # backend was asked and said no, so the capsule is uncertified and counts against the suite. What
    # changes is that it is no longer reported as wrong arithmetic. A run whose deficit is twelve
    # declines is a backend that does not cover the shape space; a run whose deficit is twelve numeric
    # mismatches is a backend whose math is wrong. Those need opposite repairs, and the score used to
    # spell them the same way.
    _declined = [r for r in graded if r.get("status") == "declined"]
    score["n_declined"] = len(_declined)
    if _declined:
        score["declined"] = sorted(
            ({"capsule": r.get("capsule"), **(r.get("declined") or {})} for r in _declined),
            key=lambda d: str(d.get("capsule")))
        score["declined_note"] = (
            "the backend STATED it does not lower these capsules rather than emitting a program that "
            "writes nothing. They are in the denominator and are NOT passes. Read them as coverage "
            "gaps (a shape/op the backend never implemented), not as numeric defects.")
    # BOTH NUMBERS, ALWAYS. When the runtime drove a loop nest over a backend's declared single tile,
    # the resulting pass is evidence about the runtime AND the backend; the backend's own coverage is the
    # smaller number. Reporting only the larger one is precisely how "our runtime covered for them" gets
    # cited as "their compiler generalizes", so the two travel together or the guardrail is not real.
    _runtime_tiled = [r for r in graded
                      if (((r.get("contract_obligations") or {}).get("capacity_fit") or {}).get("tiled_by")
                          == "declared_primitive_tile")]
    if _runtime_tiled:
        _names = sorted(str(r.get("capsule")) for r in _runtime_tiled)
        _unblocked = n_pass - sum(1 for r in _runtime_tiled if r.get("status") == "pass")
        score["backend_coverage"] = {
            "with_runtime_loop": f"{n_pass}/{len(graded)}",
            "unblocked": f"{_unblocked}/{len(graded)}",
            "runtime_tiled_capsules": _names,
            "note": ("`unblocked` is what the BACKEND lowered on its own. `with_runtime_loop` adds the "
                     "capsules that only passed because the runtime drove the M/N/K loop over the single "
                     "tile the backend declared. Quote `unblocked` for any claim about the backend "
                     "generalizing over shape; `with_runtime_loop` is a statement about the pair."),
        }
    score["functional_pass"] = int(n_pass == len(graded) and len(graded) > 0)
    # Structure-only smoke bookkeeping (honest, never a numeric pass): a capsule is structurally clean
    # when it did not FAIL a structural tier — status `pass` OR `not_gradeable_no_oracle` (numeric verdict
    # withheld under --no-oracle). `gradeable` says whether this run had a numeric oracle at all.
    n_not_gradeable = sum(1 for r in graded if r["status"] == "not_gradeable_no_oracle")
    # Fail-closed on an empty suite: if NO capsule matched the requested labels at this root, nothing was
    # graded. `all([])` is vacuously True, so numeric_all_exact / trace_all_pass would read as a phantom
    # pass and `gradeable` as True — the exact vacuous-pass trap that made a mis-rooted hidden phase
    # (n_capsules:0) look green. Report the boolean flags as null and gradeable False, never a pass.
    _empty = len(graded) == 0
    # A capsule is INCOMPLETE when a MANDATORY tier came back `unavailable`: the oracle never ran, so
    # this score says nothing about the submission for it. That is a third thing, distinct from both
    # neighbours it keeps being confused with — a FAIL means the submission was wrong, and
    # `not_gradeable_no_oracle` means --no-oracle deliberately withheld the verdict. Here the run
    # BELIEVED it was grading and the tool was simply missing.
    #
    # Measured: re-grading a frozen submission whose own run scored 33/36 produced `n_passed: 6` with 29
    # capsules incomplete ("no derived ISA encoding fact for target 'radiance'") — the six that "passed"
    # were the MX fixtures, which need no oracle. Nothing in the rollup said so: n_not_gradeable_no_oracle
    # stayed 0 and a reader would quote 6/36 as a capability collapse. It is an ENVIRONMENT gap, and the
    # scorecard has to say which of the two it is or the number is worse than no number.
    _incomplete = [r for r in graded if r.get("status") == "incomplete"]
    score["n_incomplete"] = len(_incomplete)
    if _incomplete:
        _why = sorted({(r.get("failure") or {}).get("tier_reason")
                       or (r.get("failure") or {}).get("detail") or "unknown"
                       for r in _incomplete})
        score["measurement_incomplete"] = {
            "n": len(_incomplete), "of": len(graded),
            "capsules": sorted(r["capsule"] for r in _incomplete)[:12],
            "reasons": _why[:4],
            "detail": (f"{len(_incomplete)} of {len(graded)} capsules had a MANDATORY tier come back "
                       f"unavailable, so the oracle never ran on them. n_passed is NOT a capability "
                       f"measurement for this run — fix the environment and re-grade before quoting it."),
        }
    # `gradeable` means this run had a working numeric oracle. It did not, for those capsules, so it is
    # False even though an oracle was requested — the same fail-closed posture the empty suite gets.
    score["gradeable"] = (not no_oracle) and not _empty and not _incomplete
    score["n_not_gradeable_no_oracle"] = n_not_gradeable
    score["n_structural_pass"] = n_pass + n_not_gradeable
    score["structural_pass"] = bool(not _empty and (n_pass + n_not_gradeable) == len(graded))
    score["numeric_all_exact"] = None if _empty else all(
        r.get("numeric", {}).get("status") == "pass" for r in graded)
    score["trace_all_pass"] = None if _empty else all(
        r.get("trace_check", {}).get("status") == "pass" for r in graded)
    if _empty:
        score["note"] = ("no capsules matched the requested labels at this root — nothing graded; "
                         "flags are null (not a pass). Check the capsules root / labels.")

    pub = [r for r in graded if r.get("label") in ("public", "dev")]
    hid = [r for r in graded if r.get("label") == "hidden"]
    if pub:
        score["public_passed"] = f"{sum(1 for r in pub if r['status']=='pass')}/{len(pub)}"
    if hid:
        score["hidden_passed"] = f"{sum(1 for r in hid if r['status']=='pass')}/{len(hid)}"

    tiers = ["L0", "L1", "L2", "L3", "L4", "L5"]
    for t in tiers:
        # over GRADED only: a withheld capsule has no tiers, so counting it in the denominator would
        # make tier_reached never reach the total and silently force highest_tier to None.
        score["tier_reached"][t] = sum(
            1 for r in graded if _tier_status((r.get("tiers") or {}).get(t)) == "pass")
    for t in reversed(tiers):
        if score["tier_reached"][t] == len(graded) and len(graded) > 0:
            score["highest_tier"] = t
            break

    # WHAT KIND OF EVIDENCE EACH PASS RESTS ON, stated next to the count rather than left to be
    # reconstructed from `tier_reached`. A capsule can report `pass` while a hardware tier beside it
    # reports `fail` -- when that tier was advisory at the time -- and the flattering half is the half
    # that gets quoted. Measured: one gemmini submission shipped as "20/20" whose Verilator tier passed
    # exactly 1 of 20, next to three sibling submissions whose 20/20 was RTL-clean on all 20. Both are
    # "20/20"; they are not the same result, and nothing in the headline said so.
    #
    # RTL-ness is DERIVED, never a tier-name literal: each tier record carries `derived_from_rtl`, set
    # from the target's own `cfg.rtl_tiers`, so a target whose RTL tier is L4 (or L2) is described
    # correctly without this code knowing which one it is.
    # A model capsule records its tier as a bare string, which carries no `derived_from_rtl` flag, so
    # keying only on the flag counted a whole model that PASSED the RTL tier as cheap-tier-only. Fall
    # back to the target's own declared rtl_tiers, matched by tier NAME -- still derived (the manifest
    # says which tiers are RTL for this target), never a literal.
    from .capsule_runner import _rtl_tiers_of
    _rtl_names = _rtl_tiers_of(target)

    # AN EXPLICIT DENIAL MUST WIN OVER THE NAME. `derived_from_rtl` has three states, and collapsing the
    # last two is what let a model count as RTL: True (the oracle says it ran RTL), False (the oracle says
    # it did NOT -- e.g. an arc cosim, an RTL-DERIVED model, landing on the tier named L3), and None (the
    # record is the bare-string form a model capsule writes, which states nothing). Only None may fall
    # back to the tier name. Reading `flag or name in _rtl_names` credited an oracle that had explicitly
    # denied being RTL, purely because its tier shared a name with another target's Verilator tier.
    def _is_rtl_pass(name, rec) -> bool:
        if _tier_status(rec) != "pass":
            return False
        flag = _tier_field(rec, "derived_from_rtl")
        return bool(name in _rtl_names) if flag is None else bool(flag)

    def _tier_is_rtl(name, rec) -> bool:
        """Same three-state rule as :func:`_is_rtl_pass`, without the pass requirement."""
        flag = _tier_field(rec, "derived_from_rtl")
        return bool(name in _rtl_names) if flag is None else bool(flag)

    _passed = [r for r in graded if r.get("status") == "pass"]
    _rtl_backed = [r for r in _passed
                   if any(_is_rtl_pass(n, t) for n, t in (r.get("tiers") or {}).items())]
    # A pass that cleared a tier the ORACLE ITSELF called a model (fidelity != elaborated_rtl) and no RTL
    # tier beside it. Named separately because "not RTL-backed" reads as "did not run on hardware-grade
    # evidence at all", which is not what an RTL-derived cosim pass is.
    _model_certified = [r for r in _passed
                        if r not in _rtl_backed
                        and any(_tier_status(t) == "pass" and _tier_field(t, "fidelity")
                                for t in (r.get("tiers") or {}).values())]
    score["pass_evidence"] = {
        "n_passed": len(_passed),
        "rtl_backed": len(_rtl_backed),
        "model_certified": len(_model_certified),
        "cheap_tier_only": len(_passed) - len(_rtl_backed),
        "rtl_tiers_seen": sorted({n for r in graded for n, t in (r.get("tiers") or {}).items()
                                  if _tier_is_rtl(n, t)}),
        "fidelity_seen": sorted({f for r in graded for t in (r.get("tiers") or {}).values()
                                 if (f := _tier_field(t, "fidelity"))}),
        "note": ("`rtl_backed` counts passes that cleared a tier the ORACLE reported as elaborated RTL "
                 "(`derived_from_rtl`), falling back to the target's declared rtl_tiers only when the "
                 "oracle states nothing. A tier NAME is not evidence: one target's L3 is Verilator and "
                 "another's is an RTL-derived model, and counting by name credited the model. When "
                 "`rtl_backed` is below `n_passed`, the headline score is NOT an RTL result."),
    }

    # THE CITABLE FORM OF THE SCORE. `public_passed` stays a bare "20/20" because consumers parse it as
    # a fraction (agg_agentic_results._frac does int(total)), and a bare fraction is exactly what gets
    # quoted -- one gemmini submission travelled as "20/20" while its Verilator tier passed 1 of 20,
    # indistinguishable in the headline from three siblings that were RTL-clean on all 20. So the
    # qualification is built ONCE, here, next to the evidence that justifies it, and the renderers print
    # THIS rather than reassembling a bare fraction each time. Quote `headline`; parse `public_passed`.
    score["headline"] = _headline(score)

    _agg = {"build_s": 0.0, "sim_active_s": 0.0, "oracle_wait_s": 0.0}
    for r in results:
        _cyc_by_tier = cycles_by_tier(r.get("tiers"), ladder=tiers)
        if _cyc_by_tier:
            score["cycles_diagnostic"][r["capsule"]] = _cyc_by_tier
        # active-vs-waiting timing: sum across every tier that actually ran an oracle for this capsule
        cap_tm = {"build_s": 0.0, "sim_active_s": 0.0, "oracle_wait_s": 0.0, "by_tier": {}}
        for t in ("L2", "L3", "L4", "L5"):
            tm = _tier_field((r.get("tiers") or {}).get(t), "timing")
            if tm:
                cap_tm["by_tier"][t] = tm
                for k in ("build_s", "sim_active_s", "oracle_wait_s"):
                    v = tm.get(k) or 0.0
                    cap_tm[k] += v
                    _agg[k] += v
        if cap_tm["by_tier"]:
            score["timing_diagnostic"][r["capsule"]] = {k: round(cap_tm[k], 3)
                                                         for k in ("build_s", "sim_active_s",
                                                                   "oracle_wait_s")}
        if r.get("failure"):
            p = r["failure"]["plane"]
            score["first_failure_planes"][p] = score["first_failure_planes"].get(p, 0) + 1
        entry = {
            "capsule": r["capsule"], "label": r.get("label"), "status": r["status"],
            "numeric": r.get("numeric", {}).get("status"),
            "trace": r.get("trace_check", {}).get("status"),
            "tiers": {t: _tier_status((r.get("tiers") or {}).get(t)) for t in tiers
                      if t in (r.get("tiers") or {})},
        }
        # CARRY THE QUALIFIERS WITH THE VERDICT. The roll-up kept six fields, and every one of them
        # flatters: status, tier, numeric, trace. Everything that says what the pass RESTS ON --
        # which contract obligations someone else discharged on the backend's behalf, how many of the
        # model's own layers reached the accelerator versus fell back to the host, which declared tiers
        # never ran -- stayed behind in the per-capsule result, several directories down. The score file
        # is the artifact that gets cited, so a qualifier that does not travel with it does not exist.
        if r.get("status") == "gated":
            # WHY it was deferred. "gated" with no reason is uninterpretable in the artifact -- it reads
            # as "skipped" when what it means is "the op suite did not earn the right to run this".
            entry["gate_reason"] = (r.get("failure") or {}).get("detail")
        if r.get("kind") == "model":
            entry["kind"] = "model"
            me = r.get("mesh_execution") or {}
            if me:
                entry["mesh_execution"] = {k: me.get(k) for k in (
                    "matmul_layers_on_mesh", "matmul_layers_host_fallback", "status") if k in me}
            for k in ("contract_obligations", "tiers_unexercised"):
                if r.get(k):
                    entry[k] = r[k]
        score["per_capsule"].append(entry)

    # active-vs-waiting rollup: wall is the suite wall-clock (overlapped under parallelism); the sum of
    # active_sim across capsules can exceed wall (that ratio IS the parallel speedup). oracle_wait_s is
    # time blocked on a queue/FPGA slot (≈0 for local spike/verilator).
    _active = _agg["build_s"] + _agg["sim_active_s"]
    score["timing_rollup"] = {
        "suite_wall_s": round(_suite_wall, 3),
        "build_s": round(_agg["build_s"], 3),
        "sim_active_s": round(_agg["sim_active_s"], 3),
        "active_total_s": round(_active, 3),
        "oracle_wait_s": round(_agg["oracle_wait_s"], 3),
        "max_workers": workers,
        "parallel_speedup": round(_active / _suite_wall, 2) if _suite_wall > 0 else None,
    }

    # coverage aggregate (written alongside)
    cov = CV.aggregate(results, capsules=caps, traces=traces, target=target)
    score["coverage"] = {"by_tier_reached": cov["by_tier_reached"],
                         "instruction_class_coverage": cov["instruction_class_coverage"],
                         "mode_coverage": cov["mode_coverage"], "unavailable": cov["unavailable"],
                         "acceleratable_coverage": cov["acceleratable_coverage"]}

    # PROVENANCE. This score asserts a hardware result -- N of M capsules passed on named oracle tiers --
    # so it must record WHICH hardware revision produced it. A result attributed to the wrong device is
    # worse than no result, because it gets cited. Pins for these targets already existed; nothing emitted
    # the block, and the gate could not see the score to complain, so every bench verdict on every target
    # was unattributable. Never fatal: a grade that ran must still report its numbers, with the
    # provenance gap visible rather than the whole grade lost.
    try:
        from merlin.common import provenance as _P
        _pins = {}
        for _name in _P.load_pins():
            try:
                _pins[_name] = _P.verify(_name)
            except Exception:                                 # noqa: BLE001 - checkout absent is a gap
                continue
        score["provenance"] = _P.record(pins=_pins or None,
                                        extra={"target": target, "n_capsules": score.get("n_capsules"),
                                               "n_passed": score.get("n_passed")})
    except Exception as _e:                                   # noqa: BLE001
        score["provenance"] = {"unavailable": f"{type(_e).__name__}: {_e}"}
    return score


def _headline(score: dict) -> str:
    """The score as it may be QUOTED: never a bare fraction, always with what the passes rest on.

    Reports the RTL-backed share whenever it is below the pass count, and the highest tier every graded
    capsule reached (``None`` -> stated as such rather than omitted, because an omitted tier reads as a
    high one). A reader who copies this string cannot accidentally drop the qualification, which is the
    whole failure this exists to prevent.
    """
    ev = score.get("pass_evidence") or {}
    n_passed, n_caps = score.get("n_passed", 0), score.get("n_capsules", 0)
    head = score.get("public_passed") or f"{n_passed}/{n_caps}"
    tier = score.get("highest_tier")
    bits = [f"tier {tier}" if tier else "no tier cleared by every capsule"]
    rtl, cheap = ev.get("rtl_backed"), ev.get("cheap_tier_only")
    # SAY WHAT THE NON-RTL PASSES ACTUALLY RESTED ON. "passed on cheap tiers only" is right for a spike
    # pass and wrong for an RTL-DERIVED cosim pass, and the two now arrive distinguished.
    modelc = ev.get("model_certified") or 0
    if rtl is not None and cheap:
        rest = (f"the other {cheap} on a model oracle only ({modelc} RTL-derived)" if modelc
                else f"the other {cheap} passed on cheap tiers only")
        bits.append(f"RTL-backed {rtl}/{n_passed} \u2014 {rest}")
    elif rtl is not None and n_passed:
        bits.append(f"RTL-backed {rtl}/{n_passed}")
    if score.get("hidden_passed"):
        bits.append(f"hidden {score['hidden_passed']}")
    # CAPSULES THAT WERE SCREENED AND NEVER CERTIFIED BELONG IN THE QUOTABLE STRING. Under a certify
    # budget the denominator is what was certified, so a headline of "17/18" can sit over a suite where
    # seven more capsules passed the cheap screen and nobody ever paid the RTL tier for them. Leaving
    # that to the JSON is the same mistake as leaving the tier to the JSON: the string is what gets
    # copied. The covering set guarantees the AXES were certified; it certifies nothing about these
    # capsules, and the wording has to keep those two apart.
    n_screened = score.get("n_screened_only") or 0
    if n_screened:
        bits.append(f"{n_screened} more screened only, NOT certified (outside the covering set)")
    # A DEFICIT MADE OF DECLINES IS A DIFFERENT RESULT FROM ONE MADE OF WRONG ANSWERS, and the headline
    # is what gets quoted. "14/26" reads as a compiler that computes twelve things incorrectly; if the
    # twelve were declined it is a compiler that never implemented them, which is a different repair.
    n_declined = score.get("n_declined") or 0
    if n_declined:
        bits.append(f"{n_declined} DECLINED by the backend (never lowered, not wrong answers)")
    return f"{head} ({'; '.join(bits)})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grade a backend package through capsule_bench_v0")
    ap.add_argument("--package", required=True)
    ap.add_argument("--capsules", default=None,
                    help="capsules ROOT override (a single directory); default = the TARGET's own graded "
                         "roots. Prefer the default: a hand-passed root is how a grade ends up scoring "
                         "one target's package against another target's corpus.")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--target", required=True, help="target being graded (its config/oracle are derived)")
    ap.add_argument("--contract", default=None)
    ap.add_argument("--labels", default="public,dev")
    ap.add_argument("--hidden", action="store_true", help="grade ONLY hidden capsules (post-freeze)")
    ap.add_argument("--no-oracle", action="store_true", help="L0/L1/trace only (skip spike/verilator)")
    ap.add_argument("--score", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel oracle instances (verilator/VCS/cyclotron); 0 = host-scaled default")
    a = ap.parse_args(argv)
    # Resolve the roots ABSOLUTELY before anything runs. run_capsule chdirs into the build tree, so a
    # relative contract/capsules root stops resolving mid-grade and surfaces as a FileNotFoundError
    # from inside a worker thread -- which reads as a broken package rather than a bad path. The
    # defaults also came from string literals; the repo's rule is to ask paths for its roots.
    from merlin.common.paths import data_path
    a.contract = Path(a.contract).resolve() if a.contract else data_path("contract")
    # DEFAULT TO THE TARGET'S OWN GRADED ROOTS, not the corpus parent. The parent holds every target's
    # capsules at once, so it graded the gemmini package against 173 capsules from seven targets, called
    # 89 of them "outside this target's declared capability", and printed `1/84`. That number reads as a
    # catastrophic regression and means nothing; the target's real suite is 36. The usage line in this
    # module's own docstring recommended that parent, while discover_capsules two modules over documents
    # why it is wrong -- so following the documentation produced the bad grade.
    from .corpora import graded_capsule_roots
    a.capsules = ([Path(a.capsules).resolve()] if a.capsules
                  else graded_capsule_roots(a.target, hidden=a.hidden))

    labels = {"hidden"} if a.hidden else set(a.labels.split(","))
    adapters = {} if a.no_oracle else None
    score = grade(a.package, capsules_root=a.capsules, runs_root=a.runs_root, labels=labels,
                  contract=a.contract, oracle_adapters=adapters, timeout=a.timeout,
                  max_workers=a.workers, target=a.target, no_oracle=a.no_oracle)
    out = Path(a.score) if a.score else Path(a.runs_root) / "score_capsule.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(score, indent=2), encoding="utf-8")
    print(f"wrote {out}: functional_pass={score['functional_pass']} "
          f"score={score.get('headline')} integrity={score.get('integrity_status')}")
    return 0 if score["functional_pass"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
