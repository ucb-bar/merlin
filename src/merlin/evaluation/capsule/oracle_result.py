"""Capsule evaluation: oracle result."""

from __future__ import annotations

from merlin.development.targetgen import capsule_golden as CG
from merlin.evaluation.capsule.cpu import _grade_cpu_step
from merlin.evaluation.capsule.diagnostics import _absent_outputs as _absent_outputs
from merlin.evaluation.capsule.diagnostics import _oracle_kind as _oracle_kind
from merlin.evaluation.capsule.diagnostics import _record_console as _record_console
from merlin.evaluation.capsule.diagnostics import _unwritten_output_detail as _unwritten_output_detail
from merlin.evaluation.capsule.identity import _graded_program
from merlin.evaluation.capsule.lanes import _sim_telemetry
from merlin.evaluation.capsule.numerics import _absent_output_detail
from merlin.evaluation.capsule.records import OracleMeasurement
from merlin.evaluation.capsule.tier import TierResult as TierResult


def assess_oracle(
    capsule, package_dir, paths, cfg, tier, mand, res, _tm, _conc, perf_extractor, policy, evidence, inputs
):
    if res.get("executability") is not None:
        # ADVISORY executability smoke: a bounded-cycle RTL-legality check (does a cyclotron(L2)-
        # passing ELF at least RUN on the real Verilator RTL — boots, no trap, MX PE accepts a
        # command), the RTL-grounding backstop for the non-RTL-certified perf oracle. It does NOT
        # numeric-grade and NEVER blocks: recorded as a NON-mandatory tier + an ``executability``
        # field, and — because it is never in ``required`` — it is not_run_is_not_pass-exempt by
        # construction. A slow/flaky RTL smoke must not fail a capsule whose L2 grade passed, so
        # its verdict (legal/illegal) is informational only; ``mand`` is ignored here on purpose.
        ex = res["executability"]
        evidence.executability[tier] = ex
        _sim_name = _oracle_kind(res.get("oracle")) or cfg.tier_sim.get(tier) or tier
        _clog, _cbytes = _record_console(paths, _sim_name, res.get("console"))
        _tel = _sim_telemetry(res.get("oracle"), _sim_name)
        evidence.tiers[tier] = TierResult(
            tier,
            "pass" if ex.get("legal") else "fail",
            mandatory=False,
            reason=ex.get("reason"),
            cycles=ex.get("cycles"),
            derived_from_rtl=True,
            evidence=_clog or f"{_sim_name}_console.log",
            timing=_tm,
            not_applicable=True,
            concurrency=_conc,
            console_log=_clog,
            console_bytes=_cbytes,
            **_tel,
        )
        return None
    if res.get("completion_only"):
        # An RTL cert that ran the emitted kernel to completion but cannot surface its outputs
        # for an independent numeric check (e.g. the Muon Verilator harness $finish-races the
        # UART flush). It certifies RTL COMPLETION + cycle-accurate cycles; CORRECTNESS is the
        # mandatory functional tier's job. It is NEVER allowed to stand in for a mandatory tier
        # (a required tier must verify output) — there it degrades to honest-unavailable.
        _sim_name = cfg.tier_sim.get(tier) or tier
        if mand:
            evidence.tiers[tier] = TierResult(
                tier,
                "unavailable",
                mand,
                reason="RTL cert ran to completion but cannot surface outputs for a mandatory "
                "correctness check (use the functional tier as the required gate)",
                cycles=res.get("cycles"),
                derived_from_rtl=tier in cfg.rtl_tiers,
                timing=_tm,
                toolchain=_graded_program(res, package_dir),
                concurrency=_conc,
            )
            return None
        _cg = _cp = None
        _cu: dict | None = None
        if perf_extractor is not None:
            _cperf = perf_extractor(evidence.command_buffer, res) or {}
            _cg, _cp = _cperf.get("gflops"), _cperf.get("pct_fp_peak")
            _cu = {k: v for k, v in _cperf.items() if k not in ("gflops", "pct_fp_peak", "flops")} or None
        _sim_name = _oracle_kind(res.get("oracle")) or _sim_name
        _clog, _cbytes = _record_console(paths, _sim_name, res.get("console"))
        _tel = _sim_telemetry(res.get("oracle"), _sim_name)
        evidence.tiers[tier] = TierResult(
            tier,
            "pass",
            mand,
            reason="RTL completion + cycle-accurate perf cert (correctness gated by the required functional tier)",
            cycles=res.get("cycles"),
            derived_from_rtl=tier in cfg.rtl_tiers,
            toolchain=_graded_program(res, package_dir),
            cycle_accurate=tier in cfg.rtl_tiers,
            evidence=_clog or f"{_sim_name}_console.log",
            timing=_tm,
            gflops=_cg,
            pct_fp_peak=_cp,
            utilization=_cu,
            timing_observations=res.get("timing_observations"),
            counters=res.get("counters"),
            timing_capability=res.get("timing_capability"),
            concurrency=_conc,
            console_log=_clog,
            console_bytes=_cbytes,
            **_tel,
        )
        return None
    onboard_numeric = res.get("numeric_verdict")
    onrep = None
    if onboard_numeric is not None:
        # Some RTL SoCs cannot route accelerator console bytes to the
        # host. Their trusted carrier grades a declared coherent result
        # page and exports PASS/FAIL as a retained PC symbol instead.
        # Do not fabricate `outputs` from the expected values: the
        # carrier's explicit verdict is the evidence.
        okt = onboard_numeric.get("status") == "pass"
        if inputs.direct_oracle_golden and (mand or evidence.numeric.get("status") == "skipped"):
            _onboard_policy = onboard_numeric.get("policy", policy)
            if isinstance(_onboard_policy, dict):
                _onboard_policy = _onboard_policy.get("compare", "exact_int")
            evidence.numeric = {
                "status": "pass" if okt else "fail",
                "policy": _onboard_policy,
                "golden_source": inputs.golden_source,
                "elements_checked": onboard_numeric.get("elements_checked"),
                "witness": onboard_numeric.get("witness"),
                "max_abs_diff": None,
                "max_rel_error": None,
                "mismatch_count": 0 if okt else None,
                "first_mismatch": None,
                "per_output": {},
                "missing_outputs": [],
            }
    elif inputs.direct_oracle_golden:
        # Direct full-program grade: oracle output vs the capsule's semantic golden.  For float
        # this is the independent golden.yaml; for an explicit whole-program integer kernel it
        # is the independently recomputed operation golden.  In both cases the measured answer
        # comes only from the submitted ELF -- never from harness-side model arithmetic.
        onrep = _grade_cpu_step(
            CG.compare_declared_outputs, inputs.gold, res["outputs"], policy, **inputs.output_contract
        )
        okt = onrep["status"] == "pass"
        # The numeric verdict must ride the MANDATORY/gold tier, not an additive one: otherwise an
        # additive tier later in the ladder (e.g. an approximate or unbuilt model) would overwrite
        # the authoritative numeric. Record from a mandatory tier, or from the first tier to run if
        # none has set it yet.
        if mand or evidence.numeric.get("status") == "skipped":
            evidence.numeric = {
                "status": onrep["status"],
                "policy": onrep["policy"],
                "golden_source": inputs.golden_source,
                "max_abs_diff": onrep["max_abs_error"],
                "max_rel_error": onrep["max_rel_error"],
                "mismatch_count": onrep["mismatch_count"],
                "first_mismatch": onrep["first_mismatch"],
                # carry the per-output breakdown + the DROPPED-store list through to
                # capsule_result.json so the agent (and the self-check, which reads this
                # dict) sees WHICH output failed and WHY — a missing store otherwise reads
                # as mismatch_count>0 with max_abs_error==0, a self-contradictory signal.
                "per_output": onrep.get("per_output", {}),
                "missing_outputs": _absent_outputs(onrep),
            }
            CG.write_numeric_report(paths.generated / "numeric_report.yaml", onrep)
    else:
        okt = all(
            _grade_cpu_step(CG.compare_declared_outputs, expected, res["outputs"], policy, **inputs.output_contract)[
                "status"
            ]
            == "pass"
            for expected in (inputs.gold, inputs.reference, inputs.simulation)
        )
    # tier_sim is a display LABEL; a tier supplied by an adapter (e.g. arc L3) may have no static
    # label, so fall back to the adapter's oracle-provenance string, else the tier name.
    sim_name = cfg.tier_sim.get(tier) or (res.get("oracle") if isinstance(res.get("oracle"), str) else None) or tier
    # SIMT perf headline (gflops / % of peak) when a perf extractor is supplied; None (systolic)
    # leaves the fields off the TierResult so the output stays byte-identical.
    _gflops = _pct_peak = None
    _util: dict | None = None
    if perf_extractor is not None:
        _perf = perf_extractor(evidence.command_buffer, res) or {}
        _gflops = _perf.get("gflops")
        _pct_peak = _perf.get("pct_fp_peak")
        # Everything the extractor returned beyond the two headline numbers is utilization
        # detail. Taken as the remainder rather than a fixed list so a target can report the
        # counters ITS oracle actually has, without this shared runner naming any of them.
        _util = {k: v for k, v in _perf.items() if k not in ("gflops", "pct_fp_peak", "flops")} or None
    # Engineer framing (no "golden"): the emitted artifact, run on the RTL, does not compute the
    # declared operation. There is no answer key handed to the agent — the reference is the op's
    # own definition, which the agent can reproduce from the declared inputs. The appended
    # _encoding_divergence_hint adds that the cheap tiers agreed, so the defect is in the encoding.
    # A DROPPED declared output (kernel never wrote it) gets a precise, distinct detail so the
    # agent isn't left with the baffling "N mismatches, 0 error" of a store that never fired.
    # A store that never landed is named precisely, in BOTH of its shapes: the output missing
    # from the readback entirely (_absent_output_detail) and the output read back as a single
    # untouched constant (_unwritten_output_detail). Without the second, a writeback failure is
    # reported as a numeric mismatch whose count cannot move -- measured at six wasted rounds.
    _absent_detail = (
        (
            _absent_output_detail(onrep, sim_name, inputs.gold, res["outputs"])
            or _unwritten_output_detail(onrep, sim_name)
        )
        if inputs.direct_oracle_golden and onrep is not None
        else None
    )
    _mismatch_reason = _absent_detail or (
        f"on {sim_name}, your emitted artifact does not compute the declared operation within tolerance"
        if policy.get("compare") == "tolerance_float"
        else f"on {sim_name}, your emitted artifact does not compute the declared operation"
    )
    # ``oracle`` may be a rich dict ({kind, derived_from_rtl, fidelity}) OR a plain provenance
    # string; default to the tier's RTL classification only when it doesn't declare
    # derived_from_rtl. THE ORACLE'S OWN WORD OUTRANKS THE TIER NAME: an RTL-DERIVED model that
    # happens to land on the tier named L3 is not RTL certification, and classifying by name
    # credited it as such on every target whose L3 is not Verilator.
    _oracle_meta = res.get("oracle")
    _derived_from_rtl = (
        _oracle_meta.get("derived_from_rtl", tier in cfg.rtl_tiers)
        if isinstance(_oracle_meta, dict)
        else (tier in cfg.rtl_tiers)
    )
    _fidelity = _oracle_meta.get("fidelity") if isinstance(_oracle_meta, dict) else None
    # NAME THE EVIDENCE AFTER THE ORACLE THAT PRODUCED IT. ``sim_name`` comes from the contract's
    # static ``tier_sim`` map, which cannot know that a faster RTL-derived engine replaced the
    # declared one at runtime (the muon backend swaps GSIM in for Verilator whenever its emu is
    # configured). The console was then written to ``verilator_console.log`` by an engine that is
    # not Verilator -- a certification filed under the wrong tool's name, which is the same defect
    # as reporting a score without the tier that produced it. Same rule as ``derived_from_rtl``
    # above: the oracle's own word outranks the declared name.
    _ev_name = _oracle_kind(_oracle_meta) or sim_name
    _clog, _cbytes = _record_console(paths, _ev_name, res.get("console"))
    _tel = _sim_telemetry(_oracle_meta, _ev_name)
    evidence.tiers[tier] = TierResult(
        tier,
        "pass" if okt else "fail",
        mand,
        reason=None if okt else _mismatch_reason,
        cycles=res.get("cycles"),
        derived_from_rtl=_derived_from_rtl,
        toolchain=_graded_program(res, package_dir),
        cycle_accurate=(tier in cfg.rtl_tiers and okt),
        evidence=_clog or f"{_ev_name}_console.log",
        timing=_tm,
        gflops=_gflops,
        pct_fp_peak=_pct_peak,
        utilization=_util,
        timing_observations=res.get("timing_observations"),
        counters=res.get("counters"),
        timing_capability=res.get("timing_capability"),
        fidelity=_fidelity,
        concurrency=_conc,
        console_log=_clog,
        console_bytes=_cbytes,
        **_tel,
    )
    return OracleMeasurement(okt, _mismatch_reason, sim_name, _oracle_meta)
