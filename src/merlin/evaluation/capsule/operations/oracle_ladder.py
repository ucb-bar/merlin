"""Capsule evaluation: oracle call."""

from __future__ import annotations

import dataclasses

from merlin.development.targetgen.corpus.capsule_io import _cat
from merlin.development.targetgen.grading import golden as CG
from merlin.development.targetgen.oracles import certificates as _partial_console
from merlin.development.targetgen.oracles import ordering_gates as _CO_GATE
from merlin.development.targetgen.oracles import tier_policy as _tier_policy
from merlin.development.targetgen.oracles.tier_policy import CERT_NOT_MEASURED
from merlin.development.targetgen.packaging.loading import (
    INFRASTRUCTURE_PLANE,
    CertFailure,
    InfraCategory,
    InfraFailure,
    NotMeasuredFailure,
)
from merlin.development.targetgen.packaging.records import toolchain_shas
from merlin.evaluation.capsule.evidence.certificates import (
    OracleUnavailable,
    _CarriedPreparedTier,
    _clear_stale_executable,
    _full_ladder_enabled,
    _order_after_certificate,
    _prepared_tier_reuse,
    carried_tier_result,
    record_tier_certificate,
    stamp_cache_refusals,
    suppressed_tier_result,
)
from merlin.evaluation.capsule.evidence.lanes import _sim_telemetry, oracle_source_scope
from merlin.evaluation.capsule.evidence.source_custody import _graded_program, concurrency_stamp, submission_identity
from merlin.evaluation.capsule.evidence.tiers import TierResult as TierResult
from merlin.evaluation.capsule.operations import setup_timing as _host_timing
from merlin.evaluation.capsule.operations.grading import OracleAttempt, OracleMeasurement
from merlin.evaluation.capsule.operations.numerics import (
    _absent_output_detail,
    _did_not_halt_failure,
    _did_not_halt_reason,
    _encoding_divergence_hint,
    _grade_cpu_step,
    _is_readback_refusal,
    _oracle_timeout_reason,
    _out_of_range_reason,
    _readback_refusal_detail,
)
from merlin.evaluation.capsule.operations.numerics import _absent_outputs as _absent_outputs
from merlin.evaluation.capsule.operations.numerics import _clip as _clip
from merlin.evaluation.capsule.operations.numerics import _oracle_kind as _oracle_kind
from merlin.evaluation.capsule.operations.numerics import _range_check_bounds as _range_check_bounds
from merlin.evaluation.capsule.operations.numerics import _record_console as _record_console
from merlin.evaluation.capsule.operations.numerics import _unwritten_output_detail as _unwritten_output_detail

# ---- oracle_call -------------------------------------------------------------------------------


def invoke_oracle(
    capsule,
    package_dir,
    paths,
    cfg,
    timeout,
    eff_target,
    name,
    run_id,
    tier,
    mand,
    adapter,
    workers,
    adapter_managed_tiers,
    policy,
    evidence,
    inputs,
    _pin_shas,
    _simulation_context,
    _executed_here,
):
    from merlin.development.targetgen.oracles.program_oracle import OracleUnavailable as _POUnavailable
    from merlin.development.targetgen.oracles.program_oracle import ProgramDidNotHalt as _PODidNotHalt

    _ORACLE_UNAVAILABLE = (OracleUnavailable, _POUnavailable)
    import time as _time

    _adapter_t0 = _time.perf_counter()

    def _failed_adapter_timing() -> dict:
        """Wall is measurable even when an adapter raises before returning its phase split.

        Build/sim/wait remain unknown rather than fabricated zeroes.  Previously every timeout,
        trap, did-not-halt, and oracle-unavailable result discarded even its measured wall time,
        making the most expensive L-tier attempts disappear from experiment accounting.
        """
        return {
            "build_s": None,
            "sim_active_s": None,
            "oracle_wait_s": None,
            "adapter_wall_s": round(_time.perf_counter() - _adapter_t0, 3),
            "partial": True,
            "unavailable_reason": "adapter raised before returning the phase timing split",
        }

    try:
        # Keep the compiler-owned command buffer canonical. The expected
        # answer is private runner-to-oracle state, so attach it only to
        # the shallow per-call view passed to an adapter. Besides avoiding
        # later serialization leakage, this keeps trusted golden changes
        # out of submitted-kernel build identities.
        oracle_cb = dict(evidence.command_buffer)
        oracle_cb["_oracle_expected_outputs"] = inputs.gold
        oracle_cb["_oracle_numeric_policy"] = policy
        # A frozen post-search stage carries the exact cycle count measured for THIS capsule at
        # its sealed predecessor tier. Pass it only on the private runner-to-oracle view: it is
        # observation-budget provenance, not compiler input and not part of the submitted CB.
        _stage = capsule.get("evaluation_stage") or {}
        _l2_cycles = _stage.get("predecessor_l2_cycles") if isinstance(_stage, dict) else None
        if _l2_cycles is not None:
            if not isinstance(_l2_cycles, int) or isinstance(_l2_cycles, bool) or _l2_cycles <= 0:
                raise ValueError("evaluation_stage.predecessor_l2_cycles must be a positive integer")
            oracle_cb["_oracle_l2_cycles"] = _l2_cycles
            _cycle_policy = _stage.get("cycle_budget")
            if not isinstance(_cycle_policy, dict):
                raise ValueError("evaluation_stage with predecessor cycles requires cycle_budget policy")
            oracle_cb["_oracle_gsim_cycle_policy"] = dict(_cycle_policy)
        with _prepared_tier_reuse(
            name,
            tier,
            mand,
            target=eff_target,
            generated=paths.generated,
            shas=_pin_shas,
            from_rtl=tier in cfg.rtl_tiers,
            simulation_context=_simulation_context,
            first=not _executed_here and tier not in adapter_managed_tiers,
        ):
            with oracle_source_scope(capsule, oracle_cb, evidence.lowered_text, target=eff_target):
                with _host_timing.stage("oracle_adapter." + str(tier)):
                    res = adapter(oracle_cb, evidence.lowered_text, paths.generated, timeout)
    except _CarriedPreparedTier as carried:
        evidence.tiers[tier], _executed_here = carried.result, _executed_here or carried.built
        return OracleAttempt(carried_built=_executed_here)
    except _PODidNotHalt as e:
        # The oracle RAN and returned a verdict: the program never halted. That is the AGENT's
        # bug, not a missing oracle — record it as a tier FAIL (not "unavailable") so the
        # not_run_is_not_pass path never relabels it, and fail the capsule on the named
        # did-not-halt plane. MUST precede the _ORACLE_UNAVAILABLE clause below: this is a
        # subclass of it, and being caught there is exactly what hid the diagnosis.
        evidence.tiers[tier] = TierResult(
            tier,
            "fail",
            mand,
            reason=_did_not_halt_reason(str(e)),
            derived_from_rtl=tier in cfg.rtl_tiers,
            timing=_failed_adapter_timing(),
            concurrency=concurrency_stamp(workers),
        )
        if mand:
            raise _did_not_halt_failure(str(e)) from e
        return OracleAttempt(carried_built=_executed_here)
    except _ORACLE_UNAVAILABLE as e:
        evidence.tiers[tier] = TierResult(
            tier,
            "unavailable",
            mand,
            reason=str(e),
            derived_from_rtl=tier in cfg.rtl_tiers,
            timing=_failed_adapter_timing(),
            concurrency=concurrency_stamp(workers),
        )
        return OracleAttempt(carried_built=_executed_here)
    except Exception as e:  # adapter raised — classify: a non-terminating program is the AGENT's
        # FIFTH CASE, and the only one that runs the OTHER way. The four below all rescue a
        # verdict that WAS the agent's from an infra-shaped label. This one rescues the agent
        # from a verdict that was never measured: `readback_integrity` refuses a buffer whose
        # every Nth word came back exactly zero while the rest carried data -- a transport
        # defect, not an arithmetic result -- BEFORE any value is compared. The kernel may be
        # perfectly correct. Recorded as `fail` it reads as a numeric defect the agent must
        # chase, and MEASURED 2026-09-09 it cost radiance three capsules that had already
        # passed L2 with zero mismatches, all three landing as `tool_crash` / plane
        # `verilator` -- naming a simulator that had not crashed.
        # Detected by walking the exception chain for the refusal type, never by matching its
        # message: the adapters re-wrap it (`raise MuonError(str(exc)) from exc`, then
        # `raise ... from e` again), so the TYPE is what survives, not the spelling.
        if _is_readback_refusal(e):
            _detail = _readback_refusal_detail(e)
            evidence.tiers[tier] = TierResult(
                tier,
                "unavailable",
                mand,
                reason=(f"readback refused before comparison: {_clip(_detail, 300)}"),
                derived_from_rtl=tier in cfg.rtl_tiers,
                timing=_failed_adapter_timing(),
                concurrency=concurrency_stamp(workers),
            )
            if mand:
                raise InfraFailure(
                    INFRASTRUCTURE_PLANE,
                    InfraCategory.READBACK_TRANSPORT_REFUSED,
                    f"the device readback was refused on structural grounds before any value "
                    f"was compared, so NOTHING about this kernel was measured: {_detail} "
                    f"This is a transport/harness fault, NOT a defect in the graded "
                    f"submission -- the kernel may be correct.",
                ) from e
            return OracleAttempt(carried_built=_executed_here)
            # bug (it ran to the cycle cap), a TIMEOUT fail, NOT a tool_crash. Mislabeling it
            # 'tool_crash' read as an infra problem the agent couldn't fix — so it never emitted the
            # ISA's halt instruction and every round failed identically (the atlas flat-0/11 wall).
        _sim = cfg.tier_sim.get(tier, tier)
        _msg = str(e)
        _did_not_halt = "did not halt" in _msg
        # A program that RAN on the sim but FAULTED at runtime (the bare-metal fail/trap banner
        # fired: "*** FAILED ***") is the AGENT's kernel bug — an illegal instruction or an
        # out-of-range memory access (e.g. an MVIN/MVOUT DRAM address of 0, or a baked address that
        # ignores the passed pointers / declared DRAM layout) — NOT an infra tool_crash. Classify it
        # as a functional fault with actionable feedback; mislabeling it tool_crash reads as an
        # unfixable infra problem and wastes every round (the atlas flat-0/11 pattern).
        _trapped = ("*** FAILED ***" in _msg) and not _did_not_halt
        # WALL-CLOCK, not cycle-cap. `subprocess.TimeoutExpired` stringifies as "Command '...'
        # timed out after N seconds"; `did not halt` is the other, opposite condition and wins.
        _timed_out = ("timed out after" in _msg) and not _did_not_halt
        # THIRD CASE, same lesson as the two above. A simulator that aborts with a C++
        # std::out_of_range / _M_range_check is reporting the AGENT indexing an on-chip resource
        # one past its end -- the sim dies instead of returning a verdict, so it surfaces as an
        # infra `tool_crash` carrying only a C++ backtrace. MEASURED 2026-09-08: an arm hit this
        # 8 times across several grades and never fixed it, because "terminate called after
        # throwing an instance of 'std::out_of_range'" names neither the resource, nor the index,
        # nor the limit. No other arm triggered it, so it read as that arm being broken rather
        # than as an unreadable diagnostic. Parse the bound out and say what was exceeded.
        _oob_idx, _oob_size = _range_check_bounds(_msg)
        _oob = _oob_idx is not None
        # A link failure is likewise the agent's build defect (its emitted object/driver does not
        # link), not infra -- and the ld error is the whole diagnostic, so it must not be clipped
        # to the same short budget as a generic crash.
        _link_failed = "link failed" in _msg and not _did_not_halt
        # A STOPPED run keeps its partial console and the kernel's cycles printed before it.
        _partial = _partial_console.recover(paths.generated) if _timed_out else None
        _partial_log = _record_console(paths, f"{_sim}_partial", _partial)[0]
        evidence.tiers[tier] = TierResult(
            tier,
            "fail",
            mand,
            cycles=_partial_console.metrics(_partial).get("cycles"),
            evidence=_partial_log,
            reason=(
                _did_not_halt_reason(_msg)
                if _did_not_halt
                else _oracle_timeout_reason(_sim, _msg)
                if _timed_out
                else f"kernel faulted at runtime: {_clip(_msg, 260)}"
                if _trapped
                else _out_of_range_reason(_sim, _oob_idx, _oob_size)
                if _oob
                else f"the emitted package did not LINK (your build defect, not the simulator's): {_clip(_msg, 600)}"
                if _link_failed
                else f"{_sim} crash: {_clip(_msg, 300)}"
            ),
            derived_from_rtl=tier in cfg.rtl_tiers,
            timing=_failed_adapter_timing(),
            concurrency=concurrency_stamp(workers),
        )
        if mand:
            if _did_not_halt:
                raise _did_not_halt_failure(_msg) from e
            if _trapped:
                raise CertFailure(
                    _sim,
                    _cat("FUNCTIONAL_MISMATCH"),
                    "the emitted kernel RAN but FAULTED on "
                    f"{_sim} (a trap — an illegal instruction or an out-of-range "
                    "memory access — before producing output). Check that every "
                    "memory-movement instruction uses a valid, in-range DRAM address "
                    "(derive it from the passed pointer args / the declared DRAM "
                    f"layout, never a baked 0 or a guessed address): {_clip(_msg, 300)}",
                ) from e
                # Same reclassification on the MANDATORY path, or the actionable reason built above
                # is thrown away and the agent sees TOOL_CRASH again.
            if _oob:
                raise CertFailure(
                    _sim, _cat("FUNCTIONAL_MISMATCH"), _out_of_range_reason(_sim, _oob_idx, _oob_size)
                ) from e
            if _link_failed:
                raise CertFailure(
                    "link",
                    _cat("INTERFACE_MISMATCH"),
                    "the emitted package did not LINK, so nothing could be run. "
                    "This is a defect in what your compiler emitted (a missing or "
                    "duplicate symbol, or an object built against the wrong ABI), "
                    f"not a simulator fault: {_clip(_msg, 600)}",
                ) from e
            if _timed_out:
                # Still fails CLOSED -- a mandatory tier that produced no verdict is not a pass
                # (`not_run_is_not_pass`) -- but it is reported for what it is. Calling this a
                # tool crash sends the agent to debug a tool that did not break.
                raise NotMeasuredFailure(
                    _sim,
                    _cat("TESTBENCH_TIMEOUT"),
                    _why := _oracle_timeout_reason(_sim, _msg),
                    status=CERT_NOT_MEASURED if tier in cfg.rtl_tiers else "budget_exhausted",
                    tier=tier,
                    tier_reason=_why,
                    cycles_before_stop=_partial_console.metrics(_partial).get("cycles"),
                ) from e
            raise CertFailure(_sim, _cat("TOOL_CRASH"), f"{_sim} invocation failed: {_clip(_msg, 400)}") from e
        return OracleAttempt(carried_built=_executed_here)
    _adapter_wall = _time.perf_counter() - _adapter_t0
    return OracleAttempt(completed=True, result=res, elapsed_s=_adapter_wall)


# ---- oracle_result -----------------------------------------------------------------------------
# Capsule evaluation: oracle result.


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


# ---- ladder ------------------------------------------------------------------------------------
# Capsule evaluation: ladder.


def run_oracle_ladder(
    capsule,
    package_dir,
    paths,
    cfg,
    contract,
    timeout,
    eff_target,
    target,
    name,
    run_id,
    adapters,
    required,
    policy,
    workers,
    adapter_managed_tiers,
    perf_extractor,
    evidence,
    inputs,
):
    from merlin.development.targetgen.oracles.program_oracle import OracleUnavailable as _POUnavailable

    # An endpoint that RAN the program and watched it never halt reports a VERDICT, not an absent
    # oracle. It is raised as a SUBCLASS of the unavailable exception (so every fail-closed handler
    # still works) and must therefore be caught FIRST, or it collapses back into "the oracle did not
    # run" — which is what left a whole agent run with nothing it could act on.

    # Selected support providers express infrastructure absence through the
    # shared OracleUnavailable identity, retaining their own subclasses and
    # messages. Grading must not import an unrelated target to classify errors.
    _ORACLE_UNAVAILABLE = (OracleUnavailable, _POUnavailable)
    # Tiers the CAPSULE declares cannot corroborate a result for this target (see
    # corpus_spec._inapplicable_tiers). A tier whose model disagrees with the RTL about the machine
    # itself — not about precision — grades a correct kernel as wrong on every capsule, so running it
    # produces a permanent red the agent can neither fix nor learn from. It is reported the way L0/L1
    # already are for a float datapath: skipped + not_applicable, carrying the declared reason. It is
    # never reported as a pass, and a REQUIRED tier can never be declared inapplicable (that raises at
    # generation time), so this cannot be used to switch off a failing mandatory oracle.
    _inapplicable = capsule.get("inapplicable_oracle_tiers") or {}
    # CHEAPEST MEASURED TIER FIRST, not lexicographic. The ladder aborts on the first mandatory
    # tier that refutes a capsule, so the ORDER decides what a failing capsule costs -- and sorting
    # by name meant one target paid its arc cosim before reaching the cheaper Verilator tier
    # (⚠️ the 24.5 s / 0.29 s pair quoted here and below is a THROUGHPUT figure measured under 16-way
    # parallelism, not a latency. Measured serially over 42 samples 2026-08-29: arc median 3.68 s,
    # Verilator median 0.276 s, both linear in halt cycles (3.63 vs 0.255 ms/cyc) with no build step.
    # The ordering conclusion is unchanged -- 0.276 < 3.68 either way -- but anything PRICED off
    # 24.5 s is wrong by 6.3x. See docs/design/performance_budget_unit.md.)
    # that refutes the same capsules (measured: 12 of 12, identical signature). See tier_policy.
    evidence.execution_tiers = set(cfg.oracle_tiers) | set(adapters or {})
    _execution_ceiling = capsule.get("oracle_tier_ceiling")
    if _execution_ceiling is not None:
        # A materialized phase view is allowed to make LESS fidelity reachable than the endpoint
        # can provide.  Filter BEFORE cost ordering: unknown-first calibration would otherwise pick
        # an available L3 adapter ahead of the intended L2 search oracle and silently spend GSIM.
        # Tier names are fidelity indices, so only the canonical L<number> form has an ordering.
        def _fidelity_index(value: object) -> int | None:
            spelling = str(value).upper()
            return int(spelling[1:]) if spelling.startswith("L") and spelling[1:].isdigit() else None

        _ceiling_index = _fidelity_index(_execution_ceiling)
        if _ceiling_index is None:
            raise ValueError(
                f"invalid oracle_tier_ceiling {_execution_ceiling!r} for capsule {name!r}; "
                "expected a fidelity tier such as L2"
            )
        _tier_indices = {t: _fidelity_index(t) for t in evidence.execution_tiers}
        _bad_tiers = sorted(t for t, index in _tier_indices.items() if index is None)
        if _bad_tiers:
            raise ValueError(
                f"cannot apply oracle_tier_ceiling {_execution_ceiling!r} to non-fidelity "
                f"tier name(s) {_bad_tiers} for capsule {name!r}"
            )
        evidence.execution_tiers = {
            t for t, index in _tier_indices.items() if index is not None and index <= _ceiling_index
        }
    _tier_seq = _tier_policy.tier_order(str(cfg.target or target or ""), evidence.execution_tiers)
    _screen_tier = _tier_seq[0] if _tier_seq else None
    _co_step = _CO_GATE.stepper(eff_target, _tier_seq, cfg.rtl_tiers, adapters)  # no RTL adapter: defer
    # When set, a mandatory tier failure does not abort the ladder: the remaining tiers still run and
    # record, and the FIRST failure is raised once the loop completes. Costs the later tiers on a
    # failing capsule, which is the price of being able to say what every tier thought.
    _complete_ladder = _full_ladder_enabled()
    _first_cert_failure: CertFailure | None = None
    # THE EXECUTABLE THIS LADDER MEASURES MUST BE THIS RUN'S. See the note above
    # `_clear_stale_executable`: a reused run directory holds the previous grade's ELF, and every
    # consumer that content-addresses "the program this capsule ran" would answer with it.
    _clear_stale_executable(paths.generated)
    # Resolved once per capsule and only if a tier ever asks: `toolchain_shas` walks every hardware
    # pin the target declares, and the ladder would otherwise repeat that per tier.
    _shas_memo: dict = {}

    def _pin_shas():
        if "v" not in _shas_memo:
            _shas_memo["v"] = toolchain_shas(eff_target)
        return _shas_memo["v"]

    _cache_refusals: dict = {}  # tier -> why its certificate could not even be looked up
    _executed_here = False  # has ANY tier actually run in this process, for this capsule?

    def _stamp_cache_refusals() -> None:
        stamp_cache_refusals(evidence.tiers, _cache_refusals)

    for tier in _tier_seq:
        mand = tier in required
        # THE SCREEN IS ALWAYS PAID FOR; the tiers above it are what a budget can decline. A capsule
        # inside the derived covering set is never declined -- that is what the cover is for.
        if tier != _screen_tier:
            # THE CAPSULE'S OWN DECLARED CEILING, honoured HERE and not only in promotion.
            # `tier_policy.oracle_ceiling` was called by `tier_promote` (which enqueues cert jobs
            # between grades) and by nothing inside the grade itself, even though this TierResult
            # already carries an `oracle_ceiling` field to record it. So a capsule declaring
            # `max_oracle_tier: L2` with a VERIFIED `extends` still had the cert tier run on it
            # during every grade. MEASURED on one run: six such capsules, each declaring L2 and each
            # resting on a sibling that did earn L3, ran the cert tier to the 1200s per-capsule
            # timeout and failed there -- 7230s of adapter wall, about half the grade, spent
            # re-certifying claims their own declarations said they were not making.
            _ceil = _tier_policy.oracle_ceiling(
                str(cfg.target or target or ""),
                capsule,
                tier,
                declared_tiers=_tier_seq,
                # A named sibling must be present in this selected suite run, not an ambient
                # certificate under the checkout's default output roots. Cost history remains
                # separately selected by the affordability policy.
                extends_roots=[paths.run_path.parent],
            )
            if not _ceil.allowed:
                evidence.tiers[tier] = TierResult(
                    tier,
                    "skipped",
                    mand,
                    reason=_ceil.reason,
                    derived_from_rtl=tier in cfg.rtl_tiers,
                    oracle_ceiling=_ceil.record,
                )
                continue
            _may, _why = _tier_policy.may_certify(str(cfg.target or target or ""), capsule)
            if not _may:
                evidence.tiers[tier] = TierResult(
                    tier, "skipped", mand, reason=_why, budget_deferred=True, derived_from_rtl=tier in cfg.rtl_tiers
                )
                continue
        if tier in _inapplicable and not mand:
            evidence.tiers[tier] = TierResult(
                tier,
                "skipped",
                mand,
                not_applicable=True,
                reason=str(_inapplicable[tier]),
                derived_from_rtl=tier in cfg.rtl_tiers,
            )
            continue
        adapter = (adapters or {}).get(tier)
        if adapter is None:
            if mand:
                # FAIL CLOSED on the DECLARED tier, and name it. Never substitute another tier and
                # report its verdict as the declared one — a grade that silently rides a tier the
                # capsule never asked for is not the grade the capsule specified.
                evidence.tiers[tier] = TierResult(
                    tier,
                    "unavailable",
                    True,
                    reason=(
                        f"this capsule declares required oracle tier {tier} "
                        f"({cfg.tier_sim.get(tier, 'no sim declared')}) but this phase supplies "
                        f"no adapter for it; reachable tiers here: "
                        f"{sorted(adapters or {}) or 'none'}"
                    ),
                    derived_from_rtl=tier in cfg.rtl_tiers,
                )
            continue
            # DO NOT RE-BUY A CERTIFICATE FOR BYTES THAT HAVE NOT CHANGED. Asked only once a tier has
            # actually executed in this run, because that is what guarantees the executable on disk is
            # this run's and not a previous grade's (see `_clear_stale_executable`). The lookup fails
            # closed on every ambiguity, and a hit is reported AS a carry -- never as a measurement.
        from merlin.development.targetgen.oracles.certificates import capture as _simulation_request

        _simulation_context = None
        if tier in cfg.rtl_tiers and tier not in adapter_managed_tiers:
            _simulation_context = _simulation_request(
                capsule=capsule,
                cb=evidence.command_buffer,
                gold=inputs.gold,
                reference=inputs.reference,
                direct_oracle_golden=inputs.direct_oracle_golden,
                policy=policy,
                llvm_text=evidence.lowered_text,
                adapter=adapter,
                target=eff_target,
                tier=tier,
                package_dir=package_dir,
            )
        if _executed_here and tier not in adapter_managed_tiers:
            _carried, _why_not = carried_tier_result(
                name,
                tier,
                mand,
                target=eff_target,
                generated=paths.generated,
                shas=_pin_shas(),
                from_rtl=tier in cfg.rtl_tiers,
                simulation_context=_simulation_context,
            )
            if _carried is not None:
                evidence.tiers[tier] = _carried
                continue
            if _why_not:
                # The question could not be ASKED -- distinct from asking and missing, and the
                # distinction is the whole point: unrecorded, the two read identically.
                _cache_refusals[tier] = _why_not
        attempt = invoke_oracle(
            capsule,
            package_dir,
            paths,
            cfg,
            timeout,
            eff_target,
            name,
            run_id,
            tier,
            mand,
            adapter,
            workers,
            adapter_managed_tiers,
            policy,
            evidence,
            inputs,
            _pin_shas,
            _simulation_context,
            _executed_here,
        )
        if not attempt.completed:
            _executed_here = _executed_here or attempt.carried_built
            continue
        res, _adapter_wall = attempt.result, attempt.elapsed_s
        # Split active (build + sim) vs waiting (queue/FPGA slot). Adapters that route through a
        # queue set oracle_wait_s themselves; for the rest, any wall not accounted by build+sim is
        # treated as wait (≈0 for local spike/verilator).
        _tm = dict(res.get("timing") or {})
        _tm.setdefault("build_s", None)
        _tm.setdefault("sim_active_s", None)
        if _tm.get("oracle_wait_s") in (None, 0.0):
            _acct = (_tm.get("build_s") or 0.0) + (_tm.get("sim_active_s") or 0.0)
            _tm["oracle_wait_s"] = round(max(0.0, _adapter_wall - _acct), 3)
        _tm["adapter_wall_s"] = round(_adapter_wall, 3)
        # STAMPED HERE, at the moment of measurement -- not at suite start. The load average this
        # reads is the one covering the adapter call whose wall time sits beside it.
        _conc = concurrency_stamp(workers)
        _tier_policy.record_cost(str(cfg.target or target or ""), tier, _adapter_wall)
        if tier != _screen_tier:
            _tier_policy.note_spend(str(cfg.target or target or ""), _adapter_wall)
        measurement = assess_oracle(
            capsule, package_dir, paths, cfg, tier, mand, res, _tm, _conc, perf_extractor, policy, evidence, inputs
        )
        if measurement is None:
            continue
        okt = measurement.passed
        _mismatch_reason = measurement.mismatch_reason
        sim_name = measurement.simulator
        _oracle_meta = measurement.oracle
        # Live cross-engine custody must capture the same complete tier
        # record that finalization publishes, before another engine reuses
        # its build paths. These labels grant no numerical or cache credit.
        evidence.tiers[tier].submission = submission_identity(package_dir, run_id=run_id)
        _stamp_cache_refusals()
        # Capture original L2 custody after numerical comparison and before
        # a later engine can reuse mutable build paths. No observer means
        # no work; this grants only protected model scheduling evidence.
        from merlin.development.backend.feedback.certification.rolling_model_gate import capture_tier

        capture_tier(
            name, tier, evidence.tiers[tier], generated=paths.generated, package_dir=package_dir, shas=_pin_shas()
        )
        # THIS TIER WAS PAID FOR; write down what it bought, against the bytes and the instrument
        # that produced it, so the next grade need not buy it again. Only a pass is stored (a
        # failure is what an agent acts on and is re-run for its detail); never raises.
        _executed_here = True
        record_tier_certificate(
            name,
            tier,
            evidence.tiers[tier],
            target=eff_target,
            generated=paths.generated,
            shas=_pin_shas(),
            from_rtl=tier in cfg.rtl_tiers,
            run_id=run_id,
            simulation_context=_simulation_context,
        )
        # COMPLETION ORDER: does this pass rely on commands completing in an order the hardware lacks?
        if okt and mand and tier not in cfg.rtl_tiers and evidence.completion_order is None:
            evidence.completion_order, _co_fail = _co_step(tier, evidence.tiers, paths.generated)
            if _co_fail is not None and not _complete_ladder:
                _CO_GATE.suppress_later(
                    evidence.tiers, _tier_seq, tier, required, cfg.rtl_tiers, suppressed_tier_result
                )
                _stamp_cache_refusals()
                raise _co_fail
            _first_cert_failure = _first_cert_failure or _co_fail
            # Only a MANDATORY/gold tier mismatch fails the capsule. An ADDITIVE lower-fidelity tier
            # (one not in required_oracle_tiers — e.g. a fast functional model with known approximation
            # gaps) records its fail in the tiers dict but must NOT abort: aborting here would pre-empt the
            # required RTL oracle that follows in the sorted ladder — the same "a cheaper check short-
            # circuits the authoritative oracle" class we fixed on the trace side, here on the oracle side.
        if not okt and mand:
            _cf = CertFailure(
                sim_name,
                _cat("FUNCTIONAL_MISMATCH"),
                _mismatch_reason
                + _encoding_divergence_hint(
                    evidence.trace_check,
                    inputs.direct_oracle_golden,
                    cb=evidence.command_buffer,
                    capsule=capsule,
                    trace=evidence.decoded_trace,
                ),
            )
            # COMPLETE THE LADDER, then fail. Raising here aborts the loop, so every tier ordered
            # AFTER the refuting one is left with no record at all -- not "skipped", absent. Measured
            # on atlas: 11 of 26 capsules carried an L4 fail and NO L3 entry, 1 the reverse, because
            # tier_order runs the cheaper Verilator tier before the arc cosim. A tier with no record
            # is not evidence (`not_run_is_not_pass`), and it makes "these 12 failed" unanswerable at
            # the other tier -- which is exactly what you need when the shared defect is fixed.
            # The capsule still fails, on the FIRST refuting plane; only the remaining tiers now run.
            if _first_cert_failure is None:
                _first_cert_failure = _cf
            if not _complete_ladder:
                # The opt-out still records: a tier the ladder declined to run says `skipped`,
                # with the reason, never `fail` and never nothing at all.
                for _later in _tier_seq[_tier_seq.index(tier) + 1 :]:
                    if _later not in evidence.tiers:
                        evidence.tiers[_later] = suppressed_tier_result(
                            _later, _later in required, tier, from_rtl=_later in cfg.rtl_tiers
                        )
                _stamp_cache_refusals()
                raise _cf

    _stamp_cache_refusals()
    if _first_cert_failure is None and evidence.completion_order is None:  # a carried functional tier: check it here
        evidence.completion_order, _first_cert_failure = _co_step(None, evidence.tiers, paths.generated, cfg)
    if _first_cert_failure is None:
        # MEMORY ORDER (opt-in): a certificate earned on an in-order memory model says nothing about a
        # program whose answer depends on two loads completing in issue order. Its trigger is the
        # completion-order verdict above when one was reached.
        _order =_order_after_certificate (eff_target ,cfg ,evidence .tiers ,evidence .decoded_trace ,paths .generated ,
        gold =inputs .gold ,policy =policy ,gsource =inputs .golden_source ,
        trigger =evidence .completion_order ,
        readback_outputs =inputs .output_contract ["readback_outputs"])  # fmt: skip
        if _order is not None:
            evidence.order_check = _order.record
            if _order.failed:
                _cert_tier = evidence.order_check["cert_tier"]
                evidence.tiers[_cert_tier] = dataclasses.replace(
                    evidence.tiers[_cert_tier], status="fail", reason=_order.reason
                )
                _first_cert_failure = CertFailure(f"{_cert_tier}_order", _cat("FUNCTIONAL_MISMATCH"), _order.reason)
    if _first_cert_failure is not None:
        raise _first_cert_failure
