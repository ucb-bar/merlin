"""Capsule evaluation: oracle call."""

from __future__ import annotations

from merlin.development.backend.feedback import setup_timing as _host_timing
from merlin.evaluation.capsule.adapters import OracleUnavailable
from merlin.evaluation.capsule.certificates import _CarriedPreparedTier, _prepared_tier_reuse
from merlin.evaluation.capsule.diagnostics import _clip as _clip
from merlin.evaluation.capsule.diagnostics import _range_check_bounds as _range_check_bounds
from merlin.evaluation.capsule.diagnostics import _record_console as _record_console
from merlin.evaluation.capsule.identity import concurrency_stamp
from merlin.evaluation.capsule.lanes_evidence import oracle_source_scope
from merlin.evaluation.capsule.numerics import (
    _did_not_halt_failure,
    _did_not_halt_reason,
    _is_readback_refusal,
    _oracle_timeout_reason,
    _out_of_range_reason,
    _readback_refusal_detail,
)
from merlin.evaluation.capsule.records import OracleAttempt
from merlin.evaluation.capsule.tier import TierResult as TierResult
from merlin.targetgen import partial_console as _partial_console
from merlin.targetgen.capsule_common import _cat
from merlin.targetgen.oot_runner import INFRASTRUCTURE_PLANE, CertFailure, InfraCategory, InfraFailure
from merlin.targetgen.package_runtime import NotMeasuredFailure
from merlin.targetgen.tier_integrity import CERT_NOT_MEASURED


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
    from merlin.targetgen.program_oracle import OracleUnavailable as _POUnavailable
    from merlin.targetgen.program_oracle import ProgramDidNotHalt as _PODidNotHalt

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
