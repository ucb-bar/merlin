"""Capsule evaluation: ladder."""

from __future__ import annotations

import dataclasses

from merlin.evaluation.capsule.adapters import OracleUnavailable
from merlin.evaluation.capsule.availability import _full_ladder_enabled
from merlin.evaluation.capsule.certificates import (
    _clear_stale_executable,
    _order_after_certificate,
    carried_tier_result,
    record_tier_certificate,
    stamp_cache_refusals,
    suppressed_tier_result,
)
from merlin.evaluation.capsule.identity import concurrency_stamp, submission_identity
from merlin.evaluation.capsule.numerics import _encoding_divergence_hint
from merlin.evaluation.capsule.oracle_call import invoke_oracle
from merlin.evaluation.capsule.oracle_result import assess_oracle
from merlin.evaluation.capsule.tier import TierResult as TierResult
from merlin.targetgen import completion_order_gate as _CO_GATE
from merlin.targetgen import tier_policy as _tier_policy
from merlin.targetgen.capsule_common import _cat
from merlin.targetgen.oot_runner import CertFailure
from merlin.targetgen.provenance import toolchain_shas


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
    from merlin.targetgen.program_oracle import OracleUnavailable as _POUnavailable

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
        from merlin.targetgen.simulation_certificate import capture as _simulation_request

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
        from merlin.experiments.phase1.feedback.rolling_model_gate import capture_tier

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
        _order = _order_after_certificate(eff_target, cfg, evidence.tiers, evidence.decoded_trace, paths.generated,
                                          gold=inputs.gold, policy=policy, gsource=inputs.golden_source,
                                          trigger=evidence.completion_order,
                                          readback_outputs=inputs.output_contract["readback_outputs"])  # fmt: skip
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
