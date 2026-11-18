"""Tier result record and its stable serialized measurement fields."""

from __future__ import annotations

import dataclasses
from merlin.development.targetgen.oracle_policy import _screen_tiers_of


# ---- tier --------------------------------------------------------------------------------------


@dataclasses.dataclass
class TierResult:
    tier: str
    status: str  # pass | fail | skipped | unavailable
    mandatory: bool
    reason: str | None = None
    cache_unavailable: str = ""  # why this tier's certificate could not even be LOOKED UP.
    # Distinct from a lookup that missed: both used to render as an
    # absence, which is how a certificate cache that had never once
    # fired for two of three targets went unnoticed. Rides the
    # result so a reader sees it without re-deriving anything.
    cycles: int | None = None
    derived_from_rtl: bool = False
    cycle_accurate: bool = False
    evidence: str | None = None
    timing: dict | None = None  # {build_s, sim_active_s, oracle_wait_s} — active vs waiting
    gflops: float | None = None  # perf (SIMT); None -> omitted so systolic output is unchanged
    pct_fp_peak: float | None = None
    timing_observations: list | None = None
    # FINER TIMING THE ORACLE COULD ACTUALLY SEE: a list of
    # {quantity, value, unit, concurrent, note} entries (per-unit
    # busy, per-op latency, per-cycle activity). Every functional
    # run already executes on a real oracle, and discarding what it
    # timed throws away calibration evidence the run already paid
    # for. An adapter with NO timing capability emits nothing here
    # — not the key, not a list of zeros, because "not reported"
    # and "cost nothing" are different facts. None -> omitted, so
    # an adapter that carries none is byte-identical to before.
    counters: dict | None = None  # COUNTS the oracle reported alongside the cycle total — bytes
    # moved, reuse hits, commits. Deliberately NOT `utilization`:
    # those are fractions of a cycle window and these are integers
    # of a different kind, and filing a byte count under a name
    # that means "fraction of time" is how a number gets read as
    # something it is not. A model that computes these and an
    # adapter that drops them are indistinguishable downstream,
    # which is what happened: the arc oracle has reported movement
    # and residency counts all along and the adapter returned only
    # cycles. None -> omitted, so a target reporting none is
    # byte-identical to before.
    utilization: dict | None = None  # WHERE THE TIME WENT, as fractions of the oracle's own cycle
    # window (warp occupancy, per-unit busy, memory conflicts).
    # Latency says a kernel is slow; this says why, which is the
    # difference between an actionable result and a number. Shape is
    # the target's to define; None -> omitted, so a target whose
    # oracle reports no counters is byte-identical to before.
    oracle_ceiling: dict | None = None
    # WHY THIS TIER WAS NOT BOUGHT, when a per-capsule oracle-tier
    # ceiling declined it: the cap, where it came from (declared /
    # derived from measured cost / unpriced), the budget that set it,
    # the `extends` sibling the claim rests on, and the affordability
    # calculation. A capped tier is ALWAYS recorded -- `skipped` with
    # this block -- never omitted: a tier with no record is not
    # evidence, and an absent key is what made two model capsules read
    # as though they had never been graded.
    # See merlin.development.targetgen.tier_policy.oracle_ceiling.
    not_applicable: bool = False  # tier honestly N/A for this capsule's datatype (e.g. the integer
    # L0/L1 floor on a float datapath) — a legitimate skip, not a
    # not_run_is_not_pass violation (unlike an unavailable RTL oracle)
    budget_deferred: bool = False  # tier APPLIED and was DELIBERATELY NOT PAID FOR: this capsule is
    # outside the derived covering set and the certify budget is gone.
    # Never a pass, never a fail — the capsule was screened, not
    # certified, and it says so by name.
    fidelity: str | None = None  # WHAT THE ORACLE ACTUALLY WAS, in the oracle's own words:
    # elaborated_rtl | rtl_derived_model | functional_model. The tier
    # NAME cannot carry this — one target's L3 is Verilator and
    # another's is a model — so a reader of the record could not tell
    # a hardware verdict from a model one.

    concurrency: dict | None = None  # WHAT ELSE THE HOST WAS DOING WHILE THIS WAS MEASURED --
    # {workers, nproc, load_avg, serial}, stamped at the moment of
    # measurement. CYCLES are concurrency-invariant (verified:
    # identical serial and at 16 workers); WALL TIMES are not (the
    # same query measured 3.7 s serial and 23.4 s at 16 workers, a
    # 6.3x spread). So every ``timing`` block is only interpretable
    # next to the concurrency it was taken at, and a block without
    # one is not comparable with any other. Absent -> the run
    # predates the stamp; see merlin.analysis.perf.observations.
    # concurrency_of, which marks that rather than assuming it.
    submission: dict | None = None  # WHICH SUBMISSION these numbers are about, stated rather than
    # inferred from where the file sits. Cycles are a property of
    # the submission, not of the capsule: the SAME capsule has
    # measured 1090 / 3078 / 8889 across three submissions of one
    # task, an 8.2x spread on identical inputs. A path heuristic
    # that mis-keys them pools three different programs into one
    # "latency", so the field is explicit and the heuristic becomes
    # the fallback it should always have been.
    timing_capability: dict | None = None
    # WHAT THE INSTRUMENT COULD AND COULD NOT SEE, beside the
    # observations it produced: which units went unread, whether
    # its buckets partition the timeline, its alias accounting, and
    # everything it refused. An unread signal is UNMEASURED, never
    # zero, and this is where that distinction is recorded.

    engine: str | None = None  # WHICH SIMULATOR ANSWERED, as a first-class field.
    # It was recorded nowhere: the only trace of it was the `evidence`
    # FILENAME (`<engine>_console.log`), so telling a GSIM cert from a
    # Verilator one meant string-parsing a path, and every consumer
    # that did not (the verdict, the perf ledger, cert_cost) simply
    # could not see which engine produced the seconds it was fitting.
    # A cert tier is a fidelity, not a simulator -- which is exactly
    # why the simulator has to be written down.
    engine_selection: dict | None = None
    # HOW that engine was chosen and what it was chosen OVER, from
    # `rtl_engine_policy.select`: {engine, fidelity, reason,
    # passed_over, considered}. Falling back to the slow engine was
    # SILENT -- the run recorded a cert and not the fact that the fast
    # one was unavailable, so nobody learned there was anything to fix.
    sim_provenance: dict | None = None
    # WHICH BUILD of that simulator: binary digest, and the build
    # receipt binding it to an RTL revision when one exists. A result
    # that claims a hardware verdict must record which hardware
    # revision it came from, and a prebuilt simulator binary is the
    # one input to that verdict nothing else in the record identifies.
    console_log: str | None = None  # The sim console FILE (name under the capsule's artifacts dir) and
    console_bytes: int | None = None  # its size. `evidence` already names it, but only by convention;
    # a perf-tuning pass reading these records should not have to
    # reconstruct the filename to find the log it needs.

    carried: dict | None = None  # THIS TIER WAS NOT EXECUTED IN THIS RUN. Present only on a record
    # whose verdict was CARRIED from an earlier execution of the same
    # program on the same instrument (merlin.development.targetgen.tier_cache):
    # it names the execution identity and instrument digest the
    # verdict was earned under, when, and by which run. A reused
    # certificate reported as a fresh measurement is not a saving, it
    # is a false claim about what this run did -- so the record says
    # so, and `to_dict` additionally stamps `measured_now: false` on
    # a carried record and `true` on every executed one, so the
    # distinction survives into anything that reads only the JSON.
    toolchain: str | None = None  # WHICH PROGRAM was graded, as reported by the adapter. A block-
    # scaled MX capsule is graded on the harness's own reference MX
    # kernel rather than the submission, so a pass there measures the
    # fixture. Recording it keeps a score decomposable instead of
    # silently overstating the backend by the size of the MX set.

    def to_dict(self) -> dict:
        d = {
            "status": self.status,
            "mandatory": self.mandatory,
            "not_run_is_not_pass": True,
            "reason": self.reason,
            "cycles": self.cycles,
            "derived_from_rtl": self.derived_from_rtl,
            "cycle_accurate": self.cycle_accurate,
            "evidence": self.evidence,
            "timing": self.timing,
        }
        if self.cache_unavailable:
            d["cache_unavailable"] = self.cache_unavailable
        if self.not_applicable:
            d["not_applicable"] = True
        if self.budget_deferred:
            d["budget_deferred"] = True
        if self.oracle_ceiling:
            d["oracle_ceiling"] = dict(self.oracle_ceiling)
        if self.fidelity:
            d["fidelity"] = self.fidelity
            # perf fields ride the result ONLY when populated (SIMT) — keeps systolic output byte-identical.
        if self.gflops is not None:
            d["gflops"] = self.gflops
        if self.pct_fp_peak is not None:
            d["pct_fp_peak"] = self.pct_fp_peak
        if self.counters:
            d["counters"] = dict(self.counters)
        if self.utilization:
            d["utilization"] = dict(self.utilization)
        if self.timing_observations:
            d["timing_observations"] = list(self.timing_observations)
        if self.timing_capability:
            d["timing_capability"] = dict(self.timing_capability)
            # Concurrency and submission ride the record ONLY when they were established. An absent
            # field is "unrecorded", which is the honest state for every run taken before the stamp
            # existed; a defaulted one would be a claim.
        if self.concurrency:
            d["concurrency"] = dict(self.concurrency)
        if self.submission:
            d["submission"] = dict(self.submission)
            # unchanged. This is what separates "the submission passed" from "the harness fixture passed":
            # a block-scaled MX capsule is graded on the reference MX kernel, not the submitted backend.
        if self.toolchain:
            d["toolchain"] = self.toolchain
            # SIMULATOR TELEMETRY. Omitted when unset, so a record from an adapter that reports none is
            # byte-identical to before; present, it is what a perf-tuning pass reads to know which engine
            # produced these cycles and seconds, which build of it, and where its console went.
        if self.engine:
            d["engine"] = self.engine
        if self.engine_selection:
            d["engine_selection"] = dict(self.engine_selection)
        if self.sim_provenance:
            d["sim_provenance"] = dict(self.sim_provenance)
        if self.console_log:
            d["console_log"] = self.console_log
        if self.console_bytes is not None:
            d["console_bytes"] = self.console_bytes
            # WAS THIS MEASURED NOW? Stamped on EVERY record, not only on a carried one: a reader that has
            # to infer "measured" from the ABSENCE of a key cannot tell a fresh record from one written by
            # a version that did not know about carrying. Both states are asserted.
        d["measured_now"] = self.carried is None
        if self.carried:
            d["carried"] = dict(self.carried)
        return d


# ---- model_tiers -------------------------------------------------------------------------------
# Capsule evaluation: model tiers.


def _rtl_tiers_of(target: str | None) -> frozenset[str]:
    """The tiers this target counts as RTL-derived, from its capability manifest. Fails soft to an empty
    set so the caller falls back to the capsule's own declaration rather than guessing a tier name."""
    if not target:
        return frozenset()
    try:
        from merlin.development.targetgen.runner_config import runner_config_from_manifest
        from merlin.development.targetgen.target_experiment import load_capability_manifest

        return frozenset(runner_config_from_manifest(load_capability_manifest(target)).rtl_tiers)
    except Exception:  # noqa: BLE001 — unresolvable manifest
        return frozenset()


def rtl_tiers_of(target: str | None) -> frozenset[str]:
    """Public accessor for the tiers ``target``'s capability manifest counts as RTL-derived.

    Callers outside this module (experiment harnesses that must RECORD which tiers a verdict could
    have been cited at, so a reader can re-derive the decision without the manifest in hand) were
    reaching for the private spelling; give them a supported one rather than growing that habit.
    """
    return _rtl_tiers_of(target)


def model_citable_rtl_tier(declared: list[str] | tuple[str, ...], target: str | None) -> str | None:
    """The ONE tier a whole-model capsule can be CITED at on ``target``, or ``None`` if there is none.

    A model capsule never enters the tier ladder: :func:`_model_tier_map` synthesises its tier block
    from the model's own layer accounting and emits exactly ONE execution tier -- the LAST declared
    tier the target's capability manifest counts as RTL. Everything else a capsule declares is either
    not_applicable (L0/L1 interpret a command buffer a whole model does not have) or simply never
    emitted, so an admission gate that demands, say, ``L2 == "pass"`` on a target whose RTL tiers begin
    at L3 reads ``None`` on a FLAWLESS run and refuses it unconditionally. That is exactly what an
    Arm-4 performance campaign hit: every full-model admission failed before a perf cell ever ran.

    Callers that gate on a citable hardware verdict must ask for this tier by derivation rather than
    writing one down, and must FAIL CLOSED on ``None`` -- a target/declaration pair with no RTL tier
    has no citable whole-model verdict to admit, and the non-RTL fallback ``_model_tier_map`` uses to
    keep a refusal attributable is a label, not a hardware claim.
    """
    rtl = [t for t in declared if t in _rtl_tiers_of(target)]
    return rtl[-1] if rtl else None


def _batched_tiers_of(target: str | None) -> frozenset[str]:
    """The tiers ``target``'s OWN descriptor declares as batched (measured N windows per job).

    Read from the descriptor rather than named here on purpose: the runner must not know that some
    particular tier means an FPGA behind a shared queue. A target that declares nothing yields the
    empty set and every existing path is unchanged.
    """
    try:
        from merlin.development.targetgen.target_experiment import batched_oracle_tiers

        return batched_oracle_tiers(target)
    except Exception:  # noqa: BLE001 — unresolvable descriptor
        return frozenset()


def _model_screen_tier(tier: str, sim: str, tile_exec: dict | None) -> TierResult:
    """The verdict for a whole model's functional SCREEN tier, read from its per-tile tally.

    The screen is the cheapest-first rung each synthesized tile clears before the cert oracle sees it
    (``mesh_tile_verification``: ``n_screened`` / ``n_screen_passed`` / ``n_screen_failed`` /
    ``n_screen_unavailable``). Four distinct facts, never collapsed:

      * no tile record at all             -> skipped (not applicable: this grade synthesized nothing)
      * tiles verified, no screen tally   -> unavailable (a real hole: the cheap rung was not run)
      * a tally that does not add up      -> unavailable (unaccounted tiles are UNKNOWN, never a pass)
      * otherwise                         -> fail if any failed, unavailable if any could not run,
                                             skipped if none were screened, pass if all passed
    """
    evidence = "mesh_tile_verification.per_tile[].screen"

    def _result(status: str, reason: str, *, not_applicable: bool = False) -> TierResult:
        return TierResult(
            tier,
            status,
            True,
            reason=reason,
            evidence=evidence,
            derived_from_rtl=False,
            cycle_accurate=False,
            not_applicable=not_applicable,
        )

    if not isinstance(tile_exec, dict) or not tile_exec:
        return _result(
            "skipped", "no tiles were synthesized for this grade, so there was nothing to screen", not_applicable=True
        )
    if tile_exec.get("n_screened") is None:
        return _result(
            "unavailable",
            f"tiles were verified but no {sim} screen tally was recorded, so whether they cleared the "
            f"{tier} screen is UNKNOWN",
        )
    n = int(tile_exec["n_screened"])
    passed = int(tile_exec.get("n_screen_passed") or 0)
    failed = int(tile_exec.get("n_screen_failed") or 0)
    unavailable = int(tile_exec.get("n_screen_unavailable") or 0)
    if passed + failed + unavailable != n:
        return _result(
            "unavailable",
            f"{n} tile(s) screened but only {passed + failed + unavailable} accounted for; the "
            f"unaccounted tiles' {tier} verdict is UNKNOWN",
        )
    if failed:
        return _result("fail", f"{failed} of {n} tile(s) failed the {sim} screen")
    if unavailable:
        return _result("unavailable", f"the {sim} screen could not run on {unavailable} of {n} tile(s)")
    if n == 0:
        return _result("skipped", "no tile was screened")
    return _result("pass", f"all {n} tile(s) passed the {sim} screen")


def _model_tier_map(
    declared: list[str],
    target: str | None,
    model_exec: dict | None,
    tile_exec: dict | None = None,
    *,
    measurement: dict | None = None,
) -> dict[str, TierResult]:
    """The tier block for a WHOLE-MODEL capsule, derived from the model's OWN layer accounting.

    A model capsule used to carry no tier block at all on most paths -- the derivation ran only when
    per-tile verification was present, and every fail-closed branch returned before it. So a capsule
    could report a verdict beside `tiers: {}`, which is how a suite reads N/N while nothing was
    certified. The tests that pin this read `tiers[<rtl tier>]["status"]`, so it must exist on EVERY
    path, including the ones that refuse.

    L0/L1 interpret a COMMAND BUFFER. A whole model has none, so they are honestly not_applicable --
    a legitimate skip, not a tier that failed to run.

    The RTL tier reads the model's own counters, and the four cases are deliberately distinct:

      * layers ran and none fell back  -> pass
      * no layer ran at all            -> skipped   (the target never ran this model)
      * some layer fell back           -> fail      (a hole in the claim, not a rounding error)
      * counters absent                -> unavailable (nobody could tell; absent must never read as 0)
    """
    tiers: dict[str, TierResult] = {}
    for t in declared:
        if t in ("L0", "L1"):
            tiers[t] = TierResult(
                t, "skipped", True, not_applicable=True, reason="a whole model has no command buffer to interpret"
            )
            # A BATCHED TIER IS NEVER DERIVED FROM THE MODEL'S OWN COUNTERS. The counters below say which
            # layers reached the accelerator in the run that produced them -- a statement about a SIMULATOR
            # run, not about the FPGA. A tier the target declares batched was measured elsewhere or not at
            # all, so it reads `unavailable` until a measurement is handed in, and `not_run_is_not_pass`
            # turns that into `incomplete`. It is recorded rather than omitted for the reason every other
            # tier is: a tier with no record is not evidence, and an absent key reads as though the capsule
            # was never graded at that tier at all.
    _batched = _batched_tiers_of(target)
    for t in declared:
        if t not in _batched or t in tiers:
            continue
        seen = (measurement or {}).get(t)
        if not isinstance(seen, dict) or not seen.get("status"):
            tiers[t] = TierResult(
                t,
                "unavailable",
                True,
                reason=f"{t} is a batched tier on this target: a capsule accumulates into a "
                "round-group and the group is measured as one queue job. No measurement "
                "for this capsule was handed to the grade, so whether it runs on the "
                "device is UNKNOWN",
            )
            continue
        tiers[t] = TierResult(
            t,
            str(seen["status"]),
            True,
            reason=str(seen.get("reason") or f"{t} measured as one window of a batched job"),
            cycles=seen.get("cycles"),
            derived_from_rtl=bool(seen.get("derived_from_rtl", True)),
            cycle_accurate=bool(seen.get("cycle_accurate", False)),
            evidence=(lambda v: str(v) if v else None)(seen.get("evidence")),
            fidelity=(lambda v: str(v) if v else None)(seen.get("fidelity")),
        )
        # A declared SCREEN tier (a cheap functional simulator below the RTL tiers, from the target's own
        # `tier_sim`) is read from the per-tile tally. Dropping it left a declared rung with no record,
        # which a downstream `tiers[<screen>] == "pass"` check reads as nothing at all.
    for t, sim in _screen_tiers_of(target):
        if t in declared and t not in tiers:
            tiers[t] = _model_screen_tier(t, sim, tile_exec)
            # The counter-derived verdict below belongs to the tier a SIMULATOR answered at, so a batched
            # tier is excluded from the citable-tier selection; without this, declaring the batched tier
            # would silently REPLACE the simulator tier's record with an unavailable one.
    _rtl = model_citable_rtl_tier([t for t in declared if t not in _batched], target)
    tier = _rtl if _rtl is not None else (declared[-1] if declared else None)
    if tier is None or tier in tiers:
        return tiers
    on = (model_exec or {}).get("matmul_layers_on_mesh")
    fb = (model_exec or {}).get("matmul_layers_host_fallback")
    if on is None:
        tiers[tier] = TierResult(
            tier,
            "unavailable",
            True,
            reason="no mesh execution counters were reported, so whether this "
            "model's layers ran on the accelerator is UNKNOWN",
        )
    elif int(fb or 0) > 0:
        tiers[tier] = TierResult(tier, "fail", True, reason=f"{int(fb)} matmul layer(s) fell back to the host kernel")
    elif int(on) == 0:
        tiers[tier] = TierResult(tier, "skipped", True, reason="no matmul layer executed on the accelerator")
    else:
        tiers[tier] = TierResult(tier, "pass", True, reason=f"{int(on)} matmul layer(s) executed on the accelerator")
    return tiers


def _source_region_execution_verdict(certificate: dict) -> tuple[str, str, str] | None:
    """Fail closed on the completed source-region/dispatch join for a required device model."""
    observed = certificate.get("source_region_execution")
    if not isinstance(observed, dict):
        return (
            "incomplete",
            "SOURCE_REGION_EXECUTION_NOT_MEASURED",
            "no complete source-region-to-completed-dispatch join was recorded; a routing plan "
            "or module census cannot establish which eligible operations actually ran on the accelerator",
        )
    host = observed.get("eligible_host_region_ids")
    mixed = observed.get("eligible_mixed_region_ids")
    accel = observed.get("eligible_accelerator_region_ids")
    total = observed.get("n_eligible_source_regions")
    if isinstance(host, list) and isinstance(mixed, list) and (host or mixed):
        return (
            "fail",
            "FALLBACK_ON_ELIGIBLE_REGION",
            "completed runtime calls placed eligible source regions on the host; "
            f"host={host[:8]}, mixed={mixed[:8]}. This is execution evidence, not merely a planned route.",
        )
    if observed.get("status") != "measured":
        return (
            "incomplete",
            "SOURCE_REGION_EXECUTION_NOT_MEASURED",
            "some eligible source regions could not be joined to completed dispatches; "
            "a planned route or a partial ledger is not execution evidence for the missing regions",
        )
    if observed.get("outline_inventory_status") != "matched":
        return (
            "incomplete",
            "SOURCE_REGION_EXECUTION_NOT_MEASURED",
            "the exact source-operation inventory did not reconcile with the runtime outline; "
            "a completed call for one operation cannot certify other operations in its region",
        )
    if (
        not isinstance(host, list)
        or not isinstance(mixed, list)
        or not isinstance(accel, list)
        or not isinstance(total, int)
        or total <= 0
    ):
        return (
            "incomplete",
            "SOURCE_REGION_EXECUTION_NOT_MEASURED",
            "the source-region execution join has no well-formed, nonempty eligible population",
        )
    if len(accel) != total:
        return (
            "incomplete",
            "SOURCE_REGION_EXECUTION_NOT_MEASURED",
            "the source-region execution join does not account for every eligible region",
        )
    return None


def _model_transform_audit_verdict(model_exec: dict) -> tuple[str, str, str] | None:
    """A target model grade needs replayed captured→normalized→outlined IR."""
    qualification = model_exec.get("transform_audit_qualification")
    if not isinstance(qualification, dict) or (
        qualification.get("status") != "structural_replay_matched"
        or qualification.get("normalization_replay") != "matched"
    ):
        return (
            "incomplete",
            "TRANSFORM_REPLAY_NOT_VERIFIED",
            "the exact captured model was not deterministically replayed through normalization "
            "and outlining; hashes or a green numerical result alone cannot verify the pass sequence",
        )
    return None
