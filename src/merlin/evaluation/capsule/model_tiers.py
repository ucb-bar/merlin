"""Capsule evaluation: model tiers."""

from __future__ import annotations

from merlin.evaluation.capsule.tier import TierResult as TierResult
from merlin.targetgen.oracle_policy import _screen_tiers_of


def _rtl_tiers_of(target: str | None) -> frozenset[str]:
    """The tiers this target counts as RTL-derived, from its capability manifest. Fails soft to an empty
    set so the caller falls back to the capsule's own declaration rather than guessing a tier name."""
    if not target:
        return frozenset()
    try:
        from merlin.targetgen.runner_config import runner_config_from_manifest
        from merlin.targetgen.target_experiment import load_capability_manifest

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
        from merlin.targetgen.target_experiment import batched_oracle_tiers

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
