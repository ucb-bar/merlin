"""Carry compiler observations into model results before admission decisions."""

from __future__ import annotations

import os
from pathlib import Path

from merlin.evaluation.capsule.evidence.lanes import dispatch_boundary_report, lane_report
from merlin.evaluation.capsule.evidence.lanes import dispatch_boundary_report as dispatch_boundary_report
from merlin.evaluation.capsule.evidence.source_custody import _MODEL_NUMERIC_UNSET, CONTRACT_VERSION
from merlin.evaluation.capsule.evidence.tiers import TierResult as TierResult
from merlin.evaluation.capsule.evidence.tiers import (
    _model_tier_map,
    _model_transform_audit_verdict,
    _rtl_tiers_of,
    _source_region_execution_verdict,
)
from merlin.evaluation.capsule.models.bundle import _model_runtime_bundle as _model_runtime_bundle
from merlin.evaluation.capsule.models.runtime import (
    _resolve_model_host_lane,
    _resolve_model_numeric_policy,
    _verified_model_host_lane_snapshot,
)
from merlin.evaluation.capsule.operations.numerics import (
    _numeric_not_compared,
    _numeric_when_not_accelerated,
    _unexercised_note,
)

# ---- model_evidence ----------------------------------------------------------------------------


def carry_model_evidence(capsule, result, out):
    if out.get("routing_plan") is not None:  # per-op mesh routing for the target
        result["routing_plan"] = out["routing_plan"]
    if out.get("coverage_certificate") is not None:  # ARR certificate (numerator×independent oracle)
        result["coverage_certificate"] = out["coverage_certificate"]
    if out.get("placement_census") is not None:  # whole-module population, including unmatched operations
        result["placement_census"] = out["placement_census"]
        # THE PLACEMENT SURFACE, carried into the verdict rather than left in the compile output. It runs in
        # SHADOW -- routing is the authority -- so this is reported, not gated. But two facts only it can
        # state were computed and read by nothing:
        #
        #   `n_emulated`  -- ops the host took in a format it cannot natively carry. The numeric gate hides
        #                    these completely: emulated arithmetic gets the right answer slowly, so a capsule
        #                    passes while the model runs on a datapath nobody chose.
        #   `divergence`  -- where placement and routing disagree. Flipping placement to authority is only
        #                    defensible once that set is empty, and until it is reported nobody can tell.
    if out.get("placement") is not None:
        _pl = out["placement"]
        result["placement"] = _pl
        _n_emul = _pl.get("n_emulated")
        if isinstance(_n_emul, int) and _n_emul > 0:
            result.setdefault("advisories", []).append(
                {
                    "plane": "placement",
                    "category": "EMULATED_ON_HOST",
                    "n": _n_emul,
                    "detail": f"{_n_emul} op(s) were placed on the host in a format it cannot natively "
                    f"carry; the numeric gate cannot see this, because emulated arithmetic is "
                    f"correct and merely slow",
                }
            )
    if out.get("mesh_execution") is not None:  # THIS MODEL's layers: on-mesh vs host
        result["mesh_execution"] = out["mesh_execution"]
    if out.get("mesh_tile_verification") is not None:  # synthesized tiles of the routed shapes
        result["mesh_tile_verification"] = out["mesh_tile_verification"]
    if result.get("mesh_execution") is not None:
        result["boundary_execution"] = dispatch_boundary_report(result["mesh_execution"])
        _lane = lane_report(capsule, result.get("routing_plan"), result["mesh_execution"])
        if _lane is not None:
            result["lane_report"] = _lane


# ---- model_assessment --------------------------------------------------------------------------
# Capsule evaluation: model assessment.


def assess_model_execution(capsule, result, out, *, target, dtype, run_where):
    st, gate = out.get("status"), (out.get("verify") or {}).get("gate_ok")
    engine = f"merlin-compile model --target {target} --run {run_where} --verify"
    carry_model_evidence(capsule, result, out)
    # A quantized whole model diverges from the fp32 golden BY DESIGN (int8/fp8 rounding). When the strict
    # gate rejects it only on that expected drop, accept it if the cosine similarity to the golden still
    # clears a quant floor (MERLIN_MODEL_QUANT_COS, default 0.90) — reasonable quantization error is a pass,
    # a gross codegen defect (cosine below the floor / structural mismatch) is still a fail.
    _v = out.get("verify") or {}
    _cos = max(float(_v.get("fp32_cos", 0.0)), float(_v.get("w8a8_cos", 0.0)))
    _quant = dtype in ("int8", "i8", "fp8", "fp8_e4m3")
    _quant_floor = float(os.environ.get("MERLIN_MODEL_QUANT_COS", "0.90") or "0.90")

    # A HOST RUN NEVER PASSES A MODEL CAPSULE, and never fails one either. It is merlin's own x86
    # dispatch runtime -- our compiler -- so however well or badly it does, it is evidence about US and
    # not about the submission. Reporting its mismatch as the capsule's `fail` attributes a defect in
    # our reference to a backend that never executed; reporting its match as `pass` is how a suite reads
    # N/N while the submitted backend never ran. So the diagnostic is RECORDED and the verdict WITHHELD.
    # No provenance stamp rides a withheld verdict: it makes no hardware claim to attribute.
    if run_where == "host":
        result["host_reference"] = {
            "status": st,
            "gate": _v,
            "engine": engine,
            "note": "merlin's own dispatch runtime — a diagnostic, never an oracle for a submitted backend",
        }
        result.update(
            status="not_gradeable_no_oracle",
            numeric=_numeric_not_compared(
                engine,
                "host_reference",
                _v,
                "the whole model ran on merlin's host dispatch runtime; the numeric verdict "
                "is withheld because that is our compiler, not the target",
            ),
            failure={
                "plane": "not_gradeable_no_oracle",
                "category": "NOT_GRADEABLE_NO_ORACLE",
                "detail": f"no target mesh ran this model (run={run_where}); a host "
                f"reference cannot certify a submitted backend",
            },
        )
        result.pop("provenance", None)
        return result
        # FAIL CLOSED ON AN UNEXERCISED TIER. A whole-model capsule declares required_oracle_tiers like any
        # other, but this function never entered the tier ladder, so those tiers were dead metadata and a
        # capsule could report `pass` with `tiers == {}` -- a verdict backed by no oracle at all. Worse, the
        # functional gate here is the HOST x86 run unless mesh execution was requested, so "pass" could mean
        # "the CPU computed the model correctly", which is not a statement about the accelerator.
        #
        # Record what actually ran, and refuse to call it a pass when nothing the capsule DECLARED ran. The
        # accelerator evidence is `mesh_tile_verification` (per-tile on-mesh certification); a routing plan is
        # a plan, not an execution, and is never counted as a tier.
    declared = [str(x) for x in (capsule.get("required_oracle_tiers") or [])]
    mesh_exec = out.get("mesh_tile_verification") or {}
    model_exec = out.get("mesh_execution") or {}
    n_tiles = int(mesh_exec.get("n_tiles") or 0) if isinstance(mesh_exec, dict) else 0
    # Set BEFORE the fail-closed branches below, every one of which returns early: a refusal must still
    # say which tier refused it, and this block used to be attached only on the success path. That
    # includes the transform-replay refusal: it returned ahead of this block, so every grade it
    # refused carried no tier record at all.
    _model_tiers = _model_tier_map(declared, target, model_exec, mesh_exec)
    result["tiers"] = {k: v.to_dict() for k, v in _model_tiers.items()}
    _transform_verdict = _model_transform_audit_verdict(model_exec)
    if _transform_verdict is not None:
        _status, _category, _detail = _transform_verdict
        result.update(status=_status, failure={"plane": "model", "category": _category, "detail": _detail})
        return result
    exercised: dict[str, str] = {}
    # THE LANE CONTRACT IS EVALUATED UNCONDITIONALLY. It used to sit inside `if n_tiles:` below, so a
    # capsule whose tile verification produced nothing -- no mesh_verify, no default OOT package, an
    # unavailable oracle -- had its declared lanes never checked at all, and `result["lane_report"]`
    # was simply absent. A capsule that names lanes is asserting composition; not looking is not the
    # same as looking and finding it satisfied.
    #
    # An interop capsule states the lanes it requires -- the routing plan's OWN keys (`on_mesh`,
    # `in_contract_vector_scalar`, `scalar_rvv_lane`) -- and passes only when EVERY named lane actually
    # carried work. Such a capsule withholds `must_accelerate` on purpose: host-lane work is the
    # behaviour under test, so "no host fallback" would be the wrong bar and would fail the exact thing
    # the capsule exists to prove.
    _lane_rep = lane_report(capsule, result.get("routing_plan"), model_exec, out.get("host_execution"))
    _lane_ok = True
    if _lane_rep is not None:
        result["lane_report"] = _lane_rep
        _lane_ok = not _lane_rep["unexercised"]
        # A LANE EVIDENCED ONLY BY THE PLAN IS NOT EVIDENCE. Passing on it would repeat, on the host
        # side, precisely the defect already fixed on the mesh side: 15 matmuls assigned to the mesh and
        # 15 host fallbacks at run time. `incomplete` rather than `fail` -- nothing about the submission
        # was disproved, we simply did not measure it.
        # A FORBIDDEN LANE THAT CARRIED WORK IS A FAIL, not an incomplete: this was measured, and the
        # compiler accelerated a family the target's manifest does not admit. That is as much a defect
        # as failing to accelerate an admitted one.
        if _lane_rep.get("violated"):
            _lane_ok = False
            result.update(
                status="fail",
                failure={
                    "plane": "model",
                    "category": "ACCELERATED_A_FORBIDDEN_LANE",
                    "detail": f"capsule forbids lane(s) {_lane_rep['violated']} but work "
                    f"executed there; this target's capability manifest does "
                    f"not admit this capsule's family, so the compiler must "
                    f"route it to the host lane",
                },
            )
            return result
            # A forbidden lane nobody measured is UNMEASURED, never satisfied. Absence of a routing-plan
            # entry is the default state, so accepting it would make the negative assertion vacuously true.
        if _lane_rep.get("unmeasured_forbidden"):
            result.update(
                status="incomplete",
                failure={
                    "plane": "model",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "detail": f"capsule forbids lane(s) "
                    f"{_lane_rep['unmeasured_forbidden']} but no execution "
                    f"accounting was recorded for them; a routing plan cannot "
                    f"establish that nothing ran",
                },
            )
            return result
        if _lane_ok and _lane_rep.get("plan_only_lanes"):
            result.update(
                status="incomplete",
                failure={
                    "plane": "model",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "detail": f"capsule requires lane(s) "
                    f"{_lane_rep['plan_only_lanes']} but only the ROUTING PLAN "
                    f"reports them; no execution accounting was recorded, so "
                    f"whether work reached those lanes is unmeasured",
                },
            )
            return result
    if n_tiles:
        # The verdict comes from the per-tile COUNTS. Reading a boolean `ok` key -- which this dict does
        # not carry -- made every on-mesh execution record "fail", including one where all 15 layers
        # passed on the oracle. A tile that was unavailable or unsynthesizable is NOT a pass either, so
        # the tier passes only when every tile is accounted for and every one of them passed.
        _passed = int(mesh_exec.get("n_passed") or 0)
        _failed = int(mesh_exec.get("n_failed") or 0)
        _unavail = int(mesh_exec.get("n_unavailable") or 0)
        _unsynth = int(mesh_exec.get("n_unsynthesizable") or 0)
        _ok = mesh_exec.get("ok")
        _tile_ok = (
            bool(_ok) if _ok is not None else (_failed == 0 and _unavail == 0 and _unsynth == 0 and _passed == n_tiles)
        )

        # THE MODEL'S OWN LAYERS DECIDE A MODEL CAPSULE'S TIER, not tiles of its shapes. Certifying a
        # synthesized tile proves the SHAPE is runnable; the capstone is a claim about THIS model, and
        # the two came apart once already -- a run with all 15 layers on the host reported "15 of 15
        # tiles passed". When the model ran the mesh lane, its own per-layer accounting must agree:
        # every routed layer on the accelerator, none fallen back. A host-lane model run has no such
        # accounting and is left to the tile record alone.
        _model_ok = True
        if model_exec:
            _on = model_exec.get("matmul_layers_on_mesh")
            _model_ok = _on is not None and int(_on) > 0 and not int(model_exec.get("matmul_layers_host_fallback") or 0)
            # INTEROP capsules invert one half of that. A capsule may DECLARE that its point is composition
            # across lanes -- part of the model on the accelerator, the rest on the scalar/vector lane the
            # target also owns -- in which case "no host fallback" is the wrong bar: it would fail the exact
            # behaviour under test. Such a capsule names the lanes it requires, and passes only when EVERY
            # named lane actually carried work.
            #
            # The lane names are the routing plan's OWN keys (`on_mesh`, `in_contract_vector_scalar`,
            # `scalar_rvv_lane`), so this asserts against what the compiler reported rather than a vocabulary
            # invented here; a capsule naming a lane the plan does not report fails closed with that lane
            # named, which is the actionable direction.

            # WHICH tier the mesh oracle corresponds to, DERIVED from the target's own declared RTL tiers.
            # This was `[x for x in declared if x not in ("L0", "L1")]` with an `"L3"` fallback -- three tier
            # -name literals standing in for a fact the manifest already carries, which names the wrong tier
            # confidently on any target whose ladder differs (e.g. one that grades at L3+L4, not L3).
        _rtl = [t for t in declared if t in _rtl_tiers_of(target)]
        _tier = _rtl[-1] if _rtl else (declared[-1] if declared else None)
        if _tier:
            exercised[_tier] = "pass" if (_tile_ok and _model_ok and _lane_ok) else "fail"
            # WHOSE COMPILER PASSED. When the runtime had to discharge a contract obligation the target backend
            # owes -- residency tiling for `capacity_fit` -- the verdict is about runtime+backend together, and
            # saying so is the difference between "this compiler handles a 512x512 layer" and "this layer ran".
    _delegated = (model_exec or {}).get("capacity_fit_delegated_to_runtime") or []
    if _delegated:
        # If ANY delegated layer was tiled from the backend's DECLARED tile rather than from a capacity
        # limit, this verdict rests on the runtime having driven a loop nest the backend does not have.
        # That is a different claim and the score reports it as one (`backend_coverage`).
        _tb = sorted({str(d.get("tiled_by")) for d in _delegated if d.get("tiled_by")})
        result["contract_obligations"] = {
            "capacity_fit": {
                "discharged_by": "merlin runtime (host-side residency tiling)",
                "tiled_by": (
                    "declared_primitive_tile" if "declared_primitive_tile" in _tb else (_tb[0] if _tb else None)
                ),
                "n_layers": len(_delegated),
                "layers": _delegated[:8],
                "detail": (
                    "the target backend did not satisfy capacity_fit at these extents; the "
                    "runtime split them so the model could run. This verdict is evidence about "
                    "the runtime AND the backend, not about the backend alone."
                ),
            }
        }
        # Overlay the per-tile verdict where one exists; the counter-derived map is the floor.
    for _t, _s in exercised.items():
        base = _model_tiers.get(_t)
        _model_tiers[_t] = TierResult(_t, _s, True, reason=(base.reason if base else None))
    result["tiers"] = {k: v.to_dict() for k, v in _model_tiers.items()}
    # The tile record, kept beside the verdict as the SEPARATE and weaker evidence it is: it speaks about
    # synthesized tiles of this model's shapes, never about the model.
    if mesh_exec:
        result["tile_evidence"] = {
            "n_tiles": n_tiles,
            "n_passed": mesh_exec.get("n_passed"),
            "note": "synthesized tiles OF THIS MODEL'S SHAPES — evidence that the shapes run, not that this model ran",
        }
    result["op_coverage"] = {
        "note": "the op-pass fraction this capsule was gated on is OP COVERAGE, not a verdict on the model"
    }
    unexercised = [x for x in declared if x not in exercised]

    # ORDER MATTERS: the SPECIFIC cause is reported before the generic consequence. Layers falling back
    # to the host now also drag the tier verdict down (the model, not its tiles, decides it), so leaving
    # the failed-tier branch first turned "2 of 37 layers fell back" into a bare FUNCTIONAL_MISMATCH --
    # true, and useless. The fallback check runs first so the result names what to fix.
    # MUST_ACCELERATE IS ABOUT THIS MODEL, NOT ABOUT TILES OF ITS SHAPES. Certifying a synthesized tile
    # proves the shape is runnable; it says nothing about whether the model's own layer reached the
    # device. Measured: atlas routed 15 matmul layers and the dispatch runtime fell back to the host
    # kernel on all 15, while the tile record read "15 of 15 passed" -- so a run with ZERO layers on the
    # accelerator reported `pass` with `lane: mesh`, which is exactly the CPU-only pass this flag exists
    # to forbid. The two records now live under separate keys and this checks the model one.
    if (capsule.get("semantic") or {}).get("must_accelerate") and model_exec:
        _on = model_exec.get("matmul_layers_on_mesh")
        _fb = int(model_exec.get("matmul_layers_host_fallback") or 0)
        if _on is None:
            result.update(
                status="incomplete",
                failure={
                    "plane": "model",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "detail": "capsule declares must_accelerate but the run recorded no "
                    "per-layer mesh accounting; cannot confirm the model "
                    "reached the accelerator",
                },
            )
            return result
            # AN UNMEASURED LAYER IS NOT A FALLEN-BACK LAYER. When the oracle could not tell us whether the
            # mesh runs a layer, the honest verdict is `incomplete` (NOT_RUN_IS_NOT_PASS), not a compiler
            # failure -- the same rule the tier ladder already applies one level up. Measured: a whole model
            # whose every layer the mesh executes correctly reported "15 of 15 fell back" because the
            # simulator timed out at its per-layer budget, and that number failed must_accelerate.
        _unavail = int(model_exec.get("matmul_layers_oracle_unavailable") or 0)
        if int(_on) == 0 and _unavail and not _fb:
            result.update(
                status="incomplete",
                numeric=_numeric_when_not_accelerated(st, gate, _v, _cos, engine, measured_on="host_lane_unmeasured"),
                failure={
                    "plane": "model",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "detail": f"the mesh oracle could not measure {_unavail} of this "
                    f"model's matmul layer(s) (timed-out or unreachable "
                    f"simulator), so whether they run on the {target} mesh is "
                    f"UNKNOWN -- not a fallback, and not evidence about the "
                    f"backend. Re-run with an oracle that completes.",
                },
            )
            return result
        if int(_on) == 0 and not _fb and not n_tiles:
            # NOTHING RAN AT ALL -- no certified tile and no layer of the model itself. That is not a
            # fallback (nothing was claimed and then missed); it is the absence of a run, and the
            # difference matters: a fallback is evidence about the backend's reach, an empty record is
            # evidence about nothing. Withheld, never failed.
            result.update(
                status="incomplete",
                numeric=_numeric_not_compared(
                    engine,
                    run_where,
                    _v,
                    "nothing was compared ON THE ACCELERATOR: no declared tier ran, so the host-side number "
                    "is not a verdict about this backend. A comparison that did not happen is not a "
                    "passing comparison.",
                ),
                failure={
                    "plane": "model",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "detail": f"neither a certified tile nor a single matmul layer of "
                    f"this model executed on the {target} mesh, so there is no "
                    f"verdict to report either way",
                },
            )
            return result
            # AN ELIGIBLE REGION THE ROUTER DECLINED IS A COMPILER DEFECT, and until now nothing gated on it:
            # the coverage certificate counted `false_fallback` and no verdict ever read the count. This is a
            # claim about the router's own CHOICE -- the region's family and dtype are admitted by the
            # target's manifest and it was left on the host anyway -- so plan-derived evidence is the right
            # evidence for it, unlike the execution questions elsewhere in this file.
            #
            # Gated only for a capsule that DEMANDS acceleration. An interop capsule withholds
            # `must_accelerate` precisely because host work is the behaviour under test, and failing it for
            # declining a region would fail the thing it exists to prove.
        _cert = result.get("coverage_certificate") or {}
        _false_fb = _cert.get("false_fallback_count")
        _census = result.get("placement_census")
        if not isinstance(_census, dict):
            result.update(
                status="incomplete",
                failure={
                    "plane": "model",
                    "category": "PLACEMENT_NOT_MEASURED",
                    "detail": "the compile recorded no whole-module placement census; the routing plan "
                    "cannot prove that every eligible source operation reached the accelerator",
                },
            )
            return result
        _census_status = _census.get("silent_fallbacks_status")
        if _census_status != "offloaded":
            result.update(
                status="incomplete",
                failure={
                    "plane": "model",
                    "category": "PLACEMENT_NOT_MEASURED",
                    "detail": "the whole-module placement census did not establish a decided offload "
                    f"population ({_census_status or 'missing status'}); missing or uncertain "
                    "operations cannot be treated as successful accelerator placement",
                },
            )
            return result
        _silent = _census.get("silent_fallbacks")
        if not isinstance(_silent, list):
            result.update(
                status="incomplete",
                failure={
                    "plane": "model",
                    "category": "PLACEMENT_NOT_MEASURED",
                    "detail": "the whole-module placement census has no silent-fallback ledger",
                },
            )
            return result
        if _silent:
            result.update(
                status="fail",
                numeric=_numeric_when_not_accelerated(st, gate, _v, _cos, engine, measured_on="host_lane_fallback"),
                failure={
                    "plane": "model",
                    "category": "FALLBACK_ON_ELIGIBLE_REGION",
                    "detail": f"{len(_silent)} whole-module region(s) admitted by the independent "
                    "eligibility oracle were left on the host without a hardware refusal; "
                    f"first regions: {_silent[:8]}",
                },
            )
            return result
        _source_verdict = _source_region_execution_verdict(_cert)
        if _source_verdict is not None:
            _status, _category, _detail = _source_verdict
            if _status == "fail":
                result["numeric"] = _numeric_when_not_accelerated(
                    st, gate, _v, _cos, engine, measured_on="host_lane_fallback"
                )
            result.update(
                status=_status,
                failure={"plane": "model", "category": _category, "detail": _detail},
            )
            return result
        if isinstance(_false_fb, int) and _false_fb > 0:
            result.update(
                status="fail",
                numeric=_numeric_when_not_accelerated(st, gate, _v, _cos, engine, measured_on="host_lane_fallback"),
                failure={
                    "plane": "model",
                    "category": "FALLBACK_ON_ELIGIBLE_REGION",
                    "detail": f"{_false_fb} region(s) this target's capability manifest "
                    f"ADMITS were routed to the host anyway; the capsule "
                    f"declares must_accelerate, so declining work the hardware "
                    f"can take is a compiler defect, not a placement choice",
                },
            )
            return result
        if int(_on) == 0 or _fb:
            result.update(
                status="fail",
                numeric=_numeric_when_not_accelerated(st, gate, _v, _cos, engine, measured_on="host_lane_fallback"),
                failure={
                    "plane": "model",
                    "category": "FALLBACK_ON_ELIGIBLE_REGION",
                    "detail": f"capsule declares must_accelerate but only {_on} matmul "
                    f"layer(s) executed on the {target} mesh and {_fb} fell back "
                    f"to the host kernel; the numeric gate therefore measures "
                    f"the HOST, not the accelerator",
                },
            )
            return result

            # A tier that RAN AND FAILED is not a pass, whatever the host-side numeric gate says. The guard below
            # only refuses the case where nothing ran at all, so a failing accelerator tier still reported
            # `status: pass` beside `tiers: {L3: fail}` -- the two halves of the same result contradicting each
            # other, with the flattering half being the one anybody reads.
            # Counter-derived failures gate too. Reading only the per-tile record let a model with three layers
            # fallen back report `pass` beside `tiers: {L2: fail}` -- the two halves of one result contradicting
            # each other, with the flattering half being the one anybody reads.
    _failed_tiers = sorted(
        {t for t, v in exercised.items() if v != "pass"} | {k for k, v in _model_tiers.items() if v.status == "fail"}
    )
    if _failed_tiers:
        result.update(
            status="fail",
            numeric=_numeric_when_not_accelerated(st, gate, _v, _cos, engine, measured_on=run_where),
            failure={
                "plane": "model",
                "category": "FUNCTIONAL_MISMATCH",
                "detail": f"declared oracle tier(s) {_failed_tiers} RAN and did not pass "
                f"(on-mesh execution: {mesh_exec.get('n_passed')} of "
                f"{n_tiles} tile(s) passed, {mesh_exec.get('n_failed')} failed, "
                f"{mesh_exec.get('n_unavailable')} unavailable, "
                f"{mesh_exec.get('n_unsynthesizable')} unsynthesizable); a "
                f"whole-model verdict cannot be a pass over a failing tier",
            },
        )
        if unexercised:
            result["tiers_unexercised"] = _unexercised_note(unexercised, exercised)
        return result

        # A tier the model's own counters resolved COUNTS as exercised. Keying this only on the per-tile
        # record made every run without tile verification report `incomplete`, including one whose layers
        # demonstrably ran on the mesh.
    _counter_ran = {k for k, v in _model_tiers.items() if v.status in ("pass", "fail")}
    if declared and not exercised and not _counter_ran:
        # Same reasoning as the branches above: the arithmetic WAS measured before we got here, so report
        # it. The verdict stays `incomplete` -- no declared tier ran, and a number alone is not a tier.
        result.update(
            status="incomplete",
            numeric=_numeric_not_compared(
                engine,
                run_where,
                _v,
                "nothing was compared ON THE ACCELERATOR: no declared tier ran, so the host-side number "
                "is not a verdict about this backend. A comparison that did not happen is not a "
                "passing comparison.",
            ),
            failure={
                "plane": "model",
                "category": "NOT_RUN_IS_NOT_PASS",
                "detail": f"declares required oracle tiers {declared} and ran NONE of them "
                f"(the functional gate here is the {run_where} reference, not the "
                f"accelerator); a whole-model verdict backed by no declared tier "
                f"is reported UNKNOWN, never a pass",
            },
        )
        return result

    if st == "verified" and gate:
        # A whole-model PASS must carry the strength of the gate that produced it. The numeric gate has
        # a per-element ceiling for exactly the failure aggregates hide (a single element 1209% wrong at
        # cos 0.9999986), but its cosine-only tier bypasses that ceiling for regression outputs which
        # cannot meet it. Both are legitimate passes; they are not the same claim, and "certified" read
        # off `status` alone erases the difference.
        _v = out.get("verify") or {}
        _guarded = _v.get("per_element_guarded")
        # A PASS EARNED ON THE TARGET IS A HARDWARE CLAIM, so it must say which tree produced it. A
        # result attributed to the wrong revision is worse than no result, because it gets cited.
        try:
            from merlin.common import provenance as _prov

            result["provenance"] = _prov.record(extra={"target": target, "run": run_where})
        except Exception as _pe:  # noqa: BLE001 - never block a verdict
            result["provenance"] = {"status": "UNKNOWN", "detail": f"{type(_pe).__name__}: {_pe}"}
        result.update(
            status="pass",
            numeric={
                "status": "pass",
                "engine": engine,
                "gate": _v,
                "gate_tier": _v.get("tier_ok"),
                "per_element_guarded": _guarded,
                "evidence": (
                    "per-element-guarded"
                    if _guarded
                    else "AGGREGATE ONLY — the cosine-only tier carried this verdict; no per-element bound was applied"
                ),
            },
        )
        if unexercised:
            result["tiers_unexercised"] = _unexercised_note(unexercised, exercised)
    elif _quant and st == "run_mismatch" and _cos >= _quant_floor:
        result.update(
            status="pass",
            numeric={
                "status": "pass",
                "engine": engine,
                "gate": _v,
                "quant_tolerance": {"cos": _cos, "floor": _quant_floor, "dtype": dtype},
            },
            note=(
                f"quantized ({dtype}) whole-model output within quant tolerance of the fp32 "
                f"golden (cos {_cos:.4f} >= floor {_quant_floor}); the drop vs fp32 is expected "
                f"quantization error, not a codegen defect."
            ),
        )
    elif st == "not_run":
        result.update(
            status="incomplete",
            failure={
                "plane": "model",
                "category": "NOT_RUN_IS_NOT_PASS",
                "detail": out.get("reason", "whole-model run toolchain unavailable"),
            },
        )
    else:
        result.update(
            status="fail",
            numeric={"status": "fail", "engine": engine, "gate": out.get("verify"), "model_status": st},
            failure={
                "plane": "model",
                "category": "FUNCTIONAL_MISMATCH",
                "detail": f"the whole model did not verify (status={st}, gate_ok={gate})",
            },
        )
    return result


# ---- model -------------------------------------------------------------------------------------
# Capsule evaluation: model.


def _grade_model_capsule_inline(
    capsule: dict,
    *,
    target: str | None = None,
    timeout: int,
    package_dir: str | Path | None = None,
    numeric_policy=_MODEL_NUMERIC_UNSET,
    numeric_policy_identity: dict | None = None,
) -> dict:
    """Grade a whole-model (kind == "model") capsule end to end via the target-aware whole-model flow
    (``compile_model``): route each op across the target's compute units (matmul/systolic -> the mesh, the
    rest -> the vector/scalar lane), compile the functional whole model (the scalar/RVV reference,
    numerically correct across every op), and gate its output vs the model's golden — attaching the per-op
    mesh-routing plan for the target. ``MERLIN_MODEL_GRADE_RUN`` selects ``host`` (default; x86 dispatch
    runtime) or ``spike``. Honestly reports ``incomplete`` when the whole-model toolchain (m2m venv /
    clang-23) is absent — never a silent pass."""
    import os
    from pathlib import Path as _P

    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    model, dtype = attrs.get("model"), attrs.get("compile_dtype", "fp32")
    # Which lane executes the model. The default was the host dispatch runtime unconditionally, so a
    # capsule that DEMANDS acceleration was graded on a lane that cannot provide it and passed anyway.
    # Derive it instead: a capsule whose semantic block asserts must_accelerate runs on the mesh lane,
    # everything else keeps the host lane. The env var still overrides, for a deliberate diagnostic run.
    _sem = capsule.get("semantic") or {}
    # WHICH LANE THE MODEL RUNS ON. must_accelerate implies the mesh lane. So does an interop capsule
    # that REQUIRES on_mesh: it withholds must_accelerate on purpose (host work is the behaviour under
    # test), but running it on the host lane means the mesh never executes and its own lane requirement
    # becomes unverifiable -- which is exactly how the first such capsule passed while executing nothing
    # on the accelerator.
    _req_lanes = [str(x) for x in ((capsule.get("lanes") or {}).get("require") or [])]
    # THE TARGET'S OWN MESH IS THE ORACLE WHENEVER WE HAVE A TARGET. This used to key on the capsule's
    # must_accelerate / required lanes, which left a capsule declaring neither graded on the HOST even
    # when a target was named -- and merlin's x86 dispatch runtime is OUR compiler, so however well it
    # does it is evidence about us, never about the submission. Keying on the target subsumes both
    # declarations: a capsule demanding acceleration on a target with no mesh has nothing to run on
    # either way. The env var still overrides for a deliberate diagnostic run.
    run_where = os.environ.get("MERLIN_MODEL_GRADE_RUN") or ("mesh" if target else "host")
    _ = _req_lanes  # still read below by the lane report
    # MESH VERIFICATION IS THE CAPSULE'S DEMAND, NOT AN OPERATOR'S OPT-IN. It EXECUTES each mesh-routed
    # matmul as a single systolic tile on the target's real mesh oracle (compile_model mesh_verify),
    # which is the only evidence that the matmul layers run ON the mesh rather than that a routing plan
    # was produced.
    #
    # It used to be off unless an env var said otherwise, because the oracle build/run is heavy. The
    # cost is real; the default was still wrong. A capsule that declares `must_accelerate`, or that
    # names `on_mesh` in `lanes.require`, is asking for exactly this evidence -- and without it the tier
    # ladder below has nothing to record, so the capstone reports a verdict backed by the functional
    # lane alone. That is `capstone-graded-tiles-not-the-model` with the accelerator left out entirely.
    # Derive the default from what the capsule demands; the env var still overrides in BOTH directions,
    # for a deliberate diagnostic run. The wall-clock cost is bounded by MERLIN_MODEL_BUDGET_S.
    _demands_mesh = bool(_sem.get("must_accelerate")) or "on_mesh" in _req_lanes
    _mv_env = os.environ.get("MERLIN_MESH_VERIFY", "").strip().lower()
    mesh_verify = (_mv_env in ("1", "true", "yes", "on")) if _mv_env else _demands_mesh
    result: dict = {
        "capsule": capsule["name"],
        "kind": "model",
        "label": capsule.get("label"),
        "operation": {"op": "model", "model": model, "dtype": dtype, "run": run_where, "target": target},
        "contract_version": CONTRACT_VERSION,
    }
    if not model:
        result.update(
            status="incomplete",
            failure={
                "plane": "model",
                "category": "RUNNER_CRASH",
                "detail": "model capsule missing operation.attributes.model",
            },
        )
        return result

        # The target submission and the scalar/vector host lane are TWO compiler packages. The former is
        # threaded below as ``mesh_package``; the latter is frozen experiment infrastructure and MUST come
        # from this target's descriptor. Passing None used ``default_package`` (the currently certified RVV
        # champion), which could differ from the package granted read-only to the agent. The resulting
        # boundary/interop verdict then depended on a compiler the experiment neither declared nor exposed.
    _host_package = None
    _host_identity = None
    _host_experiment = None
    if target:
        try:
            if numeric_policy is _MODEL_NUMERIC_UNSET:
                numeric_policy, numeric_policy_identity = _resolve_model_numeric_policy(target)
            _host_experiment, _host_package, _host_identity = _resolve_model_host_lane(target, dtype)
            result["host_lane"] = _host_identity
        except Exception as e:  # noqa: BLE001 — an unpinned host compiler cannot produce a verdict
            result.update(
                status="incomplete",
                failure={
                    "plane": "model",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "detail": f"frozen host lane unavailable: {type(e).__name__}: {str(e)[:300]}",
                },
            )
            return result
    if numeric_policy is _MODEL_NUMERIC_UNSET:
        numeric_policy = None
    if numeric_policy_identity is not None:
        result["numeric_policy_input"] = numeric_policy_identity
        # The captured linalg (visible grounding) drives the per-op mesh routing when present. Read the name
        # the CAPSULE DECLARES -- every model capsule ships `linalg_mlir: capsule.interface.mlir`, while this
        # looked only for `capsule.linalg.mlir`, so it never found one. `linalg_mlir` was therefore always
        # None, `compile_model`'s `if target and linalg_mlir:` never fired, and the routing plan, the coverage
        # certificate AND the mesh verification were all skipped on every whole-model capsule ever graded --
        # silently, because a skipped block leaves no trace in the result.
    linalg_mlir = None
    linalg_path = None
    cdir = capsule.get("__dir__")
    if cdir:
        for _name in (capsule.get("linalg_mlir"), capsule.get("interface_mlir"), "capsule.linalg.mlir"):
            if not _name:
                continue
            lp = _P(cdir) / str(_name)
            if lp.is_file():
                linalg_path = lp
                linalg_mlir = lp.read_text(encoding="utf-8")
                break
    if target and cdir:
        # The declaration-side composition is derived independently from the capsule interface.  Later
        # the model check compares it with the runtime ledger, so M2's A->H->A seam cannot be credited
        # from a static routing plan and M3's must-accelerate claim cannot pass on synthesized tiles alone.
        from merlin.development.targetgen.grading import boundary as _boundary

        result["boundary_expectation"] = _boundary.profile_capsule(cdir, target).to_dict()
    from merlin.compiler.dialects.lowering.passes import MODEL_BOUNDARY_CAPSTONE

    _requires_boundary_plane = MODEL_BOUNDARY_CAPSTONE in (capsule.get("pass_requirements") or ())
    if _requires_boundary_plane and linalg_path is None:
        result.update(
            status="incomplete",
            failure={
                "plane": "pass_obligations",
                "category": "NOT_RUN_IS_NOT_PASS",
                "detail": "model capsule has no readable declared linalg/interface module; "
                "the production boundary passes cannot be exercised",
            },
        )
        return result
    try:
        # This is the production, in-process boundary plane: outline the REAL model, materialize and
        # verify its dispatch DAG, partition it, and attach the C ABI. It is deliberately separate from
        # the OOT package's four subprocess stages, which run below on each mesh tile. A pass-obligation
        # failure is structural and blocks the capsule before expensive oracle work.
        if _requires_boundary_plane:
            from merlin.compiler.dialects.lowering.passes import run_dialect_plane
            from merlin.compiler.frontend.linalg_mlir import parse_mlir_file

            # WITH the target, so a dispatch is a whole compute group. Without it this plane took the
            # single-op baseline while `compile_cli` formed groups over the same model.
            _plane = run_dialect_plane(parse_mlir_file(linalg_path), n_harts=1, target=target)
            result["pass_obligations"] = _plane.stats

        from merlin.cli.compile_workload import compile_model

        # `dtype` is the capsule's compile_dtype (an RVV compile mode). The capsule ALSO declares the
        # exact datapath format in operation.attributes.dtype -- pass that for routing so a demand is
        # matched against the unit's declared format rather than against a compile-mode token.
        _attrs = (capsule.get("operation") or {}).get("attributes") or {}
        # THE PACKAGE UNDER TEST MUST REACH THE MESH ORACLE. `run_capsule` is handed the submission
        # being graded, but this function never took it and called `compile_model(package=None)`, so a
        # whole-model capsule was compiled and mesh-verified against the DEFAULT path -- not against the
        # submission. Every whole-model number that resulted (numeric verdict, layers-on-mesh accounting,
        # tile certification) was therefore a statement about the reference flow, not about the backend
        # the capsule was supposed to be judging. Measured: mesh verification reported
        # `n_tiles: 0, reason: "no default OOT backend package for target"` while a perfectly good
        # submission sat in the caller's hand.
        #
        # It goes to `mesh_package` (the OOT ACCELERATOR backend that certifies tiles), while `package`
        # is the independently frozen RVV host lane -- two things that a single name would conflate.
        _pkg = str(package_dir) if package_dir else None
        _host_package_arg = str(_host_package) if _host_package is not None else None
        # A model capsule is itself the frozen capture.  Resolving a same-named live recapture here
        # let mutable operator state choose the program/golden after launch (and made capsule-local
        # models impossible to run at all).  Materialize the exact runtime ABI from the declared,
        # digest-bound capsule assets and forbid compile_model's recapture resolver on this path.
        with _model_runtime_bundle(capsule, timeout=timeout) as (
            _capture_bundle,
            _bundle_provenance,
            _verify_bundle_unchanged,
        ):
            result["model_runtime_bundle"] = _bundle_provenance
            out = compile_model(
                model,
                dtype,
                target=target,
                run=run_where,
                verify=True,
                package=_host_package_arg,
                auto_capture=False,
                timeout=timeout,
                linalg_mlir=linalg_mlir,
                mesh_verify=mesh_verify,
                mesh_package=_pkg,
                routing_dtype=_attrs.get("dtype"),
                capture_bundle=_capture_bundle,
                numeric_policy=numeric_policy,
                transform_audit="exact",
            )
            _verify_bundle_unchanged()
    except SystemExit as e:  # toolchain/bundle unavailable — honest skip
        result.update(
            status="incomplete",
            failure={
                "plane": "model",
                "category": "NOT_RUN_IS_NOT_PASS",
                "detail": f"whole-model compile/run unavailable: {str(e)[:300]}",
            },
        )
        return result
    except Exception as e:  # noqa: BLE001
        result.update(
            status="incomplete",
            failure={
                "plane": "model",
                "category": "NOT_RUN_IS_NOT_PASS",
                "detail": f"whole-model grade error: {type(e).__name__}: {str(e)[:300]}",
            },
        )
        return result
        # Detect drift across the compile, not only before it. For a bwrap run this re-verifies the same
        # immutable snapshot aggregate and resolves the package inside its repo payload; it NEVER switches
        # to the live worktree. Direct/non-bwrap grading retains its live-tree pre/post check.
    if _host_experiment is not None and _host_identity is not None:
        try:
            _snapshot_repo_after, _snapshot_after = _verified_model_host_lane_snapshot()
            _snapshot_before = _host_identity.get("run_snapshot")
            if (_snapshot_before is None) != (_snapshot_after is None):
                raise ValueError("model host-lane run snapshot context changed during grading")
            if _snapshot_before is not None:
                if any(
                    _snapshot_after.get(key) != _snapshot_before.get(key)
                    for key in ("root", "content_sha256", "manifest_sha256")
                ):
                    result["host_lane_snapshot_after"] = _snapshot_after
                    raise ValueError("model host-lane run snapshot changed during grading")
                    # RE-RESOLVE THE SAME LANE, not the default one. The check below compares package digests
                    # before and after; resolving a different profile would compare two different packages and
                    # report drift that never happened (or, worse, miss real drift by comparing the wrong pair)
                    # the moment a target declares more than one precision.
            _, _host_after = _host_experiment.resolve_host_lane(root=_snapshot_repo_after, dtype=dtype)
        except Exception as e:  # noqa: BLE001
            result.update(
                status="incomplete",
                failure={
                    "plane": "model",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "detail": f"frozen host lane became unreadable during grading: {type(e).__name__}: {str(e)[:300]}",
                },
            )
            return result
        if _host_after["package_sha256"] != _host_identity["package_sha256"]:
            result.update(
                status="incomplete",
                host_lane_after=_host_after,
                failure={
                    "plane": "model",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "detail": "frozen host-lane package changed during grading "
                    f"({_host_identity['package_sha256'][:12]} -> "
                    f"{_host_after['package_sha256'][:12]}); refusing a "
                    "mixed-compiler verdict",
                },
            )
            return result
    return assess_model_execution(capsule, result, out, target=target, dtype=dtype, run_where=run_where)
