"""Capsule evaluation: model assessment."""

from __future__ import annotations

import os

from merlin.evaluation.capsule.lanes import lane_report
from merlin.evaluation.capsule.lanes_evidence import dispatch_boundary_report as dispatch_boundary_report
from merlin.evaluation.capsule.model_evidence import carry_model_evidence
from merlin.evaluation.capsule.model_tiers import (
    _model_tier_map,
    _model_transform_audit_verdict,
    _rtl_tiers_of,
    _source_region_execution_verdict,
)
from merlin.evaluation.capsule.numerics import _numeric_not_compared, _numeric_when_not_accelerated, _unexercised_note
from merlin.evaluation.capsule.tier import TierResult as TierResult


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
