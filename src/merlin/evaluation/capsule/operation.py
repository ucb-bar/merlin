"""Capsule evaluation: operation."""

from __future__ import annotations

import datetime as _dt
import json
import traceback as _traceback
from collections.abc import Callable
from pathlib import Path

from merlin.common.jsonio import readable_json
from merlin.evaluation.capsule.adapters import _adapter_readback_policy
from merlin.evaluation.capsule.availability import _resolve_oracle_adapters
from merlin.evaluation.capsule.inputs import bind_program_inputs, emit_program
from merlin.evaluation.capsule.ladder import run_oracle_ladder
from merlin.evaluation.capsule.model_runtime import _grade_model_capsule, model_budget_seconds
from merlin.evaluation.capsule.numerics import _merge_match_policy
from merlin.evaluation.capsule.publication import publish_operation
from merlin.evaluation.capsule.records import ProgramEvidence, ReferenceData
from merlin.evaluation.capsule.reference import check_reference
from merlin.evaluation.capsule.trace import check_trace
from merlin.targetgen.capsule_common import make_run_paths
from merlin.targetgen.package_runtime import BackendDeclined, CertFailure, InfraFailure, Package
from merlin.targetgen.runner_config import _config_for_target


def run_capsule(
    capsule: dict,
    package_dir: str | Path,
    *,
    runs_root: str | Path,
    run_id: str | None = None,
    contract: str | Path | None = None,
    oracle_adapters: dict[str, Callable] | None = None,
    pkg: Package | None = None,
    timeout: int = 600,
    target: str | None = None,
    suite: str | None = None,
    dtype: str = "i8xi8_i32",
    config=None,
    perf_extractor: Callable | None = None,
    no_oracle: bool = False,
    workers: int | None = None,
    adapter_managed_tiers: tuple[str, ...] = (),
    model_execution_selection: dict | None = None,
    checked_source_diagnostic=None,
    model_budget_s: float | None = None,
    _defer_result_publication: bool = False,
) -> dict:
    """Run the original target-selected package/capsule ladder and full gates.

    ``model_execution_selection`` selects exact public facts/timer data, never
    live package/resource authority, protected access or waived numeric tiers.
    ``adapter_managed_tiers`` bypass carried certificates for supplied adapters.
    ``workers`` records actual caller fanout; absent means unstated.
    Missing engines remain unavailable; structural counts supply no RTL timing.
    """
    from merlin.targetgen.provenance import toolchain_shas

    # The effective target comes from the config (authoritative — cfg.target drives the run) when one is
    # supplied, else the explicit ``target`` argument. If NEITHER is given we refuse to run rather than
    # silently defaulting to one target (the OV2 rule: no core path silently operates on a default target).
    eff_target = config.target if config is not None else target
    if eff_target is None:
        raise ValueError("run_capsule requires a target (or a config carrying one); no default target is assumed")
    cfg = config or _config_for_target(eff_target, suite, dtype)
    # L1/oracle output equality uses the capsule's numeric_policy (integer -> exact). A float/SIMT target
    # ALSO declares a force_match_policy (a global float tolerance) so an integer-policy capsule is still
    # graded with tolerance — but the global force must NOT DISCARD the capsule's OWN declared tolerance:
    # a bf16 / low-precision capsule legitimately declares a looser atol/rtol than the global f32 default,
    # and letting the tight global override it makes a CONFORMANT kernel unpassable (RP3). Merge: keep the
    # force policy's compare MODE, take the LOOSER (max) atol/rtol per field.
    policy = _merge_match_policy(cfg.force_match_policy, capsule.get("numeric_policy"))
    name = capsule["name"]
    run_id = run_id or f"{name}"
    # An unrouted grade (oracle_adapters=None) resolves to the TARGET'S OWN endpoint oracle from the
    # contract — never a fixed experiment's simulator pair. `{}` stays honest no-oracle (L0/L1/trace only).
    adapters = oracle_adapters if oracle_adapters is not None else _resolve_oracle_adapters(eff_target)
    if set(adapter_managed_tiers) - set(adapters or {}):
        raise ValueError("adapter-managed tiers require supplied oracle adapters")
    required = set(capsule.get("required_oracle_tiers", []))

    paths = make_run_paths(runs_root, run_id, suite=cfg.suite, target=cfg.target, dtype=cfg.dtype, benchmark=name)

    # Whole-model capsule: graded end to end by compiling the captured model through the merlin
    # whole-model flow (compile_rvv) and gating vs its golden — NOT the per-op tier ladder. Write the
    # same capsule_result.json shape so downstream reporting is uniform.
    if model_execution_selection is not None:
        from merlin.targetgen.public_program_timing import verify_selection

        verify_selection(capsule, model_execution_selection)
    if checked_source_diagnostic is not None:
        from merlin.targetgen.checked_source_model_diagnostic import verify_host_selection

        verify_host_selection(
            checked_source_diagnostic, capsule, target=eff_target, public_selection=model_execution_selection
        )
    from merlin.targetgen.owned_host_slice_execution import selected_source_slice

    if capsule.get("kind") == "model" or selected_source_slice(capsule):
        if _defer_result_publication:
            raise ValueError("operator publication cannot replace whole-model qualification")
        # LEAVE A TRACE BEFORE THE LONG PART. A whole-model grade can legitimately run for hours, and
        # with nothing written until it finishes an operator cannot tell a working run from a wedged
        # one -- which is exactly how a 5h30m stall was read as progress. Written first, so the run
        # directory says what started, when, and under what ceiling.
        _budget = model_budget_seconds() if model_budget_s is None else model_budget_seconds(override=model_budget_s)
        if selected_source_slice(capsule):
            _budget = timeout if _budget is None else min(_budget, timeout)
        if checked_source_diagnostic is not None and (not _budget or _budget <= 0):
            raise ValueError("checked source diagnostic requires the original sealed worker hard deadline")
        paths.run_path.mkdir(parents=True, exist_ok=True)
        (paths.run_path / "model_grade_started.json").write_text(
            json.dumps(
                {
                    "capsule": name,
                    "started_at": _dt.datetime.now(_dt.UTC).isoformat(),
                    "model_budget_s": _budget,
                    "step_timeout_s": timeout,
                    "note": "whole-model capsule; absent capsule_result.json means it is still running or was killed",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        from merlin.targetgen.owned_host_model_selection import owned_model_required

        if (
            owned_model_required(capsule, target=eff_target)
            or model_execution_selection is not None
            or checked_source_diagnostic is not None
        ):
            from merlin.evaluation.capsule.model_context import grade_selected_model

            result = grade_selected_model(
                capsule,
                target=eff_target,
                package_dir=package_dir,
                paths=paths,
                runs_root=runs_root,
                run_id=run_id,
                cfg=cfg,
                contract=contract,
                pkg=pkg,
                adapters=adapters,
                budget_s=_budget,
                timeout=timeout,
                model_execution_selection=model_execution_selection,
                checked_source_diagnostic=checked_source_diagnostic,
            )
        else:
            result = _grade_model_capsule(
                capsule, target=eff_target, timeout=timeout, package_dir=package_dir, budget_s=_budget
            )
        # This model verdict does not independently price command-buffer work.
        # A candidate has one submitted whole-program buffer; the legacy graph
        # has many. Neither absence may be mistaken for a zero-work claim.
        from merlin.perf.work_volume import command_buffer_evidence as _cb_evidence

        volume_scope = (
            "candidate whole-program buffer; volume not independently priced here"
            if bool((capsule.get("semantic") or {}).get("must_accelerate"))
            else "whole-model compilation (no single command buffer)"
        )
        _mw, _ma = _cb_evidence(None, compiler_provenance=volume_scope)
        result.setdefault("work_volume", _mw)
        result.setdefault("command_buffer_artifact", _ma)
        result.setdefault("toolchain_shas", toolchain_shas(eff_target))
        (paths.run_path / "capsule_result.json").write_text(readable_json(result), encoding="utf-8")
        # Persist the ARR coverage certificate as its own durable artifact (not only inside
        # capsule_result.json) so the report/grader can read it back per compilation.
        cert = result.get("coverage_certificate")
        if cert is not None:
            paths.generated.mkdir(parents=True, exist_ok=True)
            (paths.generated / "coverage_certificate.json").write_text(json.dumps(cert, indent=2), encoding="utf-8")
        census = result.get("placement_census")
        if census is not None:
            paths.generated.mkdir(parents=True, exist_ok=True)
            (paths.generated / "placement_census.json").write_text(json.dumps(census, indent=2), encoding="utf-8")
        return result

    from merlin.targetgen.owned_host_leaf_execution import grade_owned_host_leaf, selected_host_leaf

    if selected_host_leaf(capsule):
        if _defer_result_publication:
            raise ValueError("operator publication cannot replace the owned-host route")
        return grade_owned_host_leaf(
            capsule,
            target=eff_target,
            paths=paths,
            cfg=cfg,
            run_id=run_id,
            timeout=timeout,
            no_oracle=no_oracle,
            readback_policy=_adapter_readback_policy(adapters),
            reachable_oracle_tiers=frozenset(adapters),
        )

    evidence = ProgramEvidence()
    inputs = ReferenceData()

    try:
        pkg = emit_program(capsule, package_dir, paths, cfg, contract, timeout, eff_target, pkg, evidence)

        # --- golden + L0/L1 -----------------------------------------------------------------
        # The golden is the INDEPENDENT oracle's answer. An exact source-bound PyTorch integer
        # slice is recomputed on its captured typed operands and cross-checked against host eager.
        # Other integer capsules recompute on their declared synthetic stimulus. For a
        # float capsule that ships an independent golden.yaml (e.g. fp8-e4m3 -> bf16, golden_source
        # specir_refmodel_fp8_bf16), the integer engine cannot reproduce the float datapath, so the golden
        # is READ from golden.yaml — resolved by policy+source, never a target name.
        bind_program_inputs(capsule, paths, eff_target, evidence, inputs)

        check_reference(capsule, paths, policy, required, evidence, inputs)

        # Bound whether or not the trace gate runs: `None` means "nobody parsed these bytes here",
        # which the plan evidence answers by parsing them itself.
        check_trace(capsule, paths, cfg, contract, eff_target, required, no_oracle, evidence)

        # --- oracle tiers -------------------------------------------------------------------
        # Run every tier the config declares (tier_sim ladder) OR an injected adapter provides — so a
        # target whose RTL tier is supplied by an adapter rather than a static tier_sim (e.g. arc L3 with
        # an empty tier_sim) still runs, while a declared ladder is unchanged. The L0/L1 math floor
        # is handled above (reference/simulate), NOT here. Sorted for a stable L2<..<L5 ladder order.
        # program_oracle (the external_backend adapters) raises its OWN OracleUnavailable, a RuntimeError
        # that is a DIFFERENT class from this module's OracleUnavailable(Exception) — unrelated by mro. If
        # we catch only the local one, an honest "oracle unavailable" from the program oracle (missing
        # assembler / model venv / cosim import) falls through to the generic handler and is mislabeled a
        # TOOL_CRASH, so the unavailable->incomplete / not_gradeable_no_oracle path never fires for a self-hosted ISA.
        # Catch BOTH so unavailability is honest for every endpoint.
        run_oracle_ladder(
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
        )

    except BackendDeclined as bd:
        # A STATED REFUSAL IS NOT A WRONG ANSWER. It is still not a pass -- the capsule stays in the
        # denominator, uncertified -- but recording it as a numeric mismatch told the agent its
        # arithmetic was wrong about a program it had never emitted, and no feedback could say which
        # shapes it had declined.
        evidence.status = "declined"
        evidence.declined = bd.to_dict()
        evidence.failure = {"plane": "backend_declined", "category": "DECLINED", "detail": bd.reason}
    except InfraFailure as inf:
        # The harness could not stage this capsule's declared inputs, so NOTHING about the submission was
        # measured. `fail` would be a lie in the agent's direction and `pass` a lie in ours; this is the
        # third thing, and it is excluded from the pass/fail denominators (NOT_MEASURED_STATUSES) while
        # `capsule_grade.grade` refuses to call the run gradeable at all. Caught BEFORE CertFailure
        # because InfraFailure IS one -- the handler order is the whole mechanism, and with the narrower
        # clause absent a staging outage was recorded as 31 of 33 structurally invalid SUBMISSIONS.
        evidence.status = "infrastructure_fault"
        cat = inf.category.value if hasattr(inf.category, "value") else str(inf.category)
        evidence.failure = {"plane": inf.plane, "category": cat, "detail": inf.detail}
    except CertFailure as cf:
        evidence.status = getattr(cf, "status", "fail")  # a NotMeasuredFailure names its not-measured status
        cat = cf.category.value if hasattr(cf.category, "value") else str(cf.category)
        evidence.failure = {"plane": cf.plane, "category": cat, "detail": cf.detail, **getattr(cf, "facts", {})}
    except Exception as e:  # internal harness bug
        evidence.status = "error"
        evidence.failure = {
            "plane": "runner_internal",
            "category": "RUNNER_CRASH",
            "detail": f"{type(e).__name__}: {e}",
            "traceback": _traceback.format_exc(),
        }

    # WHAT WORK THIS RESULT'S CYCLES BOUGHT, said by the result itself. A performance consumer
    # attributes a member's cycles to a compute axis it derives from the emitted command buffer, and
    # it needs both halves: the counted work AND the raw buffer under the digest they agree on, so it
    # can recount the program and refuse the pair when the two disagree. Emitted for EVERY graded
    # capsule, including one whose entrypoints never produced a buffer -- that case carries an
    # explicit UNKNOWN (`exact_macs: null` plus the counter's refusals), never a zero and never an
    # absent key. An absent key reads as "no compute axis applies here"; a zero reads as "this
    # program does no work", which on a perf bench means "infinitely fast". Both are lies.
    return publish_operation(
        capsule,
        package_dir,
        paths,
        cfg,
        contract,
        eff_target,
        name,
        run_id,
        required,
        no_oracle,
        _defer_result_publication,
        evidence,
    )
