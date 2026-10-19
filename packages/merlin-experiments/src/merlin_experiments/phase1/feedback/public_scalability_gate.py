"""Protected original scale-emission resource gate; no numerical qualification."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from merlin.llvmlower.device_ir_scalability import DeviceIRBudget
from merlin.targetgen.public_codegen_budget import policy_from_profile, resources_complete
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
from merlin_experiments.phase2.contracts import exact_tree_record, sha256_file


class PublicResourceGateError(RuntimeError):
    """The selected resource prerequisite failed before numerical grading."""


def _exception_record(error):
    """Protected diagnostics only; public feedback uses a fixed projection."""
    return {
        "type": type(error).__name__,
        "module": type(error).__module__,
        "message": str(error)[:16384],
        "traceback": "".join(traceback.format_exception(error))[-65536:],
    }


def require_complete(result):
    """Stop before expensive grading, without changing the selected policy."""
    if result.get("required") is False:
        return
    if (result.get("required") is True and result.get("resource_complete") is True
        and result.get("status") == "accepted"):
        return
    raise PublicResourceGateError(
        "Required public codegen resource gate incomplete; numerical grading was not started. "
        "Inspect public_codegen_resources.json and the selected public --compile-only --scale-probe results."
    )


def selected_policy(owner):
    if type(owner) is not CommonTransformService:
        raise ValueError("public scalability policy requires its exact original profile owner")
    owner._verify()
    return policy_from_profile(owner.profile)


def selected_budget(owner):
    policy = selected_policy(owner)
    return DeviceIRBudget(**policy["device_ir_budget"]) if policy is not None else None


def run_gate(*, run_dir, candidate, context, capsules_root, contract, facts_path):
    """Run the complete selected public ladder under its archived profile owner."""
    from merlin_experiments.corpus.preparation import private_json

    from . import compiler_profile_runtime

    try:
        owner = compiler_profile_runtime.functional_service(run_dir)
        result, _ = _observe(
            owner=owner, candidate=candidate, context=context, capsules_root=capsules_root,
            contract=contract, facts_path=facts_path,
            work_root=Path(run_dir).absolute() / "grading_public_scalability",
        )
    except Exception as error:  # noqa: BLE001 -- preparation failure cannot skip a selected prerequisite
        result = {"status": "unavailable", "required": True, "resource_complete": False,
                  "reason": type(error).__name__, "error": _exception_record(error)}
    private_json(Path(run_dir) / "public_codegen_resources.json", result)
    return result


def _observe(*, owner, candidate, context, capsules_root, contract, facts_path, work_root):
    from merlin.targetgen.target_experiment import load_target_experiment

    from . import compile_only

    policy = selected_policy(owner) if owner is not None else None
    if policy is None:
        return {"status": "not_selected", "required": False, "resource_complete": False}, None
    result = {"status": "unavailable", "required": policy["required"], "resource_complete": False,
              "selection": policy, "compiler_tool_profile_sha256": owner.profile["sha256"],
              "scope": "original public emission resource acceptance only; no numerical or RTL credit"}
    before = exact_tree_record(Path(candidate))["sha256"]
    result["candidate_sha256"] = before
    actual = None
    try:
        actual = compile_only.run(
            submission=Path(candidate), target_experiment=load_target_experiment(context.descriptor),
            capsules_root=capsules_root, contract=contract, facts_path=facts_path, common_service=owner,
            work_root=work_root,
            timeout_s=policy["timeout_s"], selected=",".join(policy["capsules"]), scale_probe=True,
            ir_budget=DeviceIRBudget(**policy["device_ir_budget"]),
        )
        owner._verify()
        if (exact_tree_record(Path(candidate))["sha256"] != before
            or actual.get("candidate_sha256") != before
            or actual.get("compiler_tool_profile_sha256") != owner.profile["sha256"]):
            raise ValueError("public scalability source or original profile changed")
        report = work_root / "public_compile_only.json"
        if json.loads(report.read_bytes()) != actual:
            raise ValueError("public scalability lost its actual protected observation")
        accepted = resources_complete(actual, policy)
        result.update(status="accepted" if accepted else "incomplete", resource_complete=accepted,
                      candidate_sha256=before, report={"path": str(report), "sha256": sha256_file(report)})
    except Exception as error:  # noqa: BLE001 -- resource refusal cannot become a numerical verdict
        result["reason"] = type(error).__name__
        result["error"] = _exception_record(error)
        actual = None  # Failed custody cannot supply attributed public samples.
    return result, actual


def observe_authoring(*, owner, candidate, context, capsules_root, contract, facts_path, work_root):
    """Run only an original selected public ladder and return a path-free projection.

    This is independent of numerical capsule grading. The caller publishes it
    before expensive simulations, without exposing protected owner, tool or
    artifact paths. A missing original selection performs no compiler work.
    """
    result, actual = _observe(
        owner=owner, candidate=candidate, context=context, capsules_root=capsules_root,
        contract=contract, facts_path=facts_path, work_root=work_root,
    )
    if result["status"] == "not_selected":
        return None
    public = {
        "schema": "merlin.phase1.public_codegen_resources.v1",
        "status": result["status"], "required": result["required"],
        "resource_complete": result["resource_complete"],
        "selection": result["selection"],
        "compiler_tool_profile_sha256": result["compiler_tool_profile_sha256"],
        "candidate_sha256": result["candidate_sha256"],
        "samples": [],
        "scope": "selected original public emission resources only; no numerical, offload or execution credit",
        "next_action": "Inspect the named public member with --compile-only --scale-probe before expensive simulation.",
    }
    if result.get("reason"):
        public["reason"] = "original public scale observation unavailable"
    if actual is not None:
        for row in actual["cases"]:
            sample = {name: row.get(name) for name in ("member", "scale", "variant", "lowering_status", "wall_seconds")}
            growth = row.get("code_growth") or {}
            sample["code_growth"] = {name: growth[name] for name in (
                "bytes", "function_code_bytes", "lines", "inline_asm_sites", "call_sites", "functions", "basic_blocks"
            ) if name in growth}
            sample["resource_budget"] = row.get("resource_budget")
            if row.get("reason"):
                sample["reason"] = "emission unavailable or original aggregate deadline exhausted"
            public["samples"].append(sample)
    return public
