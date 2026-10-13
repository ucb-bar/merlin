"""Protected original scale-emission resource gate; no numerical qualification."""

from __future__ import annotations

import json
from pathlib import Path

from merlin.llvmlower.device_ir_scalability import DeviceIRBudget
from merlin.targetgen.public_codegen_budget import policy_from_profile, resources_complete
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
from merlin_experiments.phase2.contracts import exact_tree_record, sha256_file


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
    from merlin.targetgen.target_experiment import load_target_experiment

    from . import compile_only, compiler_profile_runtime

    owner = compiler_profile_runtime.functional_service(run_dir)
    policy = selected_policy(owner) if owner is not None else None
    if policy is None:
        return {"status": "not_selected", "required": False, "resource_complete": False}
    result = {"status": "unavailable", "required": policy["required"], "resource_complete": False,
              "selection": policy, "compiler_tool_profile_sha256": owner.profile["sha256"],
              "scope": "original public emission resource acceptance only; no numerical or RTL credit"}
    before = exact_tree_record(Path(candidate))["sha256"]
    try:
        actual = compile_only.run(
            submission=Path(candidate), target_experiment=load_target_experiment(context.descriptor),
            capsules_root=capsules_root, contract=contract, facts_path=facts_path, common_service=owner,
            work_root=Path(run_dir).absolute() / "grading_public_scalability",
            timeout_s=policy["timeout_s"], selected=",".join(policy["capsules"]), scale_probe=True,
            ir_budget=DeviceIRBudget(**policy["device_ir_budget"]),
        )
        owner._verify()
        if (exact_tree_record(Path(candidate))["sha256"] != before
            or actual.get("candidate_sha256") != before
            or actual.get("compiler_tool_profile_sha256") != owner.profile["sha256"]):
            raise ValueError("public scalability source or original profile changed")
        report = Path(run_dir).absolute() / "grading_public_scalability" / "public_compile_only.json"
        if json.loads(report.read_bytes()) != actual:
            raise ValueError("public scalability lost its actual protected observation")
        accepted = resources_complete(actual, policy)
        result.update(status="accepted" if accepted else "incomplete", resource_complete=accepted,
                      candidate_sha256=before, report={"path": str(report), "sha256": sha256_file(report)})
    except Exception as error:  # noqa: BLE001 -- resource refusal cannot become a numerical verdict
        result["reason"] = type(error).__name__
    return result
