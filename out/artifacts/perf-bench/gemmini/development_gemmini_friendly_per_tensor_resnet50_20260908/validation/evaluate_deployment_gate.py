#!/usr/bin/env python3
"""Fail-closed promotion gate for the target-friendly ResNet-50 capture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _check(rows: list[dict[str, Any]], name: str, observed: Any,
           expected: Any, relation: str = "eq") -> None:
    if relation == "eq":
        passed = observed == expected
    elif relation == "ge":
        passed = observed is not None and observed >= expected
    elif relation == "le":
        passed = observed is not None and observed <= expected
    else:  # pragma: no cover - caller invariant
        raise ValueError(f"unknown relation {relation}")
    rows.append({
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": {"relation": relation, "value": expected},
    })


def evaluate(receipt: dict[str, Any], policy: dict[str, Any],
             evidence: dict[str, Any]) -> dict[str, Any]:
    """Evaluate every declared condition; absent observations never pass."""
    checks: list[dict[str, Any]] = []
    structural = policy["structural"]
    source = receipt.get("source_capture", {})
    _check(checks, "checkpoint", receipt.get("checkpoint"), policy["model"]["checkpoint"])
    _check(checks, "complete_graph", source.get("complete_graph"), structural["complete_graph"])
    _check(checks, "opaque_operations", source.get("opaque"), structural["opaque_operations"])
    for key in ("conv2d_count", "linear_count", "native_scalar_epilogue_eligible"):
        _check(checks, key, receipt.get(key), structural[key])

    logits = receipt.get("independent_reference", {})
    logit_gate = policy["one_image_logits"]
    _check(checks, "finite_logits", logits.get("finite_logits"), logit_gate["finite_logits"])
    _check(checks, "top1", logits.get("top1"), logit_gate["expected_top1"])
    versus_fp32 = logits.get("vs_fp32", {})
    versus_qdq = logits.get("vs_qdq", {})
    _check(checks, "cosine_vs_fp32", versus_fp32.get("cosine"),
           logit_gate["minimum_cosine_vs_fp32"], "ge")
    _check(checks, "max_abs_vs_fp32", versus_fp32.get("max_abs"),
           logit_gate["maximum_absolute_error_vs_fp32"], "le")
    _check(checks, "cosine_vs_portable_qdq", versus_qdq.get("cosine"),
           logit_gate["minimum_cosine_vs_portable_qdq"], "ge")
    _check(checks, "max_abs_vs_portable_qdq", versus_qdq.get("max_abs"),
           logit_gate["maximum_absolute_error_vs_portable_qdq"], "le")

    fair = policy["fairness"]
    merlin = evidence.get("merlin", {})
    executorch = evidence.get("executorch", {})
    if fair["merlin_import_required"]:
        _check(checks, "merlin_import", merlin.get("import_complete"), True)
    if fair["executorch_export_and_execute_required"]:
        _check(checks, "executorch_export", executorch.get("export_complete"), True)
        _check(checks, "executorch_execute", executorch.get("execute_complete"), True)
    for policy_key, evidence_key in (
        ("same_qdq_graph_required", "same_qdq_graph"),
        ("same_checkpoint_required", "same_checkpoint"),
        ("same_validation_inputs_required", "same_validation_inputs"),
        ("fresh_heldout_qualification_required", "fresh_heldout_qualification"),
    ):
        if fair[policy_key]:
            _check(checks, evidence_key, evidence.get(evidence_key), True)

    imagenet_gate = policy["imagenet_validation"]
    imagenet = evidence.get("imagenet_validation", {})
    if imagenet_gate["required"]:
        _check(checks, "imagenet_validation_complete", imagenet.get("complete"), True)
        _check(checks, "imagenet_top1", imagenet.get("top1_accuracy_percent"),
               imagenet_gate["minimum_top1_accuracy_percent"], "ge")
        candidate = imagenet.get("top1_accuracy_percent")
        reference = imagenet.get("fp32_top1_accuracy_percent")
        drop = None if candidate is None or reference is None else reference - candidate
        _check(checks, "imagenet_top1_drop", drop,
               imagenet_gate["maximum_absolute_drop_from_fp32_percent"], "le")

    failures = [row for row in checks if not row["passed"]]
    return {
        "schema": "gemmini_friendly_resnet50_deployment_gate_assessment_v1",
        "status": "eligible" if not failures else "rejected",
        "promotion_eligible": not failures,
        "check_count": len(checks),
        "passed_count": len(checks) - len(failures),
        "failed_count": len(failures),
        "checks": checks,
        "failed_checks": [row["name"] for row in failures],
        "interpretation": (
            "eligible for paired Merlin/ExecuTorch benchmarking"
            if not failures else
            "candidate only; must not replace the canonical numerical contract or support a performance claim"
        ),
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--policy", type=Path, default=here / "deployment_gate.json")
    parser.add_argument("--backend-evidence", type=Path, default=here / "backend_evidence.json")
    parser.add_argument("--out", type=Path, default=here / "deployment_gate_assessment.json")
    args = parser.parse_args()
    result = evaluate(
        json.loads(args.receipt.read_text()),
        json.loads(args.policy.read_text()),
        json.loads(args.backend_evidence.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "promotion_eligible", "passed_count", "failed_count", "failed_checks")},
        sort_keys=True))
    return 0 if result["promotion_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
