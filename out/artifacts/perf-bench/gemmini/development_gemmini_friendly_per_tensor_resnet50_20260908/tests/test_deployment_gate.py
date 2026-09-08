from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from evaluate_deployment_gate import evaluate  # noqa: E402


POLICY = json.loads((ROOT / "validation/deployment_gate.json").read_text())


def passing_receipt() -> dict:
    return {
        "checkpoint": POLICY["model"]["checkpoint"],
        "conv2d_count": 53,
        "linear_count": 1,
        "native_scalar_epilogue_eligible": 54,
        "source_capture": {"complete_graph": True, "opaque": 0},
        "independent_reference": {
            "finite_logits": 1000,
            "top1": 258,
            "vs_fp32": {"cosine": 0.995, "max_abs": 0.1},
            "vs_qdq": {"cosine": 0.999, "max_abs": 0.05},
        },
    }


def passing_evidence() -> dict:
    return {
        "merlin": {"import_complete": True},
        "executorch": {"export_complete": True, "execute_complete": True},
        "same_qdq_graph": True,
        "same_checkpoint": True,
        "same_validation_inputs": True,
        "fresh_heldout_qualification": True,
        "imagenet_validation": {
            "complete": True,
            "top1_accuracy_percent": 80.0,
            "fp32_top1_accuracy_percent": 80.8,
        },
    }


def test_all_predeclared_requirements_promote() -> None:
    result = evaluate(passing_receipt(), POLICY, passing_evidence())
    assert result["status"] == "eligible"
    assert result["promotion_eligible"] is True


def test_low_fp32_logit_fidelity_rejects() -> None:
    receipt = passing_receipt()
    receipt["independent_reference"]["vs_fp32"]["cosine"] = 0.885
    result = evaluate(receipt, POLICY, passing_evidence())
    assert result["status"] == "rejected"
    assert "cosine_vs_fp32" in result["failed_checks"]


def test_missing_executorch_execution_rejects() -> None:
    evidence = passing_evidence()
    evidence["executorch"]["execute_complete"] = False
    result = evaluate(passing_receipt(), POLICY, evidence)
    assert "executorch_execute" in result["failed_checks"]


def test_missing_imagenet_evidence_rejects_fail_closed() -> None:
    evidence = copy.deepcopy(passing_evidence())
    evidence["imagenet_validation"] = {"complete": False}
    result = evaluate(passing_receipt(), POLICY, evidence)
    assert result["promotion_eligible"] is False
    assert {"imagenet_validation_complete", "imagenet_top1", "imagenet_top1_drop"} <= set(
        result["failed_checks"])
