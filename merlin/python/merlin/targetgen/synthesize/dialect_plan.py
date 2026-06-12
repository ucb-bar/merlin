"""Synthesize a dialect_plan (validates against dialect_plan.schema.yaml).

Required fields: target, dialect_name, ops, types, lowering, tests.

toy_npu -> the concrete ToyNPU dialect (res_pack/matmul/commit/evict). Real targets -> a
skeleton with no asserted ops (ops are a human-review decision), flagged for review.
"""
from __future__ import annotations

from typing import Any

from ..evidence.store import Evidence


def _toy_npu() -> dict[str, Any]:
    return {
        "target": "toy_npu",
        "dialect_name": "toynpu",
        "ops": [
            {"name": "res_pack", "summary": "pack + make RHS resident",
             "source_interface": "interface.resident_pack"},
            {"name": "matmul", "summary": "matmul vs resident tensor -> accumulator",
             "source_interface": "interface.matmul"},
            {"name": "commit", "summary": "apply epilogue + commit accumulator",
             "source_interface": "interface.commit"},
            {"name": "evict", "summary": "free resident storage",
             "source_interface": "interface.resident_evict"},
        ],
        "types": [
            {"name": "resident_tensor"},
            {"name": "accumulator"},
        ],
        "lowering": [
            {"from": "interface.resident_pack", "to": "toynpu.res_pack"},
            {"from": "interface.matmul", "to": "toynpu.matmul"},
            {"from": "interface.commit", "to": "toynpu.commit"},
            {"from": "interface.resident_evict", "to": "toynpu.evict"},
        ],
        "tests": [
            {"lit": "res_pack_roundtrip"},
            {"lit": "matmul_commit_epilogue"},
            {"lit": "evict_after_use"},
        ],
        "confidence": "high",
        "requires_human_review": False,
    }


def _conservative(evidence: Evidence) -> dict[str, Any]:
    concepts = sorted(evidence.concept_names())
    return {
        "target": evidence.target,
        "dialect_name": evidence.target,
        "ops": [],
        "types": [],
        "lowering": [],
        "tests": [],
        "detected_concepts": concepts,
        "notes": "Ops/types/lowerings are a human-review decision; do not auto-generate "
                 "dialect ops directly from instruction names.",
        "confidence": "low",
        "requires_human_review": True,
    }


def _curated(target_name: str) -> dict[str, Any] | None:
    """Load a curated in-tree dialect plan (merlin/targets/<t>/contracts/) if any."""
    import yaml

    from ...common.paths import targets_dir

    path = targets_dir() / target_name / "contracts" / "dialect_plan.yaml"
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


CURATED_TARGETS = {"saturn"}


def synthesize_dialect_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Return a dialect_plan dict for the contract's target."""
    name = target_contract.get("name")
    if name == "toy_npu":
        return _toy_npu()
    if name in CURATED_TARGETS:
        curated = _curated(name)
        if curated is not None:
            return curated
    return _conservative(evidence)
