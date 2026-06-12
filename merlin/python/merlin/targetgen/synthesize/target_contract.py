"""Synthesize a target_contract (validates against target_contract.schema.yaml).

toy_npu -> concrete, consistent with merlin/targets/toy_npu/contracts/target_contract.yaml.
Real targets -> conservative skeleton seeded from detected concepts, flagged for review.
"""
from __future__ import annotations

from typing import Any

from ..evidence.store import Evidence

# Required by target_contract.schema.yaml:
#   name, version, capabilities, memory_model, compiler_obligations,
#   hardware_promises, runtime_promises, legality


def _toy_npu() -> dict[str, Any]:
    return {
        "name": "toy_npu",
        "version": 0.1,
        # Spec-mandated abstraction surface for toy_npu (features/ops/types/runtime).
        "features": [
            "resident_packed_tensor",
            "accumulator_commit",
            "command_buffer",
            "metrics",
        ],
        "ops": ["res_pack", "matmul", "commit", "evict"],
        "types": ["resident_tensor", "accumulator"],
        "runtime": {"backends": ["simulator", "zephyr"]},
        "capabilities": {
            "ops": ["matmul", "bias_add", "requant", "relu"],
            "layouts": ["packed_rhs"],
            "resident_storage_bytes": 131072,
            "accumulator_entries": 4096,
        },
        "memory_model": {"resident": True, "accumulators": True},
        "compiler_obligations": [
            "must_prove_rhs_immutable_for_residency",
            "must_respect_resident_storage_bytes",
            "must_commit_accumulator_before_reuse",
        ],
        "hardware_promises": [
            "persistent_resident_tensor",
            "accumulator_commit_epilogue",
        ],
        "runtime_promises": ["persistent_handles", "profiling_regions"],
        "legality": [
            "res_pack requires rhs.mutable == false",
            "sum(resident_tensor.bytes) <= resident_storage_bytes",
        ],
        "confidence": "high",
        "requires_human_review": False,
    }


def _conservative(evidence: Evidence) -> dict[str, Any]:
    """Skeleton contract for a real target, seeded from detected concepts.

    Every field is intentionally coarse and flagged for human review. The detected concepts
    are surfaced under ``capabilities.detected_concepts`` rather than asserted as real ops.
    """
    concepts = sorted(evidence.concept_names())
    return {
        "name": evidence.target,
        "version": 0.0,
        # Shape-consistent with the toy contract, but empty pending human review.
        "features": [],
        "ops": [],
        "types": [],
        "runtime": {"backends": []},
        "capabilities": {
            "ops": [],
            "layouts": [],
            "detected_concepts": concepts,
        },
        "memory_model": {
            "resident": any(c in concepts for c in ("scratchpad", "resident_packed_tensor")),
            "accumulators": "accumulator" in concepts,
        },
        "compiler_obligations": ["TODO: derive from human review of evidence"],
        "hardware_promises": ["TODO: confirm against RTL/docs"],
        "runtime_promises": ["TODO: confirm runtime adapter capabilities"],
        "legality": ["TODO: derive legality predicates from human review"],
        "confidence": "low",
        "requires_human_review": True,
    }


def _curated(target_name: str) -> dict[str, Any] | None:
    """Load a curated in-tree contract (merlin/targets/<t>/contracts/) if one exists.

    Saturn is curated by hand (spike-modeled multicore RVV CPU) the way toy_npu is;
    keeping the synthesizer in lock-step with the committed YAML by construction.
    """
    import yaml

    from ...common.paths import targets_dir

    path = targets_dir() / target_name / "contracts" / "target_contract.yaml"
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# Targets with curated in-tree reference contracts (everything else is conservative).
CURATED_TARGETS = {"saturn"}


def synthesize_target_contract(evidence: Evidence, target_name: str) -> dict[str, Any]:
    """Return a target_contract dict for ``target_name``."""
    if target_name == "toy_npu":
        return _toy_npu()
    if target_name in CURATED_TARGETS:
        curated = _curated(target_name)
        if curated is not None:
            return curated
    return _conservative(evidence)
