"""Synthesize a target_contract (validates against target_contract.schema.yaml).

Registry-selected support providers and checkout examples supply authored contracts. Targets without one
get a conservative skeleton seeded from detected concepts, flagged for review.
"""

from __future__ import annotations

from typing import Any

from ..evidence.store import Evidence

# Required by target_contract.schema.yaml:
#   name, version, capabilities, memory_model, compiler_obligations,
#   hardware_promises, runtime_promises, legality


def _conservative(evidence: Evidence) -> dict[str, Any]:
    """Skeleton contract for a real target, seeded from detected concepts.

    Every field is intentionally coarse and flagged for human review. The detected concepts
    are surfaced under ``capabilities.detected_concepts`` rather than asserted as real ops.
    """
    concepts = sorted(evidence.concept_names())
    return {
        "name": evidence.target,
        "version": 0.0,
        # Empty pending human review.
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
    """Read the selected support contract, never mix it with another provider.

    Unknown targets may have no authored contract and retain conservative synthesis.
    Invalid explicit provider declarations propagate the registry's refusal.
    An explicit external selection is authoritative over a same-name checkout example.
    """
    from ..target_registry import resolve

    selected = resolve(target_name)
    if not selected.contract_path.exists() and not selected.contract_path.is_symlink():
        return None
    contract = selected.load_contract()
    if not isinstance(contract, dict):
        raise ValueError(f"{selected.contract_path}: target contract must be a mapping")
    return contract


def synthesize_target_contract(evidence: Evidence, target_name: str) -> dict[str, Any]:
    """Use selected support data or conservative synthesis.

    Provider absence allows synthesis; invalid explicit provider data does not.
    """
    curated = _curated(target_name)
    if curated is not None:
        return curated
    return _conservative(evidence)
