"""Synthesize a runtime_adapter_plan (validates against runtime_adapter_plan.schema.yaml).

Required: target, implements, command_encoding, queues, events, handles, metrics.

A target only ever *implements* the Merlin runtime ABI; it never defines its own runtime model. A
COMMAND-BUFFER tensor-resident target (declared by its contract features — the neutral toy_npu example
is the family default) realizes the concrete command-stream adapter; everything else gets a
review-flagged skeleton. Selection is keyed on the contract family, never a target name.
"""
from __future__ import annotations

from typing import Any

from .. import families as _families
from ..evidence.store import Evidence

RUNTIME_ABI_VERSION = "0.1"


def _command_stream_adapter(target: str, *, is_example: bool) -> dict[str, Any]:
    """The concrete command-stream adapter for a command-buffer tensor-resident target — the family
    default seeded from the neutral toy_npu example, parameterized by target name."""
    dialect = target.replace("_", "")
    return {
        "target": target,
        "implements": {"runtime_abi_version": RUNTIME_ABI_VERSION},
        "command_encoding": {
            "format": f"{dialect}_command_stream",
            "supports_batching": True,
            "supports_async": False,
        },
        "queues": [{"name": "compute"}, {"name": "dma"}],
        "events": {"completion": "polling", "interrupts": "optional"},
        "handles": {
            "resident_tensor": "resident_store_slot",
            "accumulator": "accumulator_slot",
        },
        "metrics": {
            "maps_to_common": {
                "bytes_moved": ["pack_bytes", "commit_bytes"],
                "command_count": ["command_count"],
                "pack_count": ["pack_count"],
                "resident_hits": ["resident_hits"],
                "evictions": ["evictions"],
                "accumulator_commits": ["accumulator_commits"],
            },
            "target_specific": [],
        },
        # The neutral example is the well-understood reference; a real family-resolved target is flagged.
        "confidence": "high" if is_example else "medium",
        "requires_human_review": not is_example,
    }


def _is_command_buffer_resident(tc: dict[str, Any]) -> bool:
    """A contract that implements the Merlin command-buffer tensor-resident runtime (packs a resident
    weight, commits an accumulator, driven by a command buffer). Detected from the contract, not a name."""
    feats = set(tc.get("features") or [])
    return "command_buffer" in feats and bool(
        feats & {"resident_packed_tensor", "accumulator_commit"})


def _conservative(evidence: Evidence) -> dict[str, Any]:
    return {
        "target": evidence.target,
        "implements": {"runtime_abi_version": RUNTIME_ABI_VERSION},
        "command_encoding": {
            "format": "TODO_human_review",
            "supports_batching": False,
            "supports_async": False,
        },
        "queues": [],
        "events": {"completion": "TODO", "interrupts": "TODO"},
        "handles": {},
        "metrics": {"maps_to_common": {}, "target_specific": []},
        "detected_concepts": sorted(evidence.concept_names()),
        "confidence": "low",
        "requires_human_review": True,
    }


def synthesize_runtime_adapter_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Return a runtime_adapter_plan dict for the contract's target: a command-buffer tensor-resident
    contract (the neutral toy_npu example is the family default) realizes the concrete command-stream
    adapter; everything else gets the review-flagged skeleton. Keyed on the contract family, not a name."""
    name = target_contract.get("name")
    if _is_command_buffer_resident(target_contract):
        return _command_stream_adapter(name, is_example=(name == _families.DEFAULT_EXAMPLE_TARGET))
    return _conservative(evidence)
