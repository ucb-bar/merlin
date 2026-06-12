"""Synthesize a runtime_adapter_plan (validates against runtime_adapter_plan.schema.yaml).

Required: target, implements, command_encoding, queues, events, handles, metrics.

A target only ever *implements* the Merlin runtime ABI; it never defines its own runtime
model. toy_npu -> concrete adapter. Real targets -> skeleton flagged for review.
"""
from __future__ import annotations

from typing import Any

from ..evidence.store import Evidence

RUNTIME_ABI_VERSION = "0.1"


def _toy_npu() -> dict[str, Any]:
    return {
        "target": "toy_npu",
        "implements": {"runtime_abi_version": RUNTIME_ABI_VERSION},
        "command_encoding": {
            "format": "toynpu_command_stream",
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
        "confidence": "high",
        "requires_human_review": False,
    }


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


def _saturn() -> dict[str, Any]:
    """Curated saturn adapter plan: the spike-modeled multicore RVV CPU backend."""
    return {
        "target": "saturn",
        "implements": {"runtime_abi_version": RUNTIME_ABI_VERSION},
        "command_encoding": {
            # Merlin opcodes are executed directly: matmuls run the hand-written
            # RVV kernel; RES_PACK/EVICT are counted layout/budget events on a CPU.
            "format": "merlin_baremetal_driver",
            "supports_batching": True,
            "supports_async": False,
        },
        "queues": [{"name": "compute"}],
        "events": {"completion": "barrier", "interrupts": "none"},
        "handles": {"packed_tensor": "memory_resident_buffer",
                    "accumulator": "vector_register_tile"},
        "backends": ["simulator", "baremetal", "vcs", "zephyr"],
        "metrics": {
            "maps_to_common": {
                "cycles": ["mcycle_delta_hart0"],
                "command_count": ["command_count"],
                "pack_count": ["pack_count"],
                "resident_hits": ["resident_hits"],
                "evictions": ["evictions"],
                "accumulator_commits": ["accumulator_commits"],
            },
            "target_specific": ["harts"],
        },
        "confidence": "medium",
        "requires_human_review": False,
    }


def synthesize_runtime_adapter_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Return a runtime_adapter_plan dict for the contract's target."""
    name = target_contract.get("name")
    if name == "toy_npu":
        return _toy_npu()
    if name == "saturn":
        return _saturn()
    return _conservative(evidence)
