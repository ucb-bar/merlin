"""Synthesize a zephyr_plan (validates against zephyr_plan.schema.yaml).

Required: target, module, devicetree, kconfig, driver_api, samples.

Zephyr is a runtime *backend* for Merlin. The driver implements the Merlin-owned submit/wait/get_metrics
API. Blocking mode comes first. A command-buffer tensor-resident target (the neutral toy_npu example is
the family default) gets the concrete module (residency devicetree props + profiling/RTIO kconfig +
sample); everything else gets a review-flagged skeleton. Keyed on the contract family, not a name.
"""
from __future__ import annotations

from typing import Any

from .. import families as _families
from ..evidence.store import Evidence


def _kconfig_symbol(dialect: str) -> str:
    return "MERLIN_" + dialect.upper()


def _command_buffer_module(target: str, *, is_example: bool) -> dict[str, Any]:
    """The concrete Zephyr module for a command-buffer tensor-resident target — the family default
    seeded from the neutral toy_npu example, parameterized by target name."""
    dialect = target.replace("_", "")
    sym = _kconfig_symbol(dialect)
    return {
        "target": target,
        "module": {"name": "merlin_" + dialect},
        "devicetree": {
            "compatible": f"ucb,{dialect}",
            "properties": [
                "reg", "interrupts", "resident-store-bytes",
                "accumulator-entries", "command-queue-depth",
            ],
        },
        "kconfig": {"symbols": [sym, sym + "_PROFILING", sym + "_RTIO"]},
        "driver_api": {
            "mode": "blocking",
            "functions": ["merlin_submit", "merlin_wait", "merlin_get_metrics"],
        },
        "samples": [f"{dialect}_repeated_rhs_matmul"],
        "confidence": "high" if is_example else "medium",
        "requires_human_review": not is_example,
    }


def _conservative(evidence: Evidence) -> dict[str, Any]:
    target = evidence.target
    sym = _kconfig_symbol(target)
    return {
        "target": target,
        "module": {"name": "merlin_" + target},
        "devicetree": {
            "compatible": f"ucb,{target}",
            "properties": ["reg", "interrupts"],
        },
        "kconfig": {"symbols": [sym]},
        "driver_api": {
            "mode": "blocking",
            "functions": ["merlin_submit", "merlin_wait", "merlin_get_metrics"],
        },
        "samples": [],
        "confidence": "low",
        "requires_human_review": True,
    }


def _is_command_buffer_resident(tc: dict[str, Any]) -> bool:
    feats = set(tc.get("features") or [])
    return "command_buffer" in feats and bool(
        feats & {"resident_packed_tensor", "accumulator_commit"})


def synthesize_zephyr_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Return a zephyr_plan dict for the contract's target (concrete module for a command-buffer
    tensor-resident contract — toy_npu is the family default; else a skeleton). Not keyed on a name."""
    name = target_contract.get("name")
    if _is_command_buffer_resident(target_contract):
        return _command_buffer_module(name, is_example=(name == _families.DEFAULT_EXAMPLE_TARGET))
    return _conservative(evidence)
