"""Synthesize a zephyr_plan (validates against zephyr_plan.schema.yaml).

Required: target, module, devicetree, kconfig, driver_api, samples.

Zephyr is a runtime *backend* for Merlin. The driver implements the Merlin-owned
submit/wait/get_metrics API. Blocking mode comes first.
"""
from __future__ import annotations

from typing import Any

from ..evidence.store import Evidence


def _kconfig_symbol(target: str) -> str:
    return "MERLIN_" + target.upper()


def _toy_npu() -> dict[str, Any]:
    sym = _kconfig_symbol("toynpu")
    return {
        "target": "toy_npu",
        "module": {"name": "merlin_toynpu"},
        "devicetree": {
            "compatible": "ucb,toynpu",
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
        "samples": ["toynpu_repeated_rhs_matmul"],
        "confidence": "high",
        "requires_human_review": False,
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


def synthesize_zephyr_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Return a zephyr_plan dict for the contract's target."""
    if target_contract.get("name") == "toy_npu":
        return _toy_npu()
    return _conservative(evidence)
