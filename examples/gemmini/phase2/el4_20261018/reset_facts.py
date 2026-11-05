"""Read the selected store-controller reset from pinned elaborated FIRRTL."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def accumulator_activation_reset(path: Path, expected_sha256: str, profile_sha256: str) -> dict:
    """Supply the existing provider's reset fact; uninitialized execute state stays unknown."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("selected FIRRTL changed while deriving accumulator activation reset")
    modules = re.findall(
        r"^  module StoreController :[^\n]*\n(.*?)(?=^  (?:module|extmodule) |\Z)", raw.decode(), re.M | re.S
    )
    if len(modules) != 1:
        raise ValueError("selected FIRRTL must contain one StoreController")
    resets = re.findall(
        r"^    regreset activation : UInt<(\d+)>, clock, reset, UInt<\d+>\(0h([0-9a-f]+)\)(?: @\[.*\])?$",
        modules[0],
        re.M,
    )
    if len(resets) != 1 or resets[0] != ("3", "0"):
        raise ValueError("selected store-controller activation must have the reviewed three-bit zero reset")
    return {
        "schema": "merlin.timing_capacity_facts.v1",
        "device_profile_sha256": profile_sha256,
        "elaboration": {"firrtl": expected_sha256},
        "facts": [
            {
                "id": "store-controller-activation-reset",
                "role": "declared_hardware_fact",
                "applies_to": "reset_accumulator_activation",
                "value": 0,
                "unit": "activation_encoding",
                "sources": [
                    {
                        "path": str(path),
                        "sha256": expected_sha256,
                        "selector": "StoreController / activation: UInt<3>, reset=UInt<3>(0h0)",
                    }
                ],
                "derivation": (
                    "The elaborated store activation register resets to zero; execute activation is separate."
                ),
            }
        ],
    }
