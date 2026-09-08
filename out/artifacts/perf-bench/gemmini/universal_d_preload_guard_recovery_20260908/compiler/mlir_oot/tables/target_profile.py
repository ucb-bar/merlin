"""Selected Gemmini target capabilities, loaded from an operator contract.

The ordinary capsule target retains the historical full-width accumulator
readout.  A deployment can select a stricter generated-hardware contract with
``--target-contract``; scheduling then consumes these facts instead of assuming
that every Gemmini configuration implements both accumulator ports.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from . import rtl_facts as F


@dataclass(frozen=True)
class TargetProfile:
    name: str
    accumulator_read_full_width: bool
    accumulator_read_small_width: bool
    d_preload_from_dram: bool = True
    rvv: bool = False
    source: str = "built-in capsule target"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "accumulator_read_full_width": self.accumulator_read_full_width,
            "accumulator_read_small_width": self.accumulator_read_small_width,
            "d_preload_from_dram": self.d_preload_from_dram,
            "rvv": self.rvv,
            "source": self.source,
        }


DEFAULT = TargetProfile("gemmini", True, True)


def load(path: str | Path | None) -> TargetProfile:
    """Load and validate the selected hardware YAML, or return the legacy target."""
    if path is None:
        return DEFAULT
    source = Path(path)
    data = yaml.safe_load(source.read_text())
    array = data.get("array") or {}
    restrictions = data.get("execution_restrictions") or {}
    soc = data.get("soc") or {}
    rows = int(array.get("mesh_rows", 0))
    cols = int(array.get("mesh_columns", 0))
    if (rows, cols) != (F.DIM, F.DIM):
        raise ValueError(
            f"target contract mesh {rows}x{cols} disagrees with backend {F.DIM}x{F.DIM}")
    if array.get("input_type") not in ("sint8", "int8", "i8"):
        raise ValueError(f"unsupported target input type {array.get('input_type')!r}")
    if array.get("accumulator_type") not in ("sint32", "int32", "i32"):
        raise ValueError(
            f"unsupported target accumulator type {array.get('accumulator_type')!r}")
    full = bool(restrictions.get("accumulator_read_full_width", False))
    small = bool(restrictions.get("accumulator_read_small_width", False))
    if not full and not small:
        raise ValueError("target contract exposes no accumulator readout port")
    d_preload = not bool(restrictions.get("hardcode_d_to_garbage_address", False))
    return TargetProfile(
        str(data.get("name") or source.stem), full, small, d_preload,
        bool(soc.get("rvv", False)), str(source.resolve()))
