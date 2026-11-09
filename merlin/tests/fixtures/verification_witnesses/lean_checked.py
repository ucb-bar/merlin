from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.planning import blocks_in
from merlin.verify import lean_checked as CHECK
from merlin.verify.lean_contract import LeanToolchain, LeanTransformReceipt, coverage_policy_record
from merlin.verify.lean_state_effects import EffectScheduleContract


def _contract():
    return EffectScheduleContract(standard_memory_sequential=True, standard_allocations_successful=True, index_bits=64)


@pytest.fixture
def selected_toolchain(tmp_path):
    selection = os.environ.get("MERLIN_LEAN_TOOLCHAIN")
    if not selection:
        pytest.skip("explicit independently selected Lean toolchain unavailable")
    return LeanToolchain.from_manifest(Path(selection), work_root=tmp_path)


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "CHECK",
    "EffectScheduleContract",
    "LeanToolchain",
    "LeanTransformReceipt",
    "Path",
    "_contract",
    "blocks_in",
    "coverage_policy_record",
    "ir_text",
    "os",
    "parse_ir",
    "pytest",
    "replace",
    "selected_toolchain",
]
