from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.compiler.transforms.analyses import blocks_in
from merlin.compiler.transforms.certificates import ir_text, parse_ir
from merlin.verification.lean import host_checks as CHECK
from merlin.verification.lean.contract import LeanToolchain, LeanTransformReceipt, coverage_policy_record
from merlin.verification.lean.state_effects import EffectScheduleContract


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
