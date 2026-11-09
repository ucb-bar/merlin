import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.verify.lean_checker import (
    accepted_pipeline_receipt,
    accepted_receipt,
    verify_pipeline_change,
    verify_scalar_transform,
)
from merlin.verify.lean_contract import LeanToolchain, coverage_policy_record, digest

BEFORE = "builtin.module {\n  func.func @kernel(%x: i8) -> i8 {\n    %z = arith.constant 0 : i8\n    %r = arith.addi %x, %z : i8\n    func.return %r : i8\n  }\n}"
AFTER = "builtin.module { func.func @kernel(%x: i8) -> i8 { func.return %x : i8 } }"


@pytest.fixture
def real_toolchain(tmp_path):
    manifest = os.environ.get("MERLIN_LEAN_TOOLCHAIN_MANIFEST")
    if not manifest:
        pytest.skip("explicit real Lean dependency selection not installed")
    return LeanToolchain.from_manifest(Path(manifest), work_root=tmp_path)


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "AFTER",
    "BEFORE",
    "LeanToolchain",
    "Path",
    "accepted_pipeline_receipt",
    "accepted_receipt",
    "coverage_policy_record",
    "digest",
    "os",
    "pytest",
    "real_toolchain",
    "replace",
    "verify_pipeline_change",
    "verify_scalar_transform",
]
