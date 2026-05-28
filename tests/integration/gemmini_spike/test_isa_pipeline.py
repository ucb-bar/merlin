"""Verifies the dialect-level lowering: the gemmini.tile_matmul ISA-tier
fixture lowers cleanly into a sequence of gemmini.intr.* RoCC intrinsic
ops via merlin-gemmini-legalize-for-llvm-export.

This is a lit-style smoke test that exercises only the dialect-level
pass (input is hand-authored memref-domain ISA-tier IR), not the full
IREE compile pipeline. The end-to-end pipeline test lives in
test_matmul_8x8x8.py / test_matmul_64x64x64.py.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
# Prefer the in-tree build of iree-opt (no `install` step required).
_IREE_OPT_CANDIDATES = [
    _REPO_ROOT / "build" / "host-merlin-debug" / "tools" / "iree-opt",
    _REPO_ROOT / "build" / "host-merlin-debug" / "install" / "bin" / "iree-opt",
]
_IREE_OPT = next(
    (p for p in _IREE_OPT_CANDIDATES if p.exists()),
    _IREE_OPT_CANDIDATES[0],
)
_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.mark.skipif(
    not _IREE_OPT.exists(),
    reason=f"iree-opt not built at {_IREE_OPT}; run ./merlin build --profile gemmini",
)
def test_tile_matmul_isa_lowers_to_intr_ops():
    """Drive the legalize-for-llvm-export pass and check we see the
    expected RoCC intrinsic ops in the output."""
    fixture = _FIXTURES / "tile_matmul_isa_int8.mlir"
    assert fixture.exists()

    res = subprocess.run(
        [
            str(_IREE_OPT),
            str(fixture),
            "--iree-plugin=gemmini",
            "--pass-pipeline=builtin.module(func.func(" "merlin-gemmini-legalize-for-llvm-export,canonicalize,cse))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"iree-opt failed:\n{res.stderr}"
    out = res.stdout
    # Expected ops in the lowered output.
    expected_intr_ops = [
        "gemmini.intr.config",
        "gemmini.intr.mvin",
        "gemmini.intr.preload",
        "gemmini.intr.compute.preloaded",
        "gemmini.intr.mvout",
        "gemmini.intr.flush",
    ]
    for op in expected_intr_ops:
        assert op in out, f"Expected {op} in lowered output, got:\n{out}"
