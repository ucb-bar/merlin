"""SpecRefSource: a capsule sourced from a specir verification spec. The spec supplies the PROGRAM (its own
command buffer), the bit-exact golden its refmodel computes over deterministic operands, and the declared
coverage. Program emission exists for the RoCC/command-buffer families (e.g. gemmini); a gen without an
emitter fails closed (never a faked program). The spec capture is skipped when specir is absent.

Target-agnostic: the gen is a parameter (the ``spec_ref`` a profile entry declares), never a code literal
in library scope — this test file is a legitimate edge that names gens as data under test."""
from __future__ import annotations

import numpy as np
import pytest

from merlin.targetgen import capsule_source as CSrc

_SPEC = CSrc.SpecRefSource()
_needs_spec = pytest.mark.skipif(not _SPEC.available(), reason="specir unavailable; set SPECIR_ROOT")


def test_parse_spec_ref():
    assert CSrc._parse_spec_ref("gemmini:op.matmul") == ("gemmini", "op.matmul")
    with pytest.raises(ValueError):
        CSrc._parse_spec_ref("nocolon")


@_needs_spec
def test_spec_capture_gemmini_program_and_golden():
    """The gemmini spec yields a command-buffer program + a bit-exact golden that equals an independent
    int matmul over the SAME deterministic operands (self-consistent, no live oracle needed)."""
    art = _SPEC.capture("gemmini:op.matmul", workload=(16, 16, 16), tile_dim=16)
    assert art.gen == "gemmini" and art.command_buffer["target"] == "gemmini"
    assert set(art.operands) == {"lhs", "weight"} and "out" in art.golden and art.compare == "exact_int"
    assert art.instructions and art.opcode_backing  # RoCC issue sequence + spec-RoCC backing present
    A0 = np.array(art.operands["lhs"], dtype=np.int64)
    W = np.array(art.operands["weight"], dtype=np.int64)
    assert np.array_equal(np.array(art.golden["out"], dtype=np.int64), A0 @ W)


@_needs_spec
def test_spec_capture_fails_closed_unknown_gen():
    with pytest.raises(CSrc.SpecProgramUnavailable):
        _SPEC.capture("not_a_gen:op.matmul")


@_needs_spec
@pytest.mark.parametrize("spec_ref,workload,td,contraction", [
    ("atlas-npu:op.matmul_mxu0", (32, 32, 32), 32, True),   # fp8 MXU command sequence
    ("radiance:op.matmul", (16, 16, 16), 16, True),          # SIMT warp schedule
])
def test_spec_capture_float_families(spec_ref, workload, td, contraction):
    """Atlas (MXU) and radiance (SIMT warp) programs: decoded role-keyed operands + a float golden that
    equals a matmul over those operands (self-consistent, no live oracle needed)."""
    art = _SPEC.capture(spec_ref, workload=workload, tile_dim=td)
    assert art.compare == "tolerance_float"
    assert set(art.operands) == {"lhs", "weight"} and "out" in art.golden
    A = np.array(art.operands["lhs"], dtype=np.float64)
    W = np.array(art.operands["weight"], dtype=np.float64)
    G = np.array(art.golden["out"], dtype=np.float64)
    rel = np.abs(G - A @ W).max() / (np.abs(A @ W).max() + 1e-9)
    assert rel < 0.25          # golden reproduces the spec matmul over the decoded operands


@_needs_spec
def test_spec_capture_fails_closed_no_program():
    """A gen no emitter authors a matmul program for (a vector unit) fails closed, never a faked program."""
    with pytest.raises(CSrc.SpecProgramUnavailable):
        _SPEC.capture("saturn:op.matmul")
