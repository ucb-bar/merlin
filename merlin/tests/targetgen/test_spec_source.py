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
    assert set(art.operands) == {"W", "A0"} and "Y0" in art.golden
    assert art.instructions and art.opcode_backing  # RoCC issue sequence + spec-RoCC backing present
    A0 = np.array(art.operands["A0"], dtype=np.int64)
    W = np.array(art.operands["W"], dtype=np.int64)
    assert np.array_equal(np.array(art.golden["Y0"], dtype=np.int64), A0 @ W)


@_needs_spec
def test_spec_capture_fails_closed_unknown_gen():
    with pytest.raises(CSrc.SpecProgramUnavailable):
        _SPEC.capture("not_a_gen:op.matmul")


@_needs_spec
def test_spec_capture_fails_closed_without_emitter():
    """A gen with no command-buffer emitter (radiance is not modeled by the merlin cb path) fails closed
    rather than fabricating a program."""
    with pytest.raises(CSrc.SpecProgramUnavailable):
        _SPEC.capture("radiance:op.matmul")
