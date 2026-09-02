"""The movement harness allocates its destination from the DECLARED OUTPUT dtype.

Regression for a harness defect, not a submission defect: ``_movement_harness_c`` pinned the destination
buffer to the target's operand type (``elem_t``, 1 byte) regardless of the capsule's declared
``output_dtype``. Movement is a CONTAINER WIDENING (operand dtype in, accumulate dtype out), and two
shipped capsules (``FT00_movement_tail_15x15``, ``FT01_movement_tail_17x15``) declare an i32 output — so
a CORRECT 4-byte store ran off the end of a 1-byte-per-element buffer, overrunning .bss by ~700 bytes and
trapping AFTER the kernel printed DONE.

Text-only: renders the harness C and inspects the declaration. No compiler, no oracle.
"""
from __future__ import annotations

import pytest

from merlin.runtime.backends import base as _bk

gem = _bk.get_backend("gemmini")


def _movement_cb(m: int, n: int, out_dtype: str):
    return {"abi_version": "0.1", "target": "gemmini",
            "tensors": {"X": {"shape": [m, n], "dtype": "i8", "role": "input"},
                        "Y0": {"shape": [m, n], "dtype": out_dtype, "role": "output"}},
            "commands": [{"opcode": "VECTOR_MAP",
                          "operands": {"lhs": "X", "rhs": "X", "dst": "Y0"},
                          "attributes": {"combine": "identity", "activation": [],
                                         "output_dtype": out_dtype}}]}


def _decl(c: str, name: str) -> str:
    lines = [ln.strip() for ln in c.splitlines()
             if ln.strip().startswith("static") and f"T_{name}[" in ln and "=" not in ln]
    assert len(lines) == 1, f"expected one declaration of T_{name}, got {lines}"
    return lines[0]


@pytest.mark.parametrize("m,n,cells", [(15, 15, 256), (17, 15, 512)])
def test_i32_output_gets_a_four_byte_destination(m, n, cells):
    """The two shipped tail capsules: a 4-byte element type, so the buffer is 4x the cell count."""
    c = gem.render_harness(_movement_cb(m, n, "i32"), target="gemmini")
    d = _decl(c, "Y0")
    assert "int32_t" in d and "elem_t" not in d
    assert f"[{cells}]" in d


def test_i8_output_still_gets_the_operand_type():
    """The unchanged case: an i8 readout lands in the operand type, byte-identical to before."""
    d = _decl(gem.render_harness(_movement_cb(16, 16, "i8"), target="gemmini"), "Y0")
    assert "elem_t" in d and "int32_t" not in d


def test_source_buffer_stays_the_operand_type():
    """Only the DESTINATION widens — the embedded source is still the operand dtype."""
    c = gem.render_harness(_movement_cb(15, 15, "i32"), target="gemmini")
    src = [ln for ln in c.splitlines() if "T_X[" in ln and "static" in ln]
    assert len(src) == 1 and "elem_t" in src[0]


def test_unsized_output_dtype_is_refused_not_guessed():
    """FAIL CLOSED. An output dtype this harness has no buffer width for is a REFUSAL — never a buffer
    quietly allocated at a guessed width, which is the defect being fixed."""
    cb = _movement_cb(16, 16, "i64")
    with pytest.raises(Exception, match="no.*buffer width|output dtype"):
        gem.render_harness(cb, target="gemmini")


def test_output_dtype_falls_back_to_the_declared_tensor():
    """A buffer whose command omits ``output_dtype`` is still sized from the DECLARED output tensor,
    not from the operand type."""
    cb = _movement_cb(15, 15, "i32")
    cb["commands"][0]["attributes"].pop("output_dtype")
    assert "int32_t" in _decl(gem.render_harness(cb, target="gemmini"), "Y0")
