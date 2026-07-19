"""The codegen march must pin the target's REAL vector length.

``-march=rv64gcv`` promises only the RVV minimum VLEN (128 bits), so a fixed-width vector we emit is
given double the LMUL a VLEN=256 core needs — doubling vector-register pressure (K-loop spills) and
running each vector instruction at half of VLMAX. These tests pin the contract; the measured effect
is recorded in :func:`merlin.rvvgen.k1.codegen_march`'s docstring.
"""
from merlin.rvvgen import k1


def test_codegen_march_pins_the_boards_vlen():
    m = k1.codegen_march()
    assert m.startswith(k1.K1_MARCH), "must extend the board march, not replace it"
    assert f"_zvl{k1.VLEN}b" in m


def test_codegen_march_is_idempotent():
    once = k1.codegen_march()
    assert k1.codegen_march(once) == once, "re-pinning an already-pinned march must not double it"


def test_codegen_march_honors_explicit_arguments():
    assert k1.codegen_march("rv64gcv", 512) == "rv64gcv_zvl512b"


def test_baseline_march_is_left_alone():
    """The cross-framework baseline arms compile against ``K1_MARCH``; pinning VLEN for OUR codegen
    must not silently re-flag a baseline mid-campaign."""
    assert "zvl" not in k1.K1_MARCH


def test_pinned_vlen_is_verified_against_the_board_not_assumed():
    """Pinning a VLEN the hardware lacks is a MISCOMPILE, not a missed optimization, so a board that
    contradicts the pin must stop the run. The harness banner already reports vlenb on every run, so
    the true value is observable — there is no reason to keep trusting a constant."""
    import pytest

    k1.verify_vlen(k1.VLEN // 8)                    # agreeing board: fine
    with pytest.raises(k1.VlenMismatch):
        k1.verify_vlen(k1.VLEN // 8 // 2)           # half the width: refuse, never measure
    with pytest.raises(k1.VlenMismatch):
        k1.verify_vlen(k1.VLEN // 8 * 2)            # wider board: refuse too (we'd idle lanes)


def test_vlen_is_a_declaration_not_a_baked_constant():
    """VLEN must be overridable per target, so the pin is a stated property of a board rather than an
    assumption compiled into the tool. The general fix remains VL-agnostic codegen (vl_strategy=
    dynamic); this pin is a stopgap for targets whose VLEN we can state AND verify."""
    assert k1.codegen_march(vlen=512).endswith("_zvl512b")
    assert "_zvl256b" not in k1.codegen_march(vlen=512)
