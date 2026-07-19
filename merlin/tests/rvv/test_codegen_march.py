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
