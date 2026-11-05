"""kernel-test: the shape sweep, the operand generator the image and the host share, and the verdict.

Nothing here runs a simulator: the console is synthetic, and the generator is checked against a pure-Python
statement of the same arithmetic the C driver spells.
"""

from __future__ import annotations

from merlin.kernels import matrix_kernel_test as KT
from merlin.kernels import matrix_kernels as MK


def test_the_sweep_covers_edges_and_k_tails_and_honours_constraints():
    shapes, skipped = KT.sweep(32, MK.Constraints(), quick=True)
    dims = {(s.m, s.n, s.k) for s in shapes}
    assert {(31, 32, 32), (33, 32, 32), (32, 65, 32), (32, 32, 1), (32, 32, 33), (1, 1, 1)} <= dims
    assert any(s.bias for s in shapes) and any(s.batch > 1 for s in shapes)
    assert all(s.macs <= KT.DEFAULT_MACS for s in shapes)
    shapes, skipped = KT.sweep(32, MK.Constraints(m_multiple="tile", bias=False), quick=True)
    assert all(s.m % 32 == 0 and not s.bias for s in shapes)
    assert any("m_multiple" in why for _, why in skipped) and any("bias: false" in why for _, why in skipped)
    _, skipped = KT.sweep(32, MK.Constraints(), budget=10_000, quick=True)
    assert any("over this run's budget" in why for _, why in skipped)


def _splitmix_py(z: int) -> int:
    m = (1 << 64) - 1
    z = (z + 0x9E3779B97F4A7C15) & m
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
    return z ^ (z >> 31)


def test_the_operand_generator_is_the_drivers_function():
    """The image and the host must draw the same operands; the driver's C is this exact arithmetic."""
    shape = KT.Shape("s", 3, 4, 5, bias=True, amplitude=7, seed=9)
    at, b, bias = KT.operands(shape, 1)
    base = lambda stream: ((10 & 0xFFFFFF) << 40) | (stream << 32)  # noqa: E731 -- seed + slice = 10
    assert at.reshape(-1)[4] == ((_splitmix_py(4 ^ base(1)) >> 32) % 15) - 7
    assert b.reshape(-1)[11] == ((_splitmix_py(11 ^ base(2)) >> 32) % 15) - 7
    assert bias[2] == ((_splitmix_py(2 ^ base(3)) >> 32) % 2001) - 1000
    driver = KT.emit_driver([shape], kernel_func="k", ref_func="r", align=16)
    assert "0x9e3779b97f4a7c15" in driver and "(seed & 0xffffffull) << 40" in driver


def test_the_verdict_localises_a_wrong_element_and_a_trap():
    good = KT.Shape("good", 2, 3, 4, seed=3)
    bad = KT.Shape("bad", 2, 3, 4, seed=5)
    lost = KT.Shape("lost", 2, 3, 4, seed=7)
    console = (
        f"KT BEGIN good\nKT CASE good 0 0 -1 -1 0 0 0 {KT.expected_digest(good):#018x} 10\n"
        "KT BEGIN bad\nKT CASE bad 2 1 0 4 1515870810 7 0 0x0 11\n"
        "KT BEGIN lost\n*** trap: mcause 2 at 0x80001234\n"
    )
    rows = KT.judge([good, bad, lost], KT.parse(console), console)
    assert rows[0]["ok"] and rows[0]["verdict"] == "ok"
    assert not rows[1]["ok"] and "2/6 elements differ (1 never written)" in rows[1]["verdict"]
    assert "first at C[1][1]: got 1515870810 (never written), reference 7" in rows[1]["verdict"]
    assert not rows[2]["ok"] and "TRAP or hang inside the kernel" in rows[2]["verdict"]
