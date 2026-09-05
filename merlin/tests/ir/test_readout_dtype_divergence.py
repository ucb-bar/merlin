"""The COMMIT readout must mean the same thing in all three engines.

**History, because the fix is only safe if you know what it replaced.** `capsule_golden` narrowed on a
default of `i32` and handled any integer width; `runtime/simulator.py` and `runtime/reference.py`
defaulted to `i8` and narrowed on an exact `== "i8"` test. They diverged from 2026-06-20 and nobody
noticed for 77 days, because every shipped capsule declares `output_dtype` and the two rules agree
whenever it is present and is `i8` or at least 32 bits.

They did NOT agree when it was absent, or was `i16`/`i4`/`u8`. The direction of the error was always
the same: at L0 the golden is compared against `reference_outputs(agent_cb)`, so a submission that
simply omitted the attribute — legal, the schema marks it optional — got an i8 clamp applied to a
result the capsule declared `i32`, and died with *"your command buffer does not compute the declared
operation"*. Measured: **85 of 130** integer contraction capsules would fail that way for a CORRECT
backend. L1 could not catch it either, because both of its sides applied the same wrong rule.

**The fix, and why this direction.** Both runtime COMMIT sites now route through
`_narrow_int_readout`, the shared rule each engine already had and whose docstring already claimed to
match the golden, with a default of `i32`. The two defaults were never symmetric in risk: `i32` means
"absent ⇒ do not narrow", which cannot destroy a wide result, while `i8` silently clamps one. And
since the golden — the authority at L2/L3 — already treated absent as `i32`, any buffer that relied on
the old clamp was already failing L0. Changing it can only fix, not break.

One site is deliberately untouched: `runtime/backends/rvv_codegen.py` still carries the old inline
test. That file is under another workstream's lock. It is named here rather than left to be
rediscovered.
"""
from __future__ import annotations

import pytest


def _engines():
    from merlin.runtime.reference import _narrow_int_readout as ref
    from merlin.runtime.simulator import _narrow_int_readout as sim
    from merlin.targetgen.capsule_golden import _narrow_to_dtype as gold

    return gold, ref, sim


@pytest.mark.parametrize("dtype,expected", [
    ("i32", 74192),   # wide: nothing to narrow
    ("i16", 32767),   # saturates to the i16 range -- the case the old exact-i8 test could not express
    ("i8", 127),
    ("i4", 7),
    ("u8", 255),
])
def test_all_three_engines_agree_on_every_declared_dtype(dtype, expected):
    """The property the L0 comparison depends on. A disagreement here fails a correct backend."""
    from merlin.runtime.tensor import Tensor

    gold, ref, sim = _engines()
    t = Tensor((1, 1), [74192], "i32")
    got = (gold(t, dtype).data[0], ref(t, dtype, "COMMIT").data[0], sim(t, dtype, "COMMIT").data[0])
    assert got == (expected, expected, expected), (
        f"output_dtype={dtype!r}: golden/reference/simulator returned {got}; they must agree, or a "
        f"correct backend is graded against a rule no single engine implements")


def test_an_absent_output_dtype_does_not_silently_clamp():
    """Absent must mean "do not narrow". The opposite default destroyed wide results for 77 days."""
    import copy

    from merlin.runtime import simulate
    from merlin.runtime.reference import reference_outputs
    from merlin.verify.evaluate import _finish_lowering, _lower_to_interface

    # K large enough that the accumulator leaves the i8 range, or the test proves nothing.
    iface, tc = _lower_to_interface(4, 64, 4, 1)
    cb = _finish_lowering(iface, tc)
    absent = copy.deepcopy(cb)
    for command in absent["commands"]:
        if command["opcode"] == "COMMIT":
            command["attributes"].pop("output_dtype", None)

    declared_sim, declared_ref = simulate(cb)["outputs"], reference_outputs(cb)
    absent_sim, absent_ref = simulate(absent)["outputs"], reference_outputs(absent)
    assert absent_sim == absent_ref, "the two engines still disagree when the attribute is absent"
    assert absent_sim == declared_sim, (
        "omitting output_dtype changed the result -- absent must mean 'do not narrow', not 'clamp "
        "to i8'; that difference is what failed 85 of 130 capsules for correct backends")


def test_every_shipped_capsule_still_declares_output_dtype():
    """Kept after the fix, for a different reason than before.

    It no longer guards a divergence — there isn't one. It guards the CONTRACT: the attribute is still
    optional in the schema, so a capsule omitting it now relies on a default rather than stating its
    intent. Declaring it is the honest form, and this fails on the first capsule that stops.
    """
    from merlin.common.paths import merlin_dir

    root = merlin_dir() / "contract" / "capsules"
    if not root.is_dir():
        pytest.skip("no corpus tree in this checkout")

    offenders: list[str] = []
    total = 0
    for path in root.rglob("capsule.interface.mlir"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "merlin_iface.commit" not in line:
                continue
            total += 1
            if "output_dtype" not in line:
                offenders.append(f"{path.relative_to(root)}:{lineno}")
    assert total, "no commit ops found; this test would be vacuous"
    assert not offenders, f"{len(offenders)} shipped commit op(s) omit output_dtype: {offenders[:5]}"


def test_the_one_unfixed_site_is_named_rather_than_forgotten():
    """`runtime/backends/rvv_codegen.py` is under another workstream's lock and still has the old rule.

    Asserted so the remaining divergence is a tracked fact rather than something rediscovered later.
    When that file is unlocked and fixed, this test fails and should be deleted.
    """
    import inspect

    from merlin.runtime.backends import rvv_codegen

    assert 'attrs.get("output_dtype", "i8") == "i8"' in inspect.getsource(rvv_codegen), (
        "rvv_codegen no longer carries the old readout rule -- if it was fixed, delete this test and "
        "the note in the module docstring above")
