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

**The first attempt at this fix over-corrected, and that is worth recording.** It made the two runtime
helpers RAISE on a non-integer `output_dtype`, on the reasoning that an integer engine has no
definition for a float readout. The golden passes such a token through unchanged, so raising replaced
one divergence with another — and this one fired on real buffers: 37 tests across
`test_xdsl_vector_ops`, `test_xdsl_whole_model_chain`, `test_gemmini_native_pooling` and
`test_rtl_checks` commit an `f32` tensor produced by a VECTOR_MAP chain, which the old `== "i8"` test
had silently passed through for as long as they had existed. The rule is not "be strict"; it is "agree
with the golden", and `test_all_three_engines_agree_on_every_declared_dtype` now covers `f32`/`bf16`
so a future strictness argument fails the suite instead of the corpus.

**And the first pass at collapsing the rules only collapsed two of five.** `COMMIT` and `CONV2D` were
routed through the shared helper in both engines and the work was reported as done, while `BIAS_ADD`,
`ATTENTION_QK` and `ATTENTION_PV` kept the exact-`i8` test — three more opcodes carrying the identical
divergence, on paths the golden narrows at any width through its one `_apply_epilogue`. They were found
by deriving the narrowing set from the engines instead of trusting the earlier claim. All five now route
through `_narrow_int_readout`, and `test_no_engine_keeps_a_private_readout_rule` fails on the next one
that does not.

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
    ("f32", 74192),   # not an integer container: pass through, as the golden does
    ("bf16", 74192),
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


def test_an_absent_output_dtype_is_refused_rather_than_defaulted():
    """The contract moved past this test's original claim, and the stronger form is worth stating.

    It used to assert that an absent `output_dtype` means "do not narrow" — true, and the fix for the
    77-day divergence. But agreement on a DEFAULT is weaker than a declaration: it makes the buffer's
    meaning depend on a convention the submission never stated, so the next divergence would be as
    silent as the last. `validate_command_buffer` now refuses a narrowing command that declares no
    container, by opcode and command index, and `simulate` raises rather than guessing.

    The engine-level guarantee is kept below, because it is what protects any buffer that reaches the
    engines by another route.
    """
    import copy

    import pytest

    from merlin.runtime import simulate
    from merlin.runtime.simulator import SimulationError
    from merlin.verify.evaluate import _finish_lowering, _lower_to_interface

    # K large enough that the accumulator leaves the i8 range, or the test proves nothing.
    iface, tc = _lower_to_interface(4, 64, 4, 1)
    cb = _finish_lowering(iface, tc)
    assert simulate(cb)["outputs"], "the declared buffer must still run"

    absent = copy.deepcopy(cb)
    for command in absent["commands"]:
        if command["opcode"] == "COMMIT":
            command["attributes"].pop("output_dtype", None)
    with pytest.raises(SimulationError, match="output_dtype"):
        simulate(absent)


def test_the_engines_still_agree_on_absent_if_one_reaches_them():
    """Defence in depth: the refusal above is a gate, and gates get bypassed.

    Both engines default to `i32` — absent means DO NOT NARROW — which is the direction that cannot
    destroy a wide result. The opposite default silently clamped one, and that is what failed 85 of 130
    integer contraction capsules for a correct backend.
    """
    from merlin.runtime.tensor import Tensor

    gold, ref, sim = _engines()
    t = Tensor((1, 1), [74192], "i32")
    assert (gold(t, "i32").data[0], ref(t, "i32", "COMMIT").data[0], sim(t, "i32", "COMMIT").data[0]) \
        == (74192, 74192, 74192)


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


def test_no_engine_keeps_a_private_readout_rule():
    """The invariant the collapse was supposed to establish, asserted instead of claimed.

    Reporting "the rules are collapsed" after collapsing two of five is what happened the first time.
    The exact-i8 test is a literal, so its absence is checkable: if a sixth narrowing site appears, or
    an existing one is reverted to its own opinion, this fails.
    """
    import inspect

    from merlin.runtime import reference, simulator

    for module in (simulator, reference):
        src = inspect.getsource(module)
        assert 'output_dtype", "i32") == "i8"' not in src, (
            f"{module.__name__} carries a private exact-i8 readout rule again; route it through "
            f"_narrow_int_readout so it cannot disagree with the golden about i16/i4/u8")


def test_the_validator_demands_a_declared_container_for_exactly_the_narrowing_opcodes():
    """`NARROWING_OPCODES` must be the set the engines actually narrow, not a list someone typed.

    Derived from the simulator's own source: every opcode passed to `_narrow_int_readout` is one whose
    readout depends on `output_dtype`, and the validator has to demand the attribute for exactly those.
    A validator that demanded it for fewer would leave the silent case open; one that demanded it for
    more would reject correct buffers.
    """
    import inspect

    from merlin.runtime import simulator
    from merlin.runtime.commandbuffer import NARROWING_OPCODES

    narrowed = set()
    for line in inspect.getsource(simulator).splitlines():
        head, sep, tail = line.partition('_narrow_int_readout(')
        if not sep or "def " in head:
            continue
        # the op name is the last quoted argument on the line
        parts = tail.split('"')
        if len(parts) >= 2:
            narrowed.add(parts[-2])
    assert narrowed, "no narrowing call sites found; this test would be vacuous"
    assert narrowed == set(NARROWING_OPCODES), (
        f"the simulator narrows {sorted(narrowed)} but the validator demands output_dtype for "
        f"{sorted(NARROWING_OPCODES)}; the two must be the same set")


def test_a_narrowing_command_without_output_dtype_is_a_named_problem():
    """The failure this replaces was a numeric mismatch hundreds of lines downstream."""
    from merlin.runtime.commandbuffer import validate_command_buffer

    cb = {"abi_version": "0.1", "target": "t", "tensors": {"A": {}},
          "commands": [{"opcode": "COMMIT", "operands": {"src": "A", "dst": "Y"},
                        "attributes": {}}]}
    problems = validate_command_buffer(cb)
    assert any("output_dtype" in p and "COMMIT" in p for p in problems), (
        f"an undeclared readout container must be reported by opcode: {problems}")

    cb["commands"][0]["attributes"]["output_dtype"] = "i32"
    assert not validate_command_buffer(cb), "a declared container must validate cleanly"


def test_a_non_narrowing_command_is_not_asked_for_one():
    """MOVEMENT copies bytes; demanding a readout width from it would reject correct buffers."""
    from merlin.runtime.commandbuffer import NARROWING_OPCODES, validate_command_buffer

    assert "MOVEMENT" not in NARROWING_OPCODES
    cb = {"abi_version": "0.1", "target": "t", "tensors": {"A": {}},
          "commands": [{"opcode": "MOVEMENT", "operands": {"src": "A", "dst": "Y"},
                        "attributes": {}}]}
    assert not validate_command_buffer(cb)
