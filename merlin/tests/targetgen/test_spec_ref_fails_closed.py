"""A spec_ref that names an op the spec does not declare must FAIL, not emit a different op's program.

Measured before this: `gemmini:op.matmul`, `gemmini:op.isa_flush` and `gemmini:op.TOTALLY_BOGUS`
returned a byte-identical command buffer, golden and coverage goal, because the upstream program
emitter accepts the op token and never reads it. A typo'd or renamed ref therefore produced a matmul
capsule under whatever name the ref carried. Deriving from a source that fails open is worse than not
deriving from it.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.capsule_source import SpecProgramUnavailable, SpecRefSource

_SRC = SpecRefSource()
pytestmark = pytest.mark.skipif(not _SRC.available(),
                                reason="no specir checkout in this environment (set SPECIR_ROOT)")

#: The gen whose ops these tests use. A per-target test is legitimately about one target.
_GEN = "gemmini"


def _module():
    import sys
    from pathlib import Path
    if _SRC.root not in sys.path:
        sys.path.insert(0, _SRC.root)
    from specir.gate import load_targets
    from specir.loading import parse_spec_file
    from specir.registry import _SPEC_ROOT
    entry = {t.get("id"): t for t in load_targets(_SPEC_ROOT)}[_GEN]
    return parse_spec_file(Path(_SPEC_ROOT) / entry["spec"])


def test_an_op_the_spec_does_not_declare_is_refused():
    with pytest.raises(SpecProgramUnavailable) as exc:
        _SRC.capture(f"{_GEN}:op.TOTALLY_BOGUS", workload=(16, 16, 16))
    msg = str(exc.value)
    assert "declares no" in msg
    assert "op.matmul" in msg, "the refusal must show what IS declared, so a typo is diagnosable"


def test_two_different_declared_ops_do_not_yield_the_same_contract():
    """⚠️ REGRESSION. The emitter ignores the op token, so the command buffer is genuinely identical
    for both. What must differ is the acceptance contract the capsule carries -- and it did not."""
    a = _SRC.capture(f"{_GEN}:op.matmul", workload=(16, 16, 16))
    b = _SRC.capture(f"{_GEN}:op.isa_flush", workload=(16, 16, 16))
    assert a.coverage_goal != b.coverage_goal, (
        "two different ops carried an identical coverage contract; the covers linkage is not read")


def test_coverage_goals_are_scoped_to_the_op_that_asked():
    """A matmul capsule carried a transcendental's test intent -- the wrong oracle and the wrong
    tolerance, stated with the same confidence as the right ones."""
    from merlin.targetgen.capsule_source import _coverage_goals

    mod = _module()
    for goal in _coverage_goals(mod, "op.matmul"):
        covers = goal.get("covers") or []
        assert not covers or "op.matmul" in covers, (
            f"{goal['node']!r} covers {covers} and was returned for op.matmul")


def test_a_gen_wide_goal_covering_nothing_is_still_returned():
    """A coverage goal with no `covers` list applies to the whole gen by construction. Dropping it
    while scoping would lose a real obligation, which is the opposite failure."""
    from merlin.targetgen.capsule_source import _coverage_goals, _covers

    mod = _module()
    goals = _coverage_goals(mod, "op.matmul")
    from specir.graph import all_nodes
    gen_wide = [n for n in all_nodes(mod)
                if (getattr(n, "name", "") or "") in ("spec.coverage_goal", "spec.test_intent")
                and not _covers(n)]
    if gen_wide:
        assert len(goals) >= len(gen_wide)


def test_the_covers_linkage_is_actually_readable():
    """⚠️ REGRESSION, and a guard that could not pass. `_covers` first tested
    `isinstance(table, dict)`; xDSL's DictionaryAttr holds an `immutabledict`, which is a Mapping and
    NOT a dict subclass, so the check returned "covers nothing" for every node and silently disabled
    the scoping. A guard that cannot pass is the same defect as a check that cannot fail."""
    from merlin.targetgen.capsule_source import _covers
    from specir.graph import all_nodes

    linked = [n for n in all_nodes(_module())
              if (getattr(n, "name", "") or "") in ("spec.coverage_goal", "spec.test_intent")
              and _covers(n)]
    assert linked, "no coverage node resolved a covers list; the linkage reader is not working"


def test_declared_ops_is_not_empty_and_names_the_op_vocabulary():
    """The op vocabulary a conformance cell cannot express: a cell collapses every one of these into
    `contraction` or `movement`."""
    from merlin.targetgen.capsule_source import declared_ops

    ops = declared_ops(_module())
    assert len(ops) > 1
    assert all(o.startswith("op.") for o in ops)
    assert "op.matmul" in ops
