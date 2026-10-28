"""A composition the screen cannot place whole is disclosed, never admitted or published."""

from __future__ import annotations

import pytest

from merlin_experiments.phase0.composition_generation import (
    require_placement,
    unplaced_members,
    withhold_unplaced,
)


def _capsule(outcome="pass", members=(3, 5, 8)):
    return {
        "name": "SY_composition_chain_scheme_aligned",
        "operation": {"op": "composition_program"},
        "semantic": {
            "composition_truth": {
                "outcome": outcome,
                "chain_id": "chain",
                "scheme_id": "scheme",
                "shape_class": "aligned",
            }
        },
        "expected": {"composition": {"source_op_indices": list(members)}},
    }


def _decision(indices, status, placement):
    return {
        "source_op_indices": list(indices),
        "status": status,
        "placement": placement,
        "classification": "accelerator_candidate" if placement == "accelerator" else "host_required",
        "reason": "screened",
    }


def _screen(*decisions):
    return {"status": "unsupported", "decisions": list(decisions)}


def test_every_member_admitted_on_the_accelerator_is_kept(tmp_path):
    screen = _screen(*(_decision([i], "admitted", "accelerator") for i in (3, 5, 8)))
    assert unplaced_members(_capsule(), screen) == []
    require_placement(_capsule(), screen)
    assert withhold_unplaced(_capsule(), screen, tmp_path) is None
    assert tmp_path.is_dir()


def test_a_refused_member_withholds_and_discloses_the_composition(tmp_path):
    written = tmp_path / "capsule"
    written.mkdir()
    (written / "capsule.yaml").write_text("name: x\n")
    screen = _screen(
        _decision([3], "admitted", "accelerator"),
        _decision([5], "unsupported", None),
        _decision([8], "unknown", None),
        _decision([1], "admitted", "host"),
    )
    with pytest.raises(ValueError, match="every original chain member"):
        require_placement(_capsule(), screen)
    record = withhold_unplaced(_capsule(), screen, written)
    assert not written.exists()
    assert record["status"] == "planned_not_generated"
    assert record["unplaced_members"] == [5, 8]
    assert (record["chain"], record["scheme"], record["shape_class"]) == ("chain", "scheme", "aligned")
    # Only the chain's own member decisions are disclosed, each exactly as the screen made it.
    assert [row["source_op_indices"] for row in record["decisions"]] == [[3], [5], [8]]
    assert [row["status"] for row in record["decisions"]] == ["admitted", "unsupported", "unknown"]


def test_a_host_admitted_member_is_not_accelerator_placement(tmp_path):
    screen = _screen(_decision([3], "admitted", "accelerator"), _decision([5, 8], "admitted", "host"))
    assert unplaced_members(_capsule(), screen) == [5, 8]


def test_a_must_refuse_composition_is_never_withheld(tmp_path):
    screen = _screen(_decision([3], "unsupported", None))
    assert unplaced_members(_capsule(outcome="refuse"), screen) == []
    assert withhold_unplaced(_capsule(outcome="refuse"), screen, tmp_path) is None
    assert tmp_path.is_dir()


def test_ambiguous_member_identity_still_fails(tmp_path):
    screen = _screen(_decision([3], "admitted", "accelerator"), _decision([3, 5], "admitted", "accelerator"))
    with pytest.raises(ValueError, match="ambiguous"):
        withhold_unplaced(_capsule(), screen, tmp_path)
    assert tmp_path.is_dir()


def test_other_programs_are_untouched(tmp_path):
    capsule = {"operation": {"op": "component_program"}}
    assert unplaced_members(capsule, _screen()) == []
    assert withhold_unplaced(capsule, _screen(), tmp_path) is None


def _withhold_unplaced_calls():
    import ast
    import inspect

    from merlin_experiments.phase0 import generation

    tree = ast.parse(inspect.getsource(generation.generate_target))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "withhold_unplaced"]
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    return calls, parents


def test_generation_discloses_an_unplaced_composition_in_every_evidence_mode():
    """Diagnostic generation discloses as verified generation does instead of raising at require_placement."""
    import ast

    calls, parents = _withhold_unplaced_calls()
    assert len(calls) == 1
    node = calls[0]
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.IfExp, ast.If)):
            assert "evidence_mode" not in ast.unparse(node.test), ast.unparse(node.test)


def test_a_diagnostic_composition_with_a_refused_member_is_disclosed_not_raised(tmp_path):
    """The generation sequence (withhold_unplaced, then require_placement) never reaches the raise."""
    written = tmp_path / "capsule"
    written.mkdir()
    (written / "capsule.yaml").write_text("name: x\n")
    screen = _screen(_decision([3], "admitted", "accelerator"), _decision([5], "unsupported", None), _decision([8], "admitted", "accelerator"))
    record = withhold_unplaced(_capsule(), screen, written)
    assert record is not None and record["status"] == "planned_not_generated" and record["unplaced_members"] == [5]
    assert not written.exists()
