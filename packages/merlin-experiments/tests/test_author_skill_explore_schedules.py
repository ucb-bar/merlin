"""The explore-schedules author skill: its code block runs as written and its held-out check is honest.

The SKILL.md snippet is executed verbatim with stand-ins for the author's own functions (a plain
Python IR whose emitted program contains transfers), so a change to ``pass_exploration`` that breaks
the documented use fails here. ``holdout_rank`` must rank on training geometries only, report the
training winner's rank on the frozen held-out geometries, and leave unknown observations unranked.
"""

from __future__ import annotations

import json
import sys

import pytest
from author_skill_fixtures import assert_generic_skill, front_matter, load_script, skill_dir

from merlin.transforms.pass_exploration import ExplorationGeometry, ValidationPartition

SKILL = "explore-schedules"


def _snippet() -> str:
    _, body = front_matter(SKILL)
    start = body.index("```python\n") + len("```python\n")
    block = body[start:body.index("```", start)]
    lines = block.splitlines()
    indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    return "\n".join(line[indent:] for line in lines).replace("SKILL_DIR", str(skill_dir(SKILL)))


def _geometry(m, family="dense"):
    return ExplorationGeometry(family, ((m, 16), (16, 8)))


def _stand_ins(held_out_rows):
    """Toy emitted transfers test the documented structural screen, not target performance."""
    def shapes_of(ir):
        return ((ir["m"], ir["k"]), (ir["k"], ir["n"]))

    def my_lower(ir, *, order, block):
        steps = (ir["m"] // block) * ir["n"] + (ir["n"] if order == "nmk" else 0)
        return {**ir, "transfers": [dict(bytes=ir['k']) for _ in range(steps)]}

    def my_emit(ir):
        return {"transfers": ir["transfers"]}

    training = (_geometry(8), _geometry(16))
    held_out = tuple(_geometry(m) for m in held_out_rows)
    return {
        "shapes_of": shapes_of, "my_lower": my_lower, "my_emit": my_emit,
        "source_ir": lambda geometry: {"m": geometry.operand_shapes[0][0], "k": 16, "n": 8},
        "count_operand_bytes": lambda data: sum(row['bytes'] for row in json.loads(data)['transfers']),
        "partition": ValidationPartition(training, held_out),
    }


def _run_snippet(capsys, held_out_rows):
    namespace = _stand_ins(held_out_rows)
    saved = list(sys.path)
    try:
        exec(compile(_snippet(), "SKILL.md", "exec"), namespace)
    finally:
        sys.path[:] = saved
    capsys.readouterr()
    return namespace["report"]


def test_skill_is_generic_bounded_and_self_describing():
    assert_generic_skill(SKILL, allowed_imports=frozenset({"merlin"}))
    fields, body = front_matter(SKILL)
    assert "holdout_rank.py" in fields["description"] and "pass_exploration" in fields["description"]
    assert "ab-library-pass" in body and "holds_on_held_out" in body and "winner_bindings" in body


def test_the_documented_snippet_finds_a_winner_that_holds(capsys):
    report = _run_snippet(capsys, held_out_rows=(32, 64))
    assert report["winner"] is not None and report["holds_on_held_out"] is True
    assert report["winner_held_out_position"] == 0
    assert report["winner_bindings"][-1]["knobs"] == {"order": "mnk", "block": 4}
    assert report["performance_qualification"] is False and report["basis"] == "static_emitted_ir"
    assert report["training"]["objectives"] == ("operand_bytes",)
    assert report["partition"]["held_out"][0]["operand_shapes"] == ((32, 16), (16, 8))


def test_a_winner_that_does_not_apply_to_a_tail_shape_does_not_hold(capsys):
    report = _run_snippet(capsys, held_out_rows=(32, 6))  # 6 is not a multiple of the winning block
    assert report["winner_bindings"][-1]["knobs"]["block"] == 4
    assert report["holds_on_held_out"] is False and report["winner_held_out_position"] is None
    unranked = dict(report["held_out"]["unranked_unknown"])
    assert report["winner"] in unranked and unranked[report["winner"]] == ("operand_bytes",)
    statuses = report["statuses"]["held_out"][_geometry(6).identity]
    assert statuses[report["winner"]] == "unsupported"
    assert report["held_out"]["ranked"] and report["held_out"]["ranked"][0][0] != report["winner"]


def test_holdout_rank_refuses_a_partition_geometry_without_a_module(capsys):
    holdout = load_script(SKILL, "holdout_rank")
    namespace = _stand_ins((32,))
    with pytest.raises(ValueError, match="no source module"):
        holdout.rank_on_holdout(namespace["partition"], {}, None, None, output="program", emission=None,
                                metrics=None, objectives=("commands",), domain="d")
    assert json.loads(holdout.dumps({"t": (1, 2)})) == {"t": [1, 2]}
