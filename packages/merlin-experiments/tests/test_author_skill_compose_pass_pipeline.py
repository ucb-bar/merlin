"""The `compose-pass-pipeline` author skill: snippets run, the generated reference is fresh.

The pass-contract script reads ``PassSpec`` and ``compile_trace.declare`` structurally; it is
checked on a fixture tree (positional, keyword, computed and rewired declarations) and against
the real library, whose generated copy in ``reference/pass_specs.md`` must match byte for byte.
"""

from __future__ import annotations

import importlib.util
import json

import pytest
from author_skill_checks import execute_blocks, skill_dir, structural_problems, unresolved_symbols

from merlin.common.paths import repo_root

NAME = "compose-pass-pipeline"
SCRIPT = skill_dir(NAME) / "scripts" / "pass_specs.py"
#: The schedule-pipeline order the skill states, as library pass names (optional ones included).
SCHEDULE_ORDER = ("semantic_admission", "configuration", "fusion", "storage", "completion", "emission",
                  "final_program")


def _script():
    spec = importlib.util.spec_from_file_location("author_skill_pass_specs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def specs_tool():
    return _script()


@pytest.fixture(scope="module")
def skill():
    return execute_blocks(NAME)


FIXTURE = '''
from dataclasses import replace
from merlin.transforms.pipeline_state import PassSpec
from merlin.common import compile_trace

STAGES = compile_trace.declare("toy", ("parse", *EXTRA, "emit"), entry="pkg.toy.run", summary="toy")


def build(extra):
    specs = (
        PassSpec("read", 0, produces=("source",), mutates_ir=False, ir_level="toy"),
        PassSpec("lower", 3, ("source",), ("lowered",), ("*",), True, "toy"),
        PassSpec("final", 4, requires=("lowered",), produces=("checked",), mutates_ir=False, ir_level="toy"),
    )
    plan = PassSpec("plan", 1, requires=("source",), produces=("plan",), mutates_ir=False, ir_level="toy")
    specs = (specs[0], plan, replace(specs[1], requires=("source", "plan")), specs[2])
    other = PassSpec(extra.name, 2, requires=extra.requires, ir_level="toy")
    return specs, other
'''


@pytest.fixture
def fixture_tree(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "toy.py").write_text(FIXTURE)
    (package / "unrelated.py").write_text("def replace(x, name=1):\n    return x\n")
    return package


def test_skill_layout_text_and_scripts_follow_the_rules():
    assert structural_problems(NAME) == []


def test_every_named_library_symbol_and_parameter_resolves():
    assert unresolved_symbols(NAME) == []


def test_scan_reads_positional_keyword_computed_and_rewired_declarations(specs_tool, fixture_tree):
    rows = specs_tool.scan(fixture_tree)
    (staged,) = specs_tool.stage_pipelines(rows)
    assert staged["pipeline"] == "toy" and staged["entry"] == "pkg.toy.run"
    assert [str(stage) for stage in staged["stages"]] == ["parse", "*EXTRA", "emit"]
    (key,) = specs_tool.pipelines(rows)
    assert key == "pkg.toy:build"
    declared = {row["name"]: row for row in rows if row["kind"] == "declare"}
    lower = declared["lower"]
    assert (lower["requires"], lower["produces"], lower["invalidates"], lower["mutates_ir"], lower["ir_level"]) == (
        ("source",), ("lowered",), ("*",), True, "toy")
    assert [str(name) for name in declared if isinstance(name, specs_tool.Dynamic)] == ["extra.name"]
    assert isinstance(declared[next(n for n in declared if isinstance(n, specs_tool.Dynamic))]["requires"],
                      specs_tool.Dynamic)
    rewires = [row for row in rows if row["kind"] == "rewire"]
    assert [(row["target"], row["requires"]) for row in rewires] == [("specs[1]", ("source", "plan"))]
    assert all(row["module"] != "pkg.unrelated" for row in rows)
    text = specs_tool.markdown(rows)
    assert "### `pkg.toy:build`" in text and "| 1 | plan | source | plan | - | False | toy |" in text


def test_check_uses_the_library_validator_and_names_the_failing_pass(specs_tool, fixture_tree):
    rows = specs_tool.scan(fixture_tree)
    assert specs_tool.check(rows, "toy:build", ["read", "plan", "lower", "final"])["status"] == "valid"
    stale = specs_tool.check(rows, "toy:build", ["read", "lower", "plan"])
    assert stale["status"] == "invalid" and stale["failing_pass"] == "plan"
    missing = specs_tool.check(rows, "toy:build", ["read", "final"])
    assert missing["failing_pass"] == "final" and "lacks prerequisites: lowered" in missing["reason"]
    assert specs_tool.check(rows, "toy:build", ["read", "nope"])["status"] == "unknown_pass"
    assert specs_tool.check(rows, "toy:build")["status"] == "dynamic"
    assert specs_tool.check(rows, "absent")["status"] == "not_found"


def test_cli_prints_markdown_json_and_check_exit_codes(specs_tool, fixture_tree, capsys):
    assert specs_tool.main(["--root", str(fixture_tree), "--grep", "toy"]) == 0
    assert "Entry-point stage orders" in capsys.readouterr().out
    assert specs_tool.main(["--root", str(fixture_tree), "--json"]) == 0
    assert {row["kind"] for row in json.loads(capsys.readouterr().out)} == {"declare", "rewire", "stages"}
    assert specs_tool.main(["--root", str(fixture_tree), "--check", "build", "--order", "read,lower,plan"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"


def test_generated_reference_matches_the_library(specs_tool):
    rows = specs_tool.scan(repo_root() / "src" / "merlin")
    expected = specs_tool.REFERENCE_HEADER + specs_tool.markdown(rows, lines=False)
    actual = (skill_dir(NAME) / "reference" / "pass_specs.md").read_text(encoding="utf-8")
    assert actual == expected, "regenerate: python scripts/pass_specs.py --root src/merlin --reference > reference/pass_specs.md"


def test_the_stated_schedule_order_is_the_library_order(specs_tool):
    rows = specs_tool.scan(repo_root() / "src" / "merlin")
    declared = {str(row["name"]) for row in specs_tool.pipelines(rows)[
        "merlin.compile.scheduling.schedule_pipeline:compile_schedule"] if row["kind"] == "declare"}
    assert set(SCHEDULE_ORDER) | {"prefetch", "post_storage.name"} == declared
    assert specs_tool.check(rows, "schedule_pipeline:compile_schedule", list(SCHEDULE_ORDER))["status"] == "valid"
    swapped = ["semantic_admission", "storage", "fusion", "completion", "emission", "final_program"]
    assert specs_tool.check(rows, "schedule_pipeline:compile_schedule", swapped)["status"] == "invalid"


def test_insert_before_wires_the_consumer_and_refuses_bad_stages(skill):
    from merlin.transforms.pipeline_state import PassSpec

    base = (PassSpec("read", 0, produces=("source",), mutates_ir=False),
            PassSpec("lower", 3, requires=("source",), produces=("lowered",)))
    plan = PassSpec("plan", 1, requires=("source",), produces=("plan",), mutates_ir=False)
    order = skill["insert_before"](base, plan, "lower")
    assert [spec.name for spec in order] == ["read", "plan", "lower"]
    assert order[2].requires == ("source", "plan")
    with pytest.raises(ValueError, match="ordering"):
        skill["insert_before"](base, PassSpec("late", 5, produces=("x",), mutates_ir=False), "lower")
    after_mutation = PassSpec("check", 4, requires=("source",), mutates_ir=False)
    with pytest.raises(ValueError, match="lacks prerequisites: source"):
        skill["insert_before"]((*order, PassSpec("tail", 9, requires=("lowered",))), after_mutation, "tail")


def _module(value=3):
    from merlin.transforms.certificates import parse_ir

    return parse_ir(f"builtin.module {{func.func @f() -> i32 {{\n  %a = arith.constant {value} : i32\n"
                    "  func.return %a : i32\n}}")


def test_run_order_records_products_and_invalidations(skill):
    from merlin.transforms.pipeline_state import PassSpec

    specs = (PassSpec("count", 0, produces=("ops",), mutates_ir=False),
             PassSpec("plan", 1, requires=("ops",), produces=("plan",), mutates_ir=False),
             PassSpec("rewrite", 2, requires=("plan",), produces=("rewritten",)))
    actions = {"count": lambda s: {"ops": len(tuple(s.module.walk()))},
               "plan": lambda s: {"plan": s.analysis("ops") * 2},
               "rewrite": lambda s: {"rewritten": s.analysis("plan")}}
    session, records = skill["run_order"](_module(), specs, actions)
    assert [name for name, _p, _i in records] == ["count", "plan", "rewrite"]
    assert session.analysis("rewritten") == 2 * session.analysis("ops") > 0
    stale = dict(actions, plan=lambda s: (setattr(s, "module", _module(4)), {"plan": 0})[1])
    with pytest.raises(ValueError, match="changed IR"):
        skill["run_order"](_module(), specs, stale)


def test_compare_orders_runs_each_order_on_its_own_clone(skill):
    from merlin.transforms.certificates import ir_digest
    from merlin.transforms.pipeline_state import PassSpec

    specs = (PassSpec("a", 0, produces=("x",), mutates_ir=False),
             PassSpec("b", 0, produces=("y",), mutates_ir=False),
             PassSpec("emit", 1, requires=("x",), produces=("out",), mutates_ir=False))
    actions = {"a": lambda s: {"x": 1}, "b": lambda s: {"y": 2},
               "emit": lambda s: {"out": ir_digest(s.module)}}
    module = _module()
    before = ir_digest(module)
    results = skill["compare_orders"](module, specs, actions, {"ab": ("a", "b", "emit"), "a": ("a", "emit")}, "out")
    assert {name: status for name, (status, _error, _result) in results.items()} == {"ab": "compiled", "a": "compiled"}
    assert results["a"][2].artifact == before and ir_digest(module) == before
    with pytest.raises(ValueError, match="prerequisites"):
        skill["compare_orders"](module, specs, actions, {"bad": ("b", "emit")}, "out")
