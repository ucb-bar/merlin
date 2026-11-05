"""family_rollup: a passing case's performance step starts at once, bundled by family and shape class.

A run split its work by compiler layer and measured its first speed-up after three hours although many
dense products had passed long before. These tests hold the readiness rules: every passing case is in
a bundle whose representative gets its performance step now, a family with failing cases gets its
finish-correctness step beside it (with the library route that builds it fast from the start), neither
waits for the other, and the idle timing lanes take bundle representatives and newly passing large
cases first -- as an ordering only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from merlin_experiments.phase1 import operation_optimizations as O
from merlin_experiments.phase1.feedback import family_rollup as R
from merlin_experiments.phase1.feedback import idle_gsim as G
from merlin_experiments.phase1.feedback import idle_performance as I
from merlin_experiments.phase1.feedback import next_steps as N
from merlin_experiments.phase1.feedback import performance_headline as H

from merlin.common.paths import repo_root

ROOT = repo_root()
FIXTURE = json.loads(Path(__file__).with_name("library_pass_fresh4.json").read_text(encoding="utf-8"))
EDGE = 4


def member(op, shapes, *, role="correctness", family="contraction", layout=None, suite="core"):
    tensors = [{"name": f"T{i}", "role": r, "shape": s, "dtype": "i8"} for i, (r, s) in enumerate(shapes)]
    return {"declared_operation": op, "family": family, "input_tensors": tensors, "workload_role": role,
            "layout_requirements": layout or {}, "selfcheck_suite": suite}


def dense(m, k, n, **kw):
    return member("matmul", [("input", [m, k]), ("weight", [k, n])], **kw)


def conv(**kw):
    layout = {"stride": [1, 1], "padding": [1, 1, 1, 1], "dilation": [1, 1], "layout": "nhwc"}
    return member("conv2d", [("input", [1, 8, 8, 4]), ("weight", [36, 16])], layout=layout, **kw)


def timed(cycles, floor):
    return {"status": "PROVISIONAL", "complete_event_accounting": True, "estimated_whole_soc_cycles": cycles,
            "structural_roofline": {"structural_compute_floor_cycles": floor}}


def row(name, status, performance=None):
    out = {"capsule": name, "status": status, "tiers": {"L2": status}}
    if performance:
        out["performance"] = performance
    return out


MEMBERS = {
    # dense, tall (M >= N + edge): a large case with headroom, a small sibling with cycles, an untimed large one
    "D_big_tall": dense(64, 32, 8, role="large_kernel", suite="scalability"),
    "D_small_tall": dense(16, 8, 4),
    "D_new_tall": dense(48, 16, 8, role="large_kernel", suite="scalability"),
    # dense, wide: one passing correctness case and one failing large case
    "D_wide": dense(4, 8, 32),
    "D_wide_fail": dense(8, 16, 64, role="large_kernel", suite="scalability"),
    # convolution: one passing, one failing, one never run
    "C_pass": conv(),
    "C_fail": conv(),
    "C_new": conv(),
}
GRADE = [
    row("D_big_tall", "pass", timed(90_000, 10_000)),
    row("D_small_tall", "pass", timed(3_000, 1_000)),
    row("D_new_tall", "pass"),
    row("D_wide", "pass", timed(2_000, 1_000)),
    row("D_wide_fail", "fail"),
    row("C_pass", "pass", timed(5_000, 1_000)),
    row("C_fail", "fail"),
    {"capsule": "C_new", "status": "gated", "tiers": {}},
]


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "PUBLIC_CAPABILITIES.json").write_text(
        json.dumps({"schema": "merlin.public_capability_inventory.v1", "members": MEMBERS}))
    (tmp_path / "qa").mkdir()
    (tmp_path / "qa" / "latest_completed_grade.json").write_text(json.dumps({"per_capsule": GRADE}))
    catalog = O.load(ROOT)
    mechanisms = json.loads((ROOT / O.MECHANISM_CATALOG).read_text(encoding="utf-8"))
    selection = O.select(catalog, {"contraction": 5, "convolution": 3}, mechanism_catalog=mechanisms)
    (tmp_path / O.FILENAME).write_text(json.dumps(selection))
    (tmp_path / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps(FIXTURE["suggestions"]))
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "backend.py").write_text("import json\n")
    (tmp_path / ".agents" / "skills" / "ab-library-pass").mkdir(parents=True)
    R._GRADE_CACHE.clear()
    return tmp_path


def test_the_rollup_counts_each_family_and_bundles_passing_cases_by_shape_class(workspace):
    found = R.build(workspace, edge=EDGE)
    families = {r["family"]: r for r in found["families"]}
    assert (families["contraction"]["correct"], families["contraction"]["cases"]) == (4, 5)
    assert families["contraction"]["large"] == 3 and families["contraction"]["timed"] == 3
    assert families["contraction"]["failing"] == ["D_wide_fail"] and families["contraction"]["left_large"] == 1
    assert (families["convolution"]["correct"], families["convolution"]["cases"]) == (1, 3)
    assert families["convolution"]["unchecked"] == ["C_new"] and families["convolution"]["label"] == "conv"
    assert "dense 4/5 (3 timed)" in found["family_line"] and "conv 1/3" in found["family_line"]
    bundles = {b["label"]: b for b in found["bundles"]}
    tall = bundles["dense tall"]
    # The large case with the most headroom represents the bundle; its passing siblings are the held-out set.
    assert tall["representative"] == "D_big_tall" and set(tall["siblings"]) == {"D_new_tall", "D_small_tall"}
    assert tall["top"]["headroom_cycles"] == 80_000 and tall["panel"].endswith("--panel dense")
    # A passing case is ready although its family still has a failing case.
    assert bundles["dense wide"]["representative"] == "D_wide"
    assert any(b["representative"] == "C_pass" for b in found["bundles"])
    assert "dense tall (3 passing: D_big_tall, +2 siblings)" in found["lines"][0]
    assert found["ready_untimed"] == ["D_new_tall"]


def test_the_rollup_is_derived_from_declarations_not_names(workspace):
    renamed = {f"case{i}": value for i, value in enumerate(MEMBERS.values())}
    (workspace / "PUBLIC_CAPABILITIES.json").write_text(
        json.dumps({"schema": "merlin.public_capability_inventory.v1", "members": renamed}))
    (workspace / "qa" / "latest_completed_grade.json").unlink()
    found = R.build(workspace, edge=EDGE)
    assert {r["family"]: r["cases"] for r in found["families"]} == {"contraction": 5, "convolution": 3}
    assert all(r["correct"] == 0 for r in found["families"])  # no result yet is unchecked, not correct


def test_this_responses_rows_overlay_the_grade(workspace):
    found = R.build(workspace, rows=[{"capsule": "C_fail", "pass": True, "tiers": {"L2": "pass"}},
                                     {"capsule": "C_new", "status": "gated"}], edge=EDGE)
    conv = next(r for r in found["families"] if r["family"] == "convolution")
    assert conv["correct"] == 2 and conv["left"] == ["C_new"]  # a gated row keeps the grade's result


def test_the_construction_route_is_the_recipe_composing_most_catalogued_mechanisms(workspace):
    found = R.build(workspace, edge=EDGE)
    families = {r["family"]: r for r in found["families"]}
    dense_routes = [r["recipe"] for r in families["contraction"]["construction"]]
    assert dense_routes[0] == "heavy-pass-recipes/dense_epoch_recipe.py"
    assert "dense_panel_epochs" in families["contraction"]["construction"][0]["composes"]
    conv_routes = [r["recipe"] for r in families["convolution"]["construction"]]
    assert conv_routes == ["heavy-pass-recipes/conv_window_recipe.py", "heavy-pass-recipes/conv_epoch_recipe.py"]
    text = R.correctness_text(families["convolution"])
    assert text.startswith("conv: 1/3 correct -> finish correctness: C_fail, C_new")
    assert "build on conv_window_recipe.py" in text and "then conv_epoch_recipe.py" in text


def test_performance_and_correctness_steps_both_appear_ranked_by_value(workspace):
    families = R.build(workspace, edge=EDGE)
    found = N.block({}, [], None, workspace=workspace, families=families)
    steps = found["steps"]
    kinds = [(s["kind"], s["test_case"]) for s in steps]
    # Large cases first: the large tall representative's mechanism step, then the family with a failing large
    # case, then the untimed large case to time -- alternating, so neither kind pushes the other out.
    assert kinds[0] == ("mechanism", "D_big_tall")
    assert kinds[1] == ("finish_correctness", "D_wide_fail")
    assert ("finish_correctness", "C_fail") in kinds
    assert steps[0]["held_out"].startswith("python agent_selfcheck.py --panel dense (2 passing dense tall")
    assert "dense_epoch_recipe.py" in steps[1]["text"] and "--suite scalability" in steps[1]["apply"]
    assert found["lines"][0] == N.HEADER and found["lines"][1].startswith("1. D_big_tall: ")
    assert found["lines"][-1].startswith("families (correct/total")
    assert all(len(text.encode()) <= N.LINE_BYTES for text in found["lines"])
    assert sum(len(text.encode()) + 1 for text in found["lines"]) <= N.BLOCK_BYTES
    # The brief's bundle line now names the chosen mechanism.
    assert families["lines"][0].startswith("dense tall (3 passing: D_big_tall, +2 siblings) -> ")


def test_a_passing_case_without_cycles_is_told_to_time_itself(workspace):
    MEMBERS_ONLY = {"D_new_tall": MEMBERS["D_new_tall"]}
    (workspace / "PUBLIC_CAPABILITIES.json").write_text(
        json.dumps({"schema": "merlin.public_capability_inventory.v1", "members": MEMBERS_ONLY}))
    (workspace / "qa" / "latest_completed_grade.json").write_text(json.dumps({"per_capsule": [GRADE[2]]}))
    families = R.build(workspace, edge=EDGE)
    found = N.block({}, [], None, workspace=workspace, families=families)
    assert found["steps"][0]["kind"] == "measure"
    assert found["steps"][0]["apply"] == "python agent_selfcheck.py --suite scalability --capsules D_new_tall"


def test_idle_timing_prefers_representatives_and_newly_passing_large_cases(workspace):
    preferred = R.preferred_cases(workspace, ["D_big_tall", "D_new_tall", "D_wide_fail", "D_small_tall"])
    assert {"D_new_tall"} <= preferred and "D_wide_fail" not in preferred
    # Spike lane: preferred untimed cases first, then the established alternation.
    cohort = ["a", "b", "c"]
    l2 = {name: "pass" for name in cohort}
    assert I.select(cohort, l2, {}, "d", set(), limit=3, preferred={"c"}) == ["c", "a", "b"]
    assert I.select(cohort, l2, {}, "d", set(), limit=3) == ["a", "b", "c"]
    # gSIM lane: a preferred family lead first; in the rest a preferred case ahead of larger headroom.
    def entry(name, family, work, headroom, preferred=False, evidence=False):
        return {"capsule": name, "family": family, "work_cycles": work, "headroom_cycles": headroom,
                "timing_evidence": evidence, "measured_before": False, "preferred": preferred}

    entries = [entry("x1", "fx", 10, 5), entry("y1", "fy", 20, 5, preferred=True),
               entry("big", "fx", 99, 900, evidence=True), entry("rep", "fy", 50, 10, preferred=True, evidence=True)]
    assert [e["capsule"] for e in G.order(entries)] == ["y1", "x1", "rep", "big"]
    for item in entries:
        item.pop("preferred")
    assert [e["capsule"] for e in G.order(entries)] == ["x1", "y1", "big", "rep"]  # unchanged without it


def test_the_grade_headline_and_its_rendering_carry_ready_bundles(workspace):
    found = H._families(workspace, {"per_capsule": []})
    assert found["ready_bundles"] and found["cases_left"][0].startswith("dense: 4/5 correct")
    lines = H.render({"schema": H.SCHEMA, "grade": {}, "correctness": "x", "families": found})
    text = "\n".join(lines)
    assert "Ready for performance now" in text and "Correctness left" in text and "families (correct/total" in text


def test_the_console_shows_the_grade_rollup(workspace):
    from merlin_experiments.tracking.console.sources import performance as P

    grade = {"performance_headline": {"families": {"line": "families (x): dense 4/5 (3 timed)",
                                                   "ready_bundles": ["dense tall (3 passing: D_big_tall)"],
                                                   "cases_left": ["conv: 1/3 correct"]}}}

    class Reader:
        def document(self, path, **_):
            return grade if path.name == "latest_completed_grade.json" else None

    class Ctx:
        experiment = {"workspace": str(workspace)}
        reader = Reader()

    found = P._readiness(Ctx())
    assert found["line"].endswith("dense 4/5 (3 timed)")
    assert found["ready_bundles"] == ["dense tall (3 passing: D_big_tall)"]


def test_the_rollup_never_raises_on_a_bare_workspace(tmp_path):
    assert R.build(tmp_path) is None and R.preferred_cases(tmp_path) == set()
    assert R.attach({}, [{"capsule": "x"}], workspace=tmp_path) is None


def test_correct_counts_say_how_many_passes_are_for_an_older_version():
    from merlin_experiments.phase1.feedback import family_rollup

    record = {"label": "conv", "cases": 4, "correct": 1, "correct_older_version": 2, "left": ["b", "c", "d"],
              "failing_count": 1}
    assert family_rollup.correctness_text(record).startswith("conv: 1/4 correct (2 passed on an older version) -> ")
    record["correct_older_version"] = 0
    assert family_rollup.correctness_text(record).startswith("conv: 1/4 correct -> ")
