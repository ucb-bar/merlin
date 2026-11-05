"""The operator-built reference map reaches a case only when the case can use it.

A reference region is useful once a case passes numerically and is a performance target, and only
when it implements the same kind of operation: these tests hold the matching grades (exact dimensions,
same shape class under the tile edge, same kind), the readiness rule (a passing bundle representative or
a large passing case with headroom; never a failing, unchecked or plain sibling case), the hardware-loop
note, the compact line under the case's next step, and the read-only staging of identical bytes.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import yaml
from merlin_experiments.phase1.feedback import family_rollup as R
from merlin_experiments.phase1.feedback import next_steps as N
from merlin_experiments.phase1.feedback import reference_map as M
from test_family_rollup import EDGE, GRADE, MEMBERS, dense, member, row, timed
from test_family_rollup import workspace as rollup_workspace  # noqa: F401 -- pytest fixture

MANIFEST_SHA = "f" * 64


def entry(
    identifier,
    kind,
    dims=None,
    *,
    file="resnet50_listing/resnet50.txt",
    locator=None,
    techniques=("weights once per channel block", "1 fence per 8 tiles"),
    loops=False,
    confidence="verified",
):
    return {
        "id": identifier,
        "file": file,
        "locator": locator or {"lines": [1200, 1490]},
        "model_context": "fixture model",
        "operation": {"kind": kind, "dims": dims or {}, "epilogue": [], "dtypes": {"input": "i8", "acc": "i32"}},
        "techniques": list(techniques),
        "uses_hardware_loops": loops,
        "confidence": confidence,
        "evidence": "fixture listing region",
    }


CONV_DIMS = {"N": 1, "C": 4, "H": 8, "W": 8, "K": 16, "R": 3, "S": 3, "stride": 1, "pad": 1}
MAP = {
    "schema": M.SCHEMA,
    "reference_manifest_sha256": MANIFEST_SHA,
    "entries": [
        entry("E1-exact-tall", "matmul", {"M": 64, "K": 32, "N": 8}),
        entry(
            "E2-tall-loops",
            "matmul",
            {"M": 128, "K": 64, "N": 16},
            loops=True,
            confidence="inferred",
            techniques=("loop_ws over the whole layer",),
        ),
        entry(
            "E3-conv",
            "conv2d",
            CONV_DIMS,
            file="conv/kernel.c",
            locator={"function": "conv3x3", "lines": [10, 40]},
            techniques=("rows of the window resident",),
        ),
        entry("E4-pool", "pool", {}, locator={"address_range": ["0x100", "0x180"]}),
        entry("E5-wide", "matmul", {"M": 4, "K": 64, "N": 256}),
    ],
    "coverage": {"kinds": ["matmul", "conv2d", "pool"]},
    "not_covered": ["attention"],
}


def conv_declaration():
    return {
        "operation": {
            "op": "conv2d",
            "attributes": {
                "ifm": "IFM",
                "weight": "W",
                "ci": 4,
                "kh": 3,
                "kw": 3,
                "stride": [1, 1],
                "padding": [1, 1, 1, 1],
                "dilation": [1, 1],
                "layout": "nhwc",
                "epilogue": [],
            },
        },
        "inputs": [
            {"name": "W", "role": "weight", "shape": [36, 16], "dtype": "i8"},
            {"name": "IFM", "role": "input", "shape": [1, 8, 8, 4], "dtype": "i8"},
        ],
    }


@pytest.fixture
def workspace(rollup_workspace):  # noqa: F811
    """The roll-up fixture plus a second large tall case with headroom, a thin case and an add."""
    members = dict(MEMBERS)
    members["D_big2_tall"] = dense(32, 16, 8, role="large_kernel", suite="scalability")
    members["D_thin"] = dense(1, 16, 16)
    members["A_add"] = member("residual_add", [("input", [4, 16]), ("input", [4, 16])], family="elementwise_map")
    grade = [
        *GRADE,
        row("D_big2_tall", "pass", timed(20_000, 10_000)),
        row("D_thin", "pass", timed(900, 300)),
        row("A_add", "pass", timed(400, 100)),
    ]
    (rollup_workspace / "PUBLIC_CAPABILITIES.json").write_text(
        json.dumps({"schema": "merlin.public_capability_inventory.v1", "members": members})
    )
    (rollup_workspace / "qa" / "latest_completed_grade.json").write_text(json.dumps({"per_capsule": grade}))
    for name in ("C_pass", "C_fail"):
        path = rollup_workspace / "layers" / name / "capsule.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(conv_declaration()))
    R._GRADE_CACHE.clear()
    return rollup_workspace


def _pin(tmp_path, document=MAP):
    path = tmp_path / "operator" / "REFERENCE_MAP.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_the_schema_is_checked_before_any_use():
    assert M.validate(copy.deepcopy(MAP)) is not None
    for damage, match in (
        (lambda d: d.update(schema="other"), "schema"),
        (lambda d: d["entries"][0]["operation"].update(kind="gemm"), "operation.kind"),
        (lambda d: d["entries"][0].update(locator={"lines": [9, 3]}), "locator"),
        (lambda d: d["entries"][0].update(file="../escape.c"), "relative"),
        (lambda d: d["entries"].append(copy.deepcopy(d["entries"][0])), "unique"),
        (lambda d: d["entries"][1].update(uses_hardware_loops="yes"), "uses_hardware_loops"),
        (lambda d: d.update(reference_manifest_sha256="x"), "reference manifest"),
    ):
        document = copy.deepcopy(MAP)
        damage(document)
        with pytest.raises(ValueError, match=match):
            M.validate(document)


def test_matching_grades_exact_same_class_technique_and_none():
    tall = M.case_operation(
        {
            "operation": {"op": "matmul", "attributes": {"lhs": "A", "weight": "W"}},
            "inputs": [
                {"name": "A", "role": "input", "shape": [64, 32]},
                {"name": "W", "role": "weight", "shape": [32, 8]},
            ],
        }
    )
    assert tall["kind"] == "matmul" and tall["dims"] == {"M": 64, "K": 32, "N": 8}
    grades = {e["id"]: M.grade(e, tall, EDGE) for e in MAP["entries"]}
    assert grades == {
        "E1-exact-tall": "exact",
        "E2-tall-loops": "same_class",
        "E3-conv": None,
        "E4-pool": None,
        "E5-wide": "technique",
    }
    # Without a tile edge no class is derivable: same kind only.
    assert M.grade(MAP["entries"][1], tall, None) == "technique"
    conv = M.case_operation(conv_declaration())
    assert conv["kind"] == "conv2d"
    assert {key: conv["dims"][key] for key in ("N", "C", "H", "W", "K", "R", "S")} == {
        key: CONV_DIMS[key] for key in ("N", "C", "H", "W", "K", "R", "S")
    }
    assert [(e["id"], g) for e, g in M.matches(MAP, conv, EDGE)] == [("E3-conv", "exact")]
    strided = conv_declaration()
    strided["operation"]["attributes"]["stride"] = [2, 2]
    assert M.grade(MAP["entries"][2], M.case_operation(strided), EDGE) != "exact"
    unstated = conv_declaration()
    for key in ("stride", "padding"):
        del unstated["operation"]["attributes"][key]
    assert M.grade(MAP["entries"][2], M.case_operation(unstated), EDGE) != "exact"  # unknown never equals stated
    add = M.case_operation(
        {
            "operation": {"op": "residual_add", "attributes": {"lhs": "X0", "rhs": "X1"}},
            "inputs": [
                {"name": "X0", "role": "input", "shape": [4, 16]},
                {"name": "X1", "role": "input", "shape": [4, 16]},
            ],
        }
    )
    assert add["kind"] == "residual_add" and M.matches(MAP, add, EDGE) == []


def test_lines_are_compact_and_name_hardware_loops_and_siblings():
    first = M.line(MAP["entries"][0], "exact", ["S1", "S2", "S3"])
    assert first == (
        "reference: resnet50_listing/resnet50.txt:1200-1490 matmul 64x32x8 (exact) — weights once per "
        "channel block; 1 fence per 8 tiles; compare on S1, S2 (+1)"
    )
    looped = M.line(MAP["entries"][1], "same_class")
    assert "[inferred]" in looped and "(uses hardware loops: translate to ordinary commands)" in looped
    conv = M.line(MAP["entries"][2], "exact")
    assert conv.startswith("reference: conv/kernel.c:conv3x3:10-40 conv2d 3x3 8x8x4→16 (exact)")
    assert M.line(MAP["entries"][3], "technique").startswith(
        "reference: resnet50_listing/resnet50.txt@0x100-0x180 pool"
    )
    long = copy.deepcopy(MAP["entries"][0])
    long["techniques"] = ["x" * 400] * 5
    assert len(M.line(long, "exact", ["a" * 200] * 4).encode()) <= M.LINE_BYTES


def test_only_passing_performance_targets_get_reference_lines(workspace):
    families = R.build(workspace, edge=EDGE)
    targets = M.targets(families)
    # Representatives and large passing cases with headroom; never plain siblings or failing/unchecked cases.
    assert {"D_big_tall", "D_big2_tall", "D_wide", "C_pass", "D_thin", "A_add"} <= set(targets)
    assert not {"D_small_tall", "D_new_tall", "D_wide_fail", "C_fail", "C_new"} & set(targets)
    found = M.build(families, workspace, document=MAP, edge=EDGE)
    cases = found["cases"]
    assert [m["grade"] for m in cases["D_big_tall"]["matches"]] == ["exact", "same_class"]
    assert cases["D_big_tall"]["lines"][0].endswith("compare on D_big2_tall, D_new_tall (+1)")
    assert "hardware loops" in cases["D_big_tall"]["lines"][1]
    assert [m["id"] for m in cases["D_big2_tall"]["matches"]] == ["E1-exact-tall", "E2-tall-loops"]
    assert [(m["id"], m["grade"]) for m in cases["D_wide"]["matches"]] == [
        ("E5-wide", "same_class"),
        ("E1-exact-tall", "technique"),
    ]
    assert [m["grade"] for m in cases["D_thin"]["matches"]] == ["technique", "technique"]
    assert [(m["id"], m["grade"]) for m in cases["C_pass"]["matches"]] == [("E3-conv", "exact")]
    assert "A_add" not in cases  # no entry of its kind: no line
    assert all(len(record["lines"]) <= M.PER_CASE for record in cases.values())


def test_reference_lines_follow_their_step_within_the_block_budget(workspace, monkeypatch, tmp_path):
    monkeypatch.setenv(M.ENV, json.dumps(_pin(tmp_path)))
    monkeypatch.setattr(R, "_edge", lambda: EDGE)
    families = R.build(workspace, edge=EDGE)
    brief = {}
    assert M.attach(brief, families, workspace=workspace) is not None
    assert brief[M.KEY]["schema"] == M.FEEDBACK_SCHEMA and "D_big_tall" in brief[M.KEY]["cases"]
    found = N.block({}, [], None, workspace=workspace, families=families)
    lines = found["lines"]
    first = next(i for i, text in enumerate(lines) if text.startswith("1. D_big_tall: "))
    assert lines[first + 1].startswith("   reference experiment: ")
    assert "via merlin." in lines[first + 1]
    assert "resnet50_listing/resnet50.txt:1200-1490 matmul 64x32x8 (exact)" in lines[first + 1]
    assert lines[first + 2].startswith("   reference: ") and "hardware loops" in lines[first + 2]
    assert not any("C_fail" in text and "reference:" in text for text in lines)
    assert sum(len(text.encode()) + 1 for text in lines) <= N.BLOCK_BYTES
    # No pinned map, no reference anywhere.
    monkeypatch.delenv(M.ENV)
    families = R.build(workspace, edge=EDGE)
    assert M.attach({}, families, workspace=workspace) is None
    assert not any(
        "reference:" in text for text in N.block({}, [], None, workspace=workspace, families=families)["lines"]
    )


def test_a_changed_map_is_never_used(monkeypatch, tmp_path):
    pin = _pin(tmp_path)
    monkeypatch.setenv(M.ENV, json.dumps(pin))
    assert M.selected()["entries"][0]["id"] == "E1-exact-tall"
    changed = copy.deepcopy(MAP)
    changed["entries"] = changed["entries"][:1]
    with open(pin["path"], "w") as handle:
        handle.write(json.dumps(changed))
    assert M.selected() is None


def test_staging_serves_identical_read_only_bytes_and_one_task_section(monkeypatch, tmp_path):
    workspace, run = tmp_path / "ws", tmp_path / "run"
    for root in (workspace, run):
        root.mkdir()
        (root / "TASK.md").write_text("# Task\n")
    monkeypatch.delenv(M.ENV, raising=False)
    assert M.stage(workspace, run) is None and not (workspace / M.FILENAME).exists()
    pin = _pin(tmp_path)
    monkeypatch.setenv(M.ENV, json.dumps(pin))
    record = M.stage(workspace, run)
    assert record == {"path": M.FILENAME, "sha256": pin["sha256"], "source": pin["path"]}
    for root in (workspace, run):
        staged = root / M.FILENAME
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == pin["sha256"]
        assert staged.stat().st_mode & 0o777 == 0o444
        text = (root / "TASK.md").read_text()
        assert text.endswith(M.TASK_SECTION) and text.count("REFERENCE_MAP.json") == 1
    with pytest.raises(ValueError, match="already staged"):
        M.stage(workspace, run)


def test_withheld_tasks_keep_the_analysis_endpoint_and_point_at_the_map(monkeypatch, tmp_path):
    from merlin_experiments.phase1.feedback import reference_analysis as analysis
    from merlin_experiments.phase1.feedback import reference_experiments as reference
    from test_reference_experiment_feedback import _fixture

    fixture = _fixture(tmp_path)
    (tmp_path / "reference-kernels").mkdir()
    (tmp_path / "reference-kernels" / "kernel.py").write_text("def k(x):\n    for i in range(2):\n        x(i)\n")
    workspace, run = tmp_path, tmp_path / "run"
    run.mkdir()
    for root in (workspace, run):
        (root / "TASK.md").write_text("# Task\n")
    monkeypatch.setenv(M.ENV, json.dumps(_pin(tmp_path)))
    M.stage(workspace, run)
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        assert reference.child_arguments()[-2:] == ("--reference-experiments-selection-tasks", "off")
        reference.stage_guidance(workspace, run)
        assert reference.project(current_compiler_sha256=fixture["compiler"]) is None
        assert reference.pending_requests(fixture["compiler"]) == ()
        document = {"next_steps": {"lines": ["keep"]}}
        assert reference.attach(document, current_compiler_sha256=fixture["compiler"]) is None
        assert document == {"next_steps": {"lines": ["keep"]}}
        lifted = analysis.handle(
            {
                "action": analysis.ACTION,
                "reference_analysis": {"operation": "lift", "path": "reference-kernels/kernel.py"},
            },
            workspace=workspace,
        )
        assert lifted["status"] == "analyzed"
    for root in (workspace, run):
        assert not (root / "REFERENCE_EXPERIMENTS.json").exists()
        text = (root / "TASK.md").read_text()
        section = text.split("## Use selected kernel reference evidence", 1)[1]
        assert "REFERENCE_EXPERIMENTS.json" not in text and "Use these tasks" not in text
        assert "`REFERENCE_MAP.json`" in section and "--reference-analysis inventory" in section
        assert text.index("## Reference map") < text.index("## Use selected kernel reference evidence")
    # A served selection keeps the original section and file.
    other, archive = tmp_path / "served", tmp_path / "served-run"
    for root in (other, archive):
        root.mkdir()
        (root / "TASK.md").write_text("# Task\n")
    with reference.invocation(fixture["path"], fixture["sha"]):
        reference.stage_guidance(other, archive)
    assert (other / "REFERENCE_EXPERIMENTS.json").is_file()
    assert "Read `REFERENCE_EXPERIMENTS.json`" in (other / "TASK.md").read_text()
    with pytest.raises(ValueError, match="explicit selection"):
        with reference.invocation(tasks=False):
            pass


def test_a_pointwise_convolution_matches_the_same_matrix_product():
    entry = {"operation": {"kind": "matmul", "dims": {"M": 3136, "K": 64, "N": 256}}}
    case = {"kind": "conv2d", "dims": {"N": 1, "C": 64, "H": 56, "W": 56, "K": 256, "R": 1, "S": 1,
                                      "stride": 1, "pad": 0}}
    assert M.grade(entry, case, 16) == "same_product"
    case["dims"]["K"] = 128
    assert M.grade(entry, case, 16) is None


def test_connected_cases_match_by_their_largest_product():
    sibling = {"operation": {"op": "sibling_feed", "attributes": {"M": 1, "K": 2048, "H": 5728, "N": 2048}}}
    assert M.case_operation(sibling)["dims"] == {"M": 1, "K": 2048, "N": 5728}
    program = {"operation": {"op": "component_program", "attributes": {"program": {
        "inputs": [{"name": "X", "shape": [64, 32]}, {"name": "W0", "shape": [32, 48]},
                   {"name": "W1", "shape": [48, 16]}],
        "nodes": [{"name": "p0", "op": "matmul", "inputs": ["X", "W0"]},
                  {"name": "p1", "op": "matmul", "inputs": ["p0", "W1"]}]}}}}
    assert M.case_operation(program)["dims"] == {"M": 64, "K": 32, "N": 48}


def test_ties_prefer_the_reference_closest_in_size():
    entries = [{"id": "tiny", "operation": {"kind": "matmul", "dims": {"M": 16, "K": 16, "N": 64}}},
               {"id": "near", "operation": {"kind": "matmul", "dims": {"M": 16, "K": 2048, "N": 32000}}}]
    case = {"kind": "matmul", "dims": {"M": 32, "K": 2048, "N": 30000}, "shape": None}
    assert [entry["id"] for entry, _ in M.matches({"entries": entries}, case, 16)][0] == "near"
