"""Shape probes leave the capsule-feedback critical path without weakening the shape gate.

* An identical candidate (and identical contract, facts, tools and grader code) reuses its sweep; any
  changed byte, mode bit or input re-probes, and an emit error is never reused.
* A grade that is not gradeable skips the sweep and can never pass.
* A background grade publishes its capsule rows first, with ``all_pass`` unknown and shape coverage
  pending, and only the final verdict may carry ``all_pass: true``.
"""

import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import loop_grading as G
from merlin_experiments.phase1.feedback import shape_coverage as SC

from merlin.targetgen import lowering_coverage as LC


def _context(root):
    descriptor = root / "descriptor.yaml"
    descriptor.write_text("target: fixture\n")
    return InvocationContext(root, descriptor, root, "fixture", root / "runs", root / "reports", root / "bundles", ())


def _service(snapshot="snapshot-a"):
    return SimpleNamespace(bundle_bytes=b"bundle", snapshot_digest=snapshot, profile={"schema": "fixture"})


@pytest.fixture()
def emit(monkeypatch):
    """A fixture backend whose outcome per corner is selected by the test; counts real probes."""
    monkeypatch.setattr(
        LC,
        "_binding",
        lambda t: type(
            "B",
            (),
            {
                "operand_dtype": "int8",
                "accum_dtype": "int32",
                "mlir_dtype": staticmethod(lambda tok: {"int8": "i8"}.get(tok, "i32")),
            },
        )(),
    )
    monkeypatch.setattr(LC, "tile_edge", lambda t: 16)
    state = SimpleNamespace(calls=0, outcomes={})

    def probe(
        package,
        *,
        target,
        m,
        k,
        n,
        operand_mlir,
        accum_mlir,
        contract=None,
        timeout=300,
        additional_forbidden=(),
        **kwargs,
    ):
        state.calls += 1
        outcome = state.outcomes.get((m, k, n), "lowered")
        return outcome, None if outcome == "lowered" else f"{outcome} fixture", 10 if outcome == "lowered" else 0

    monkeypatch.setattr(LC, "probe_shape", probe)
    monkeypatch.setenv("MERLIN_SHAPE_COVERAGE_CACHE", "1")
    for name in ("MERLIN_TARGET_EXPERIMENT", "MERLIN_TARGET_CONTRACT", "MERLIN_RTL_FACTS"):
        monkeypatch.delenv(name, raising=False)
    return state


def _tree(root, files):
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _attach(
    tmp_path, cand, *, key, contract=None, timeout=60, forbidden=(), service=None, all_pass=True, before_probe=None
):
    verdict = {"all_pass": all_pass, "n_passed": 1, "n_capsules": 1}
    G._attach_shape_generalization(
        verdict,
        cand,
        tmp_path / "run",
        0,
        timeout=timeout,
        artifact_key=key,
        context=_context(tmp_path),
        contract=contract or tmp_path / "contract",
        additional_forbidden=forbidden,
        compiler_profile_service=service or _service(),
        before_probe=before_probe,
    )
    return verdict, json.loads((tmp_path / "run/qa_history" / f"shape_coverage_round_{key}.json").read_text())


@pytest.fixture()
def inputs(tmp_path):
    _tree(tmp_path / "contract", {"schemas/command_buffer.schema.json": "{}"})
    (tmp_path / "run").mkdir()
    return _tree(tmp_path / "cand_a" / "submission", {"manifest.yaml": "language: python\n", "tool.py": "A = 1\n"})


def test_identical_candidate_is_never_reprobed_and_reports_identically(tmp_path, emit, inputs):
    first, first_record = _attach(tmp_path, inputs, key="t1")
    assert emit.calls == 8 and first_record["cache"] == {
        "hit": False,
        "key": first_record["cache"]["key"],
        "stored": True,
    }
    second_cand = _tree(
        tmp_path / "cand_b" / "submission", {"manifest.yaml": "language: python\n", "tool.py": "A = 1\n"}
    )
    second, second_record = _attach(tmp_path, second_cand, key="t2")
    assert emit.calls == 8, "byte-identical inputs reuse the stored sweep"
    assert second_record["cache"]["hit"] is True and second_record["package"] == str(second_cand)
    assert second_record["cache"]["probed_package"] == str(inputs)
    assert second == first and second["all_pass"] is True
    keep = {"corners", "emitted_work", "all_covered", "tail_cases_uncovered", "multi_tile_axes_uncovered"}
    assert {k: first_record[k] for k in keep} == {k: second_record[k] for k in keep}


@pytest.mark.parametrize(
    "change",
    [
        "byte",
        "new_file",
        "mode",
        "build_product",
        "contract",
        "timeout",
        "forbidden",
        "executor",
        "plan",
        "selected_facts",
    ],
)
def test_any_changed_input_reprobes(tmp_path, emit, inputs, monkeypatch, change):
    _attach(tmp_path, inputs, key="t1")
    assert emit.calls == 8
    cand = _tree(tmp_path / "cand_b" / "submission", {"manifest.yaml": "language: python\n", "tool.py": "A = 1\n"})
    options = {}
    if change == "byte":
        (cand / "tool.py").write_text("A = 2\n")
    elif change == "new_file":
        (cand / "extra.py").write_text("")
    elif change == "mode":
        (cand / "tool.py").chmod(0o755)
    elif change == "build_product":
        _tree(cand, {"build/tool": "binary"})
    elif change == "contract":
        (tmp_path / "contract/schemas/command_buffer.schema.json").write_text('{"type": "object"}')
    elif change == "timeout":
        options["timeout"] = 30
    elif change == "forbidden":
        options["forbidden"] = ("outputs_match",)
    elif change == "executor":
        options["service"] = _service("snapshot-b")
    elif change == "plan":
        monkeypatch.setattr(LC, "tile_edge", lambda t: 32)
    elif change == "selected_facts":
        facts = tmp_path / "facts.json"
        facts.write_text("{}")
        monkeypatch.setenv("MERLIN_RTL_FACTS", str(facts))
    _, record = _attach(tmp_path, cand, key="t2", **options)
    assert emit.calls == 16 and record["cache"]["hit"] is False


def test_an_emit_error_is_never_reused(tmp_path, emit, inputs):
    emit.outcomes[(16, 16, 32)] = "error"
    verdict, record = _attach(tmp_path, inputs, key="t1")
    assert record["cache"]["stored"] is False and verdict["all_pass"] is False
    emit.outcomes.clear()
    verdict, record = _attach(tmp_path, inputs, key="t2")
    assert emit.calls == 16 and record["cache"]["hit"] is False
    assert verdict["all_pass"] is True


def test_a_tampered_or_corrupt_entry_is_a_miss_never_coverage(tmp_path, emit, inputs):
    emit.outcomes[(16, 16, 17)] = "declined"
    _, record = _attach(tmp_path, inputs, key="t1")
    entry = tmp_path / "run/_qa_work/shape_coverage_cache" / f"{record['cache']['key']}.json"
    document = json.loads(entry.read_text())
    document["result"]["all_covered"] = True  # a forged claim without matching key material
    document["material"]["candidate"] = "0" * 64
    entry.write_text(json.dumps(document))
    verdict, record = _attach(tmp_path, inputs, key="t2")
    assert emit.calls == 16 and record["cache"]["hit"] is False
    assert verdict["all_pass"] is False and verdict["shape_coverage"]["tail_cases_uncovered"] == ["n_tail"]
    entry.write_text("{not json")
    verdict, record = _attach(tmp_path, inputs, key="t3")
    assert emit.calls == 24 and verdict["all_pass"] is False


def test_cache_requires_selection_and_the_original_profile(tmp_path, emit, inputs, monkeypatch):
    monkeypatch.setenv("MERLIN_SHAPE_COVERAGE_CACHE", "0")
    _attach(tmp_path, inputs, key="t1")
    _attach(tmp_path, inputs, key="t2")
    assert emit.calls == 16
    monkeypatch.setenv("MERLIN_SHAPE_COVERAGE_CACHE", "1")
    verdict = {"all_pass": True}
    for key in ("t3", "t4"):
        G._attach_shape_generalization(
            verdict,
            inputs,
            tmp_path / "run",
            0,
            timeout=60,
            artifact_key=key,
            context=_context(tmp_path),
            contract=tmp_path / "contract",
        )
    assert emit.calls == 32, "no profile pins the tool closure, so nothing is reused"
    monkeypatch.setenv("MERLIN_SHAPE_COVERAGE_CACHE", "yes")
    with pytest.raises(ValueError, match="MERLIN_SHAPE_COVERAGE_CACHE"):
        _attach(tmp_path, inputs, key="t5")


def test_before_probe_runs_only_when_probes_run(tmp_path, emit, inputs):
    published = []
    _attach(tmp_path, inputs, key="t1", before_probe=lambda: published.append(emit.calls))
    assert published == [0], "pending feedback precedes the first probe"
    _attach(tmp_path, inputs, key="t2", before_probe=lambda: published.append("hit"))
    assert published == [0], "a reused sweep publishes the final verdict directly"


def test_worker_selection_reaches_the_sweep(tmp_path, emit, inputs, monkeypatch):
    seen = []
    real = LC.sweep
    monkeypatch.setattr(LC, "sweep", lambda *a, **k: seen.append((k.get("workers", 1), "runner" in k)) or real(*a, **k))
    monkeypatch.setenv("MERLIN_SHAPE_COVERAGE_CACHE", "0")
    monkeypatch.setenv("MERLIN_SHAPE_COVERAGE_WORKERS", "8")
    verdict = {"all_pass": True}
    G._attach_shape_generalization(
        verdict,
        inputs,
        tmp_path / "run",
        0,
        timeout=60,
        artifact_key="t1",
        context=_context(tmp_path),
        contract=tmp_path / "contract",
    )
    record = json.loads((tmp_path / "run/qa_history/shape_coverage_round_t1.json").read_text())
    assert seen == [(8, False)], "without the original profile, corners run on threads in this process"
    assert record["timing"]["workers"] == 8 and verdict["all_pass"] is True
    for raw in ("0", "17", "two"):
        monkeypatch.setenv("MERLIN_SHAPE_COVERAGE_WORKERS", raw)
        with pytest.raises(ValueError, match="MERLIN_SHAPE_COVERAGE_WORKERS"):
            _attach(tmp_path, inputs, key="t2")


# --------------------------------------------------------------- profile-owned corners in child processes

_FAKE_CHILD = r"""
import json, sys
spec = json.load(open(sys.argv[1]))
assert spec["executor"]["snapshot_sha256"] == "snapshot-a" and spec["workspace"].endswith("workspace")
rows = []
for name, (m, k, n) in spec["probes"]:
    if name == "crash":
        sys.exit(3)
    rows.append(["declined", "declined fixture", 0] if (m, k, n) == (16, 16, 17) else ["lowered", None, 10])
probes = spec["probes"] if "mismatch" not in spec["candidate"] else spec["probes"][::-1]
json.dump({"schema": spec["schema"], "probes": probes, "outcomes": rows,
           "corner_wall_s": {name: 0.01 for name, _ in spec["probes"]}}, open(sys.argv[2], "w"))
"""


def _profile_service(tmp_path):
    return SimpleNamespace(
        bundle_bytes=b"bundle",
        snapshot_digest="snapshot-a",
        profile={"schema": "fixture"},
        workspace=tmp_path / "workspace",
        bundle_manifest=tmp_path / "bundle.yaml",
        repo=tmp_path,
    )


@pytest.mark.parametrize("workers", [2, 3, 8])
def test_profile_corners_run_in_child_processes_with_serial_results(tmp_path, emit, inputs, monkeypatch, workers):
    import sys

    launched = []

    def command(spec, out):
        launched.append(json.loads(spec.read_text()))
        return [sys.executable, "-c", _FAKE_CHILD, str(spec), str(out)]

    monkeypatch.setattr(SC, "_child_command", command)
    monkeypatch.setenv("MERLIN_SHAPE_COVERAGE_CACHE", "0")
    emit.outcomes[(16, 16, 17)] = "declined"
    serial = LC.sweep(inputs, target="fixture", contract=str(tmp_path / "contract"))
    monkeypatch.setenv("MERLIN_SHAPE_COVERAGE_WORKERS", str(workers))
    verdict, record = _attach(tmp_path, inputs, key="t1", service=_profile_service(tmp_path))
    drop = {"timing", "package"}
    assert {k: v for k, v in record.items() if k not in drop} == {k: v for k, v in serial.items() if k not in drop}
    assert len(launched) == min(workers, 8)
    assert sorted(name for spec in launched for name, _ in spec["probes"]) == sorted([*LC.CORNERS, *LC.TAIL_CORNERS])
    assert verdict["all_pass"] is False and verdict["shape_coverage"]["tail_cases_uncovered"] == ["n_tail"]
    assert not list((tmp_path / "run/_qa_work/shape_probe_children").iterdir()), "private specs are removed"


def test_a_failed_or_mismatched_child_is_an_error_never_coverage(tmp_path, emit, inputs, monkeypatch):
    import sys

    def command(spec, out):
        document = json.loads(spec.read_text())
        if any(name == "tile" for name, _ in document["probes"]):
            document["probes"] = [["crash", [1, 1, 1]], *document["probes"]]
            spec.write_text(json.dumps(document))
        return [sys.executable, "-c", _FAKE_CHILD, str(spec), str(out)]

    monkeypatch.setattr(SC, "_child_command", command)
    monkeypatch.setenv("MERLIN_SHAPE_COVERAGE_WORKERS", "4")
    verdict, record = _attach(tmp_path, inputs, key="t1", service=_profile_service(tmp_path))
    outcomes = {c["corner"]: c["outcome"] for c in record["corners"]}
    assert outcomes["tile"] == outcomes["sub_tile"] == "error"
    assert record["baseline_tile_lowered"] is False and verdict["all_pass"] is False
    assert record["cache"]["stored"] is False, "an errored sweep is never reusable"
    mismatched = _tree(tmp_path / "mismatch" / "submission", {"manifest.yaml": "language: python\n"})
    monkeypatch.setattr(
        SC, "_child_command", lambda spec, out: [sys.executable, "-c", _FAKE_CHILD, str(spec), str(out)]
    )
    verdict, record = _attach(tmp_path, mismatched, key="t2", service=_profile_service(tmp_path))
    assert {c["outcome"] for c in record["corners"]} == {"error"} and verdict["all_pass"] is False


# --------------------------------------------------------------- the grade around the sweep


def _grade_setup(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import promotion

    ws, run = tmp_path / "workspace", tmp_path / "run"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    inputs = G.GradingInputs(_context(tmp_path), "raw_baseline", tmp_path / "public", (), None)
    monkeypatch.setenv("MERLIN_BACKGROUND_CODEGEN_SCALABILITY", "0")
    monkeypatch.setattr(G, "_attach_codegen_scalability", lambda verdict, *a, **k: None)
    monkeypatch.setattr(G, "_write_stage_ledger", lambda *a, **k: None)
    monkeypatch.setattr(G, "_record_plateau", lambda *a: None)
    monkeypatch.setattr(G.FL, "record_channel_health", lambda *a: None)
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **k: (None, None, None))
    return ws, run, inputs


@pytest.mark.parametrize(
    "verdict",
    [
        {"gradeable": False, "all_pass": False, "n_capsules": 0, "n_passed": 0, "per_capsule": []},
        {"gradeable": False, "all_pass": None, "n_capsules": 1, "n_passed": 1, "per_capsule": []},
    ],
)
def test_a_grade_that_is_not_gradeable_skips_the_sweep_and_cannot_pass(tmp_path, monkeypatch, verdict):
    ws, run, inputs = _grade_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(LC, "sweep", lambda *a, **k: pytest.fail("probed a grade that cannot converge"))
    result = G.grade(ws, run, 1, False, 5, inputs=inputs, qa_runner=lambda *a, **k: dict(verdict))
    assert result["shape_coverage"]["ran"] is False and result["shape_coverage"]["skipped"] is True
    assert "NOT a pass" in result["shape_coverage"]["note"]
    assert result["all_pass"] is not True


def test_a_deferred_background_grade_skips_the_sweep(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import background_admission

    ws, run, inputs = _grade_setup(tmp_path, monkeypatch)
    # Only a superseded source defers a background grade; the author's own
    # same-source requests now outrank it in the CPU pool instead.
    monkeypatch.setenv("MERLIN_BACKGROUND_FINISH_SNAPSHOT", "0")
    digests = iter(["a" * 64])
    monkeypatch.setattr(background_admission, "source_digest", lambda ws: next(digests, "b" * 64))
    monkeypatch.setattr(LC, "sweep", lambda *a, **k: pytest.fail("probed a deferred background grade"))
    result = G.grade(
        ws,
        run,
        1,
        False,
        5,
        inputs=inputs,
        advisory_background=True,
        qa_runner=lambda *a, **k: {"all_pass": True, "n_capsules": 1, "n_passed": 1, "per_capsule": []},
    )
    assert result["all_pass"] is None and result["gradeable"] is False
    assert result["background_grade"]["status"] == "deferred"
    assert result["shape_coverage"]["ran"] is False


def test_a_structure_only_pass_still_needs_its_shape_probes(tmp_path, monkeypatch):
    ws, run, inputs = _grade_setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(LC, "sweep", lambda *a, **k: calls.append(1) or {"all_covered": False, "corners": []})
    result = G.grade(
        ws,
        run,
        1,
        True,
        5,
        inputs=inputs,
        qa_runner=lambda *a, **k: {
            "gradeable": False,
            "all_pass": True,
            "n_capsules": 1,
            "n_passed": 1,
            "per_capsule": [],
        },
    )
    assert calls == [1] and result["all_pass"] is False


@pytest.mark.parametrize("covered", [True, False])
def test_capsule_rows_publish_before_shape_with_all_pass_unknown(tmp_path, monkeypatch, covered):
    ws, run, inputs = _grade_setup(tmp_path, monkeypatch)
    monkeypatch.setenv("MERLIN_BACKGROUND_PROGRESS", "1")
    row = {"capsule": "case", "status": "pass", "tiers": {"L2": "pass"}}
    seen = {}

    def qa_runner(*args, **kwargs):
        kwargs["on_capsule_complete"](row)
        return {"all_pass": True, "gradeable": True, "n_capsules": 1, "n_passed": 1, "per_capsule": [row]}

    def sweep(*args, **kwargs):
        seen["verdict"] = json.loads((ws / "qa/verdict.json").read_text())
        seen["progress"] = json.loads((ws / "qa/progress.json").read_text())
        return {
            "all_covered": covered,
            "baseline_tile_lowered": True,
            "corners": [],
            "multi_tile_axes_uncovered": [] if covered else ["m"],
        }

    monkeypatch.setattr(LC, "sweep", sweep)
    result = G.grade(ws, run, 1, False, 5, inputs=inputs, advisory_background=True, qa_runner=qa_runner)
    pending = seen["verdict"]
    assert pending["per_capsule"] == [row] and pending["n_passed"] == 1
    assert pending["all_pass"] is None and pending["shape_coverage"]["status"] == "pending"
    assert seen["progress"]["state"] == "shape_coverage_pending" and seen["progress"]["all_pass"] is None
    final = json.loads((ws / "qa/verdict.json").read_text())
    assert final["all_pass"] is covered and result["all_pass"] is covered
    assert final["shape_coverage"]["ran"] is True
    assert json.loads((ws / "qa/progress.json").read_text())["state"] == "complete"


def test_a_failing_grade_stays_failing_while_shape_is_pending(tmp_path, monkeypatch):
    ws, run, inputs = _grade_setup(tmp_path, monkeypatch)
    monkeypatch.setenv("MERLIN_BACKGROUND_PROGRESS", "1")
    seen = []
    monkeypatch.setattr(
        LC,
        "sweep",
        lambda *a, **k: (
            seen.append(json.loads((ws / "qa/verdict.json").read_text()))
            or {"all_covered": True, "baseline_tile_lowered": True, "corners": []}
        ),
    )
    result = G.grade(
        ws,
        run,
        1,
        False,
        5,
        inputs=inputs,
        advisory_background=True,
        qa_runner=lambda *a, **k: {
            "all_pass": False,
            "gradeable": True,
            "n_capsules": 2,
            "n_passed": 1,
            "per_capsule": [],
        },
    )
    assert seen[0]["all_pass"] is False and result["all_pass"] is False


def test_grades_order_capsules_from_this_runs_history(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_scheduling

    ws, run, inputs = _grade_setup(tmp_path, monkeypatch)
    for root in ("runs_r0000_t000001", "fruns_900", "selfchecks/selfcheck_ab", "cand_r0000_t000001"):
        (run / "_qa_work" / root).mkdir(parents=True)
    seen = []

    def qa_runner(*args, **kwargs):
        seen.append(capsule_scheduling.selected_history())
        return {"all_pass": False, "gradeable": True, "n_capsules": 1, "n_passed": 0, "per_capsule": []}

    monkeypatch.setattr(LC, "sweep", lambda *a, **k: {"all_covered": True, "corners": []})
    G.grade(ws, run, 1, False, 5, inputs=inputs, qa_runner=qa_runner)
    names = sorted(p.relative_to(run / "_qa_work").as_posix() for p in seen[0])
    assert names == ["fruns_900", "runs_r0000_t000001", "selfchecks/selfcheck_ab"]
    assert capsule_scheduling.selected_history() == ()
