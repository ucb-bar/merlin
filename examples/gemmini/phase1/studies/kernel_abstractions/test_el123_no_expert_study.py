"""Operator rendering and preparation admission tests; never call a provider."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_spec = importlib.util.spec_from_file_location("el123_no_expert_study", Path(__file__).with_name("el123_no_expert_study.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
REVISION = "0123456789abcdef0123456789abcdef01234567"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def source(tmp_path):
    task = tmp_path / "operator-task"
    root = task / ".tmp/xs-el123-20261028-r2"
    policy = write(root / "policy.json", {"duration_s": 15300, "drain_s": 900, "study_protocol": {"author_s": 14400}})
    performance = {"MERLIN_SPIKE": "/pinned/spike", "MERLIN_JOINT_PERFORMANCE_SELECTION": "${PERFORMANCE_SELECTION_PIN}",
                   "MERLIN_BACKGROUND_PROGRESS": "1", "MERLIN_SHAPE_COVERAGE_CACHE": "1"}
    arms = []
    for arm in M.ARMS:
        doc = {"id": "old-" + arm, "schema": "merlin.phase1.fresh_campaign_spec.v1",
               "variables": {"TASK": str(task), "REPO": "${TASK}/merlin", "PYTHON": "${WORKSPACE}/runtime/bin/python", "WORKSPACE": str(tmp_path),
                             "STUDY_ROOT": str(root), "STUDY_ARM": arm,
                             "MATCHED_RAW_REFERENCES": "/old/fresh8-reference-kernels-only",
                             "REFERENCE_ANALYSIS_DEPS": "/old/reference-analysis/python",
                             "FRESH7_REFERENCE_MAP": "/old/REFERENCE_MAP.json"},
               "run": {"run_id": "old-" + arm},
               "inputs": {"public_references": "${MATCHED_RAW_REFERENCES}", "forbidden_reference_prefixes": ["measured-exo/"],
                          "environment": {"PYTHONPATH": "${FRAMEWORK}/src:${REFERENCE_ANALYSIS_DEPS}"}},
               "plan": {"author_profile": "xs-el-v1/" + arm,
                        "argv": ["python", "--fresh-backend", "--model", "gpt-6.1-sol", "--effort", "high",
                                 "--author-max-subagents", "8", "--codex-max-subagents", "8"],
                        "environment": {**performance, "MERLIN_REFERENCE_MAP": "old-map", "MERLIN_CCA_CAPTURE": "old-capture",
                                        "PYTHONPATH": "${FRAMEWORK}/src:${FRAMEWORK}/packages/merlin-timing/src:${REFERENCE_ANALYSIS_DEPS}"},
                        "public_reference_manifest": "/old/public-references/reference-manifest.json",
                        "reference_guidance": "off", "reference_map": "${FRESH7_REFERENCE_MAP}",
                        "reference_experiments": {"path": "/old/reference-experiments/packet-shards.json"},
                        "policy": {"base": M.pin(policy)},
                        "record": {"compiler_access": "selected_library" if arm == "el1" else "full_compiler", "limits": {}}}}
        p = root / (arm + ".yaml")
        p.write_text(yaml.safe_dump(doc, sort_keys=False))
        arms.append({"arm": arm, "spec": M.pin(p)})
    study = write(root / "study.json", {"operator_task": str(task), "arms": arms})
    return study, task / ".tmp/xs-el123-clean-new"


def rendered(source):
    original, dest = source
    receipt = M.render(original, dest, revision=REVISION, export_id=REVISION[:12])
    return dest, json.loads((dest / "study.json").read_text()), receipt


def update_spec(study, arm, change):
    row = next(r for r in study["arms"] if r["arm"] == arm)
    path = Path(row["spec"]["path"])
    doc = yaml.safe_load(path.read_text())
    change(doc)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    row["spec"] = M.pin(path)


def test_render_removes_all_reference_routes_and_keeps_full_spike(source):
    original, dest = source
    before = original.read_bytes(), [Path(r["spec"]["path"]).read_bytes() for r in json.loads(original.read_text())["arms"]]
    _, study, receipt = rendered(source)
    assert receipt["all_passed"] and receipt["native_qualification"] == "pending"
    assert receipt["author_jobs_launched"] == receipt["cpu_reservations_created"] == 0
    assert not study["launch_authorized"] and study["seed"] is None
    assert [r["supplied_module_count"] for r in receipt["arms"]] == [0, 12, 34]
    cpus = [c for row in study["arms"] for c in row["cpus"]]
    assert len(cpus) == len(set(cpus)) == 45
    for row in study["arms"]:
        doc = yaml.safe_load(Path(row["spec"]["path"]).read_text())
        assert doc["inputs"]["public_references"] is None
        assert doc["plan"]["public_reference_manifest"] is None
        assert doc["plan"]["reference_experiments"] is None
        assert doc["plan"]["reference_map"] is None
        assert doc["plan"]["reference_guidance"] == doc["plan"]["reference_experiment_tasks"] == "off"
        assert doc["plan"]["environment"]["MERLIN_SPIKE"] == "/pinned/spike"
        assert doc["plan"]["environment"]["MERLIN_SHAPE_COVERAGE_CACHE"] == "1"
        assert doc["plan"]["environment"]["PYTHONPATH"] == "${FRAMEWORK}/src:${FRAMEWORK}/packages/merlin-timing/src"
        assert not any(k.startswith(("MERLIN_REFERENCE", "MERLIN_CCA")) for k in doc["plan"]["environment"])
        assert doc["variables"]["FRAMEWORK"] == "${REPO}/out/build/run-frameworks/" + REVISION[:12]
        assert doc["run"]["preparation_cpus"] == "0,1,2,3"
        assert doc["run"]["workers"] == doc["run"]["sim_max_jobs"] == 15
    assert before == (original.read_bytes(), [Path(r["spec"]["path"]).read_bytes() for r in json.loads(original.read_text())["arms"]])
    assert not (dest / "arms").exists()  # no homes, reservations or launches


@pytest.mark.parametrize("change,match", [
    (lambda d: d["inputs"].update(public_references="bank"), "expert material"),
    (lambda d: d["plan"].update(reference_map={"path": "map"}), "reference map"),
    (lambda d: d["run"].update(workers=2), "budget"),
    (lambda d: d["plan"]["environment"].update(MERLIN_SPIKE="different"), "Spike"),
    (lambda d: d["plan"]["argv"].append("--resume"), "inherit"),
    (lambda d: d["inputs"].update(unknown_new_treatment=True), "unexplained difference"),
])
def test_static_audit_fails_reintroduced_exposure_or_unmatched_treatment(source, change, match):
    _, study, _ = rendered(source)
    update_spec(study, "el2", change)
    with pytest.raises(ValueError, match=match):
        M.audit(study)


def test_exact_source_pin_and_fresh_destination_are_required(source):
    original, dest = source
    with pytest.raises(ValueError, match="exact source"):
        M.render(original, dest, revision="moving-head", export_id="moving")
    assert not dest.exists()
    rendered(source)
    with pytest.raises(ValueError, match="fresh directory"):
        M.render(original, dest, revision=REVISION, export_id=REVISION[:12])


def test_gate_refuses_wrong_export_and_has_no_author_action(source, monkeypatch):
    dest, study, _ = rendered(source)
    marker = Path(study["operator_task"]) / "merlin/out/build/run-frameworks" / study["export_id"] / ".merlin-source-revision.json"
    write(marker, {"revision": "f" * 40})
    monkeypatch.setattr(M.subprocess, "run", lambda *_a, **_k: pytest.fail("no process may run"))
    with pytest.raises(ValueError, match="exact selected source"):
        M.gate(dest, "inputs")
    with pytest.raises(ValueError, match="non-author"):
        M.gate(dest, "launch")


def test_native_preparation_gates_use_only_preparation_cpus_and_exact_specs(source, monkeypatch):
    dest, study, _ = rendered(source)
    marker = Path(study["operator_task"]) / "merlin/out/build/run-frameworks" / study["export_id"] / ".merlin-source-revision.json"
    write(marker, {"revision": REVISION})
    calls = []
    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(M.subprocess, "run", execute)
    with pytest.raises(FileNotFoundError):
        M.gate(dest, "seal")
    assert not calls
    for action in M.GATES:
        result = M.gate(dest, action)
        assert result["all_passed"] == (action != "preflight")
        assert result["rendered_only"] == (action == "preflight")
    assert len(calls) == 15
    for argv, kwargs in calls:
        assert argv[:3] == ["taskset", "-c", "0,1,2,3"]
        assert "--export-id" in argv and argv[argv.index("--export-id") + 1] == REVISION[:12]
        assert str(dest) == kwargs["env"]["TMPDIR"]
        assert "REFERENCE_ANALYSIS" not in kwargs["env"]["PYTHONPATH"]
        assert "launch" not in argv and "--execute" not in argv
        assert "${" not in argv[3]
        assert argv[3].endswith("/runtime/bin/python")


def test_removing_full_spike_diagnostics_from_all_arms_is_not_a_permitted_common_change(source):
    _, study, _ = rendered(source)
    for arm in M.ARMS:
        update_spec(study, arm, lambda d: d["plan"]["environment"].pop("MERLIN_SHAPE_COVERAGE_CACHE"))
    with pytest.raises(ValueError, match="pinned source specification"):
        M.audit(study)


def test_real_template_style_nested_paths_expand_and_export_is_literal(source):
    _, study, _ = rendered(source)
    spec = yaml.safe_load(Path(study["arms"][0]["spec"]["path"]).read_text())
    assert M.expanded_path(spec, "REPO") == Path(study["operator_task"]) / "merlin"
    assert M.expanded_path(spec, "FRAMEWORK") == Path(study["operator_task"]) / "merlin/out/build/run-frameworks" / REVISION[:12]
    assert M.expanded_path(spec, "PYTHON").is_absolute()
    spec["variables"]["PYTHON"] = "${MISSING}/python"
    with pytest.raises(ValueError, match="unresolved"):
        M.expanded_path(spec, "PYTHON")


def test_actual_r2_python_template_is_resolved_before_subprocess():
    spec = {"variables": {"WORKSPACE": "/work/merlin",
                         "TASK": "${WORKSPACE}/tasks/el4-grader-throughput-20261024",
                         "REPO": "${TASK}/merlin",
                         "PYTHON": "${WORKSPACE}/tasks/gemmini-shared-phase1-20261021/.tmp/phase1-runtime-v1/venv/bin/python3.12"}}
    assert str(M.expanded_path(spec, "PYTHON")) == "/work/merlin/tasks/gemmini-shared-phase1-20261021/.tmp/phase1-runtime-v1/venv/bin/python3.12"
    assert str(M.expanded_path(spec, "REPO")) == "/work/merlin/tasks/el4-grader-throughput-20261024/merlin"
