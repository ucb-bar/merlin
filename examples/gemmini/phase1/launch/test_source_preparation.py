"""Selected private source admission must precede any paid author launch."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import private_full_models as M
from merlin_experiments.phase1.feedback import private_source_freeze as F
from merlin_experiments.phase1.feedback import private_source_preflight_parallel as P


@pytest.fixture
def operator():
    source = Path(__file__).with_name("launch_phase1.py")
    spec = importlib.util.spec_from_file_location("operator_source_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(operator, tmp_path, monkeypatch):
    identity = {
        "target": "fixture",
        "spec_sha256": "a" * 64,
        "programs": {"first": ["one", "two"], "second": ["one"]},
        "selected": {"source_revision": "b" * 40},
        "tool": "selected",
    }
    launch = SimpleNamespace(
        document={"inputs": {"private_full_model_spec": "spec", "descriptor": "descriptor"}}, path=Path
    )
    monkeypatch.setattr(operator, "_source_preflight_inputs", lambda _: (copy.deepcopy(identity), None))
    calls = []

    def preflight(spec, *, descriptor, out, target, native_result_format_selection, progress_path, cpus):
        calls.append((spec, descriptor, target, native_result_format_selection))
        out.mkdir()
        assert progress_path == out / P.PROGRESS
        progress_path.write_text(json.dumps({"stage": "complete"}))
        (out / "frozen-source").write_text("exact original source")
        report = {
            "mode": "source_preflight",
            "source_preflight_passed": True,
            "passed": False,
            "candidate_tree_sha256": None,
            "target": target,
            "scope": M.SOURCE_PREFLIGHT_SCOPE,
            "private_spec_sha256": identity["spec_sha256"],
            "required_models": list(identity["programs"]),
            "required_programs": copy.deepcopy(identity["programs"]),
            "models": [
                {"model": name, "status": "source_inputs_verified", "checks": {"source": {p: {} for p in programs}}}
                for name, programs in identity["programs"].items()
            ],
            "authored_source_freeze": {
                "schema": F.SCHEMA,
                "spec_sha256": identity["spec_sha256"],
                "record_sha256": "c" * 64,
                "root": str(out),
            },
        }
        (out / P.REPORT).write_text(json.dumps(report))
        return report

    monkeypatch.setattr(P, "preflight", preflight)
    return launch, identity, calls


def test_source_only_success_is_reused_without_candidate_or_retained_compiler(operator, prepared, tmp_path):
    launch, _, calls = prepared
    first = operator.source_preparation(launch, tmp_path)
    assert first["source_preflight_passed"] and first["candidate_qualified"] is False
    assert first["reused"] is False and len(calls) == 1
    second = operator.source_preparation(launch, tmp_path)
    assert second["reused"] is True and second["report"] == first["report"] and len(calls) == 1
    progress = Path(first["report"]["path"]).with_name(P.PROGRESS)
    assert json.loads(progress.read_text()) == {"stage": "complete"}


@pytest.mark.parametrize("mutation", ["false", "omitted_model", "omitted_program", "candidate_credit"])
def test_source_refusal_or_partial_roster_never_admits_authoring(operator, prepared, tmp_path, monkeypatch, mutation):
    launch, _, _ = prepared
    original = P.preflight

    def incomplete(*args, **kwargs):
        report = original(*args, **kwargs)
        if mutation == "false":
            report["source_preflight_passed"] = False
            report["models"][0]["status"] = "fail"
        elif mutation == "omitted_model":
            report["models"].pop()
        elif mutation == "omitted_program":
            report["models"][0]["checks"]["source"].pop("two")
        else:
            report["passed"] = True
        (kwargs["out"] / P.REPORT).write_text(json.dumps(report))
        return report

    monkeypatch.setattr(P, "preflight", incomplete)
    with pytest.raises(ValueError, match="source preflight refused"):
        operator.source_preparation(launch, tmp_path)
    assert list((tmp_path / "source-preflight").glob("*/" + P.REPORT))


def test_prepared_source_or_report_tampering_refuses_instead_of_reusing(operator, prepared, tmp_path):
    launch, _, calls = prepared
    first = operator.source_preparation(launch, tmp_path)
    Path(first["report"]["path"]).with_name("frozen-source").write_text("changed")
    with pytest.raises(ValueError, match="frozen source bytes changed"):
        operator.source_preparation(launch, tmp_path)
    assert len(calls) == 1


def test_changed_source_or_tool_identity_retains_old_receipt_and_rechecks(operator, prepared, tmp_path):
    launch, identity, calls = prepared
    first = operator.source_preparation(launch, tmp_path)
    identity["tool"] = "different selected tool"
    second = operator.source_preparation(launch, tmp_path)
    assert second["reused"] is False and first["report"] != second["report"] and len(calls) == 2
    assert Path(first["report"]["path"]).is_file()


def test_exact_input_identity_reopens_source_trees_tool_selection_and_native_policy(operator, tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import private_control_support as control
    from merlin_experiments.phase1.feedback import private_full_model_execution as execution

    from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256

    trees = {}
    for name in ("capture", "host_package", "recipe_derivation_root"):
        trees[name] = tmp_path / name
        trees[name].mkdir()
        (trees[name] / "original").write_text(name)
    host, catalog, native, descriptor = (
        tmp_path / name for name in ("host.yaml", "catalog.yaml", "native.json", "descriptor.yaml")
    )
    for path in (host, catalog, native, descriptor):
        path.write_text(path.name)
    row = {
        "id": "first",
        "board": "selected",
        **{name: str(path) for name, path in trees.items()},
        "capture_tree_sha256": strict_tree_sha256(trees["capture"])["sha256"],
        "host_package_tree_sha256": strict_tree_sha256(trees["host_package"])["sha256"],
        "host_capabilities": str(host),
        "host_capabilities_sha256": file_sha256(host),
        "board_catalog": str(catalog),
        "board_catalog_sha256": file_sha256(catalog),
    }
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump({"models": [row]}))
    launch = SimpleNamespace(
        document={
            "inputs": {"private_full_model_spec": str(spec), "descriptor": str(descriptor)},
            "identity": {"target": "fixture"},
        },
        path=Path,
        argv=("python", "--native-result-format-selection", str(native)),
    )
    monkeypatch.setattr(M, "requirements_for", lambda _: ("first",))
    monkeypatch.setattr(M, "program_requirements_for", lambda _: {"first": ("model",)})
    monkeypatch.setattr(execution, "gate_for", lambda *_a, **_k: None)
    monkeypatch.setattr(execution, "build_options", lambda _: pytest.fail("absent gate has no build options"))
    tool = {"compiler_sha256": "selected"}
    monkeypatch.setattr(control, "selected_build_observation", lambda *_: dict(tool))
    monkeypatch.setattr(
        operator, "operator_module", lambda _: SimpleNamespace(selected_identity=lambda _: {"source": "pinned"})
    )
    first, policy = operator._source_preflight_inputs(launch)
    assert policy == native and first["input_pins"][str(host)] == file_sha256(host)
    tool["compiler_sha256"] = "different"
    assert operator._source_preflight_inputs(launch)[0] != first
    host.write_text("changed admission")
    with pytest.raises(ValueError, match="source preparation input differs"):
        operator._source_preflight_inputs(launch)
    host.write_text("host.yaml")
    saved = tmp_path / "actual-native.json"
    native.rename(saved)
    native.symlink_to(saved)
    with pytest.raises(ValueError, match="result-format selection is absent or indirect"):
        operator._source_preflight_inputs(launch)


def test_input_change_during_source_analysis_never_publishes_reusable_success(
    operator, prepared, tmp_path, monkeypatch
):
    launch, identity, _ = prepared
    original = P.preflight

    def changed(*args, **kwargs):
        report = original(*args, **kwargs)
        identity["tool"] = "changed during analysis"
        return report

    monkeypatch.setattr(P, "preflight", changed)
    with pytest.raises(ValueError, match="inputs changed during preflight"):
        operator.source_preparation(launch, tmp_path)
    assert not list((tmp_path / "source-preflight").glob("*.json"))


def test_source_readiness_refusal_precedes_final_readiness_and_paid_run(operator, tmp_path, monkeypatch):
    bundle, prep = tmp_path / "bundle", tmp_path / "prep"
    bundle.mkdir()
    prep.mkdir()
    spec = tmp_path / "private.yaml"
    spec.write_text(yaml.safe_dump({"models": [{"id": "first"}], "numerics_bundle": {"path": str(bundle)}}))
    output = prep / "profile"
    doc = {
        "origin": {"seed_submission": None, "resume": False, "initial_submission": "empty"},
        "inputs": {"repo": str(tmp_path), "private_full_model_spec": str(spec)},
        "runtime_pins": {
            "private_bundle_manifest": {"path": str(bundle / "MANIFEST.json"), "sha256": "d"},
            "private_bundle_sums": {"path": str(bundle / "SHA256SUMS")},
        },
        "launcher": {"environment_file": str(output / "environment.sh"), "plan": str(output / "plan.json")},
        "cpu": {"author": {"cpus": list(range(6, 10))}, "compute_pool": {"cpus": list(range(10, 24))}},
    }
    launch = SimpleNamespace(
        document=doc,
        variables={"SOURCE_REVISION": "a"},
        environment={"MERLIN_MODEL_CHECK_BUNDLE": str(bundle)},
        path=Path,
    )
    monkeypatch.setattr(operator.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(operator.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(operator.B, "Bundle", lambda _: SimpleNamespace(manifest=lambda: {"models": {"first": {}}}))
    monkeypatch.setattr(operator.L, "render", lambda _, out: out.mkdir())
    monkeypatch.setattr(operator.L, "launch_plan", lambda *_: {})
    monkeypatch.setattr(operator.L, "verify", lambda _: SimpleNamespace(drifts=[]))
    monkeypatch.setattr(operator.L, "report_record", lambda *_: {"config": "fixture"})
    monkeypatch.setattr(operator.S, "verify", lambda *_a, **_k: {"status": "pass", "manifest_sha256": "d"})
    monkeypatch.setattr(operator.os, "sched_getaffinity", lambda _: set(range(6, 24)))
    launch.manager_cpus = (6,)
    events = []

    def refuse(*_):
        events.append("source")
        raise ValueError("selected source admission missing")

    monkeypatch.setattr(operator, "source_preparation", refuse)
    monkeypatch.setattr(operator, "operator_module", lambda _: events.append("final readiness"))
    monkeypatch.setattr(operator, "run", lambda *_: events.append("paid run"))
    with pytest.raises(ValueError, match="selected source admission missing"):
        operator._prepare(launch, prep)
    assert events == ["source"]
