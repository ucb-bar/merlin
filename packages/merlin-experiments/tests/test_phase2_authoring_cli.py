"""Installed authoring CLI consumes explicit locations without engine launches."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import authoring_cli as CLI


@pytest.fixture(autouse=True)
def captured_sandbox_inputs(monkeypatch):
    selected = object()
    monkeypatch.setattr(CLI.PC, "select_package_sandbox_inputs", lambda _: selected)
    return selected


def arguments(tmp_path):
    return [
        "--functional-run-id",
        "functional",
        "--functional-submission-sha256",
        "a" * 64,
        "--model",
        "fixture",
        "--wall-budget-seconds",
        "10",
        "--gsim-certificate",
        "certificate.json",
        "--gsim-certificate-sha256",
        "b" * 64,
        "--rtl-facts",
        "facts.json",
        "--telemetry-price-table",
        "prices.yaml",
        "--descriptor",
        "target.yaml",
        "--suite",
        "fixture-study",
        *[
            item
            for name in ("functional-runs-root", "stage-root", "source-root", "contract-root")
            for item in ("--" + name, str(tmp_path / name))
        ],
    ]


@pytest.mark.parametrize(
    "option", ["descriptor", "suite", "functional-runs-root", "stage-root", "source-root", "contract-root"]
)
def test_authoring_requires_explicit_inputs(tmp_path, capsys, option):
    args = arguments(tmp_path)
    index = args.index("--" + option)
    del args[index : index + 2]
    with pytest.raises(SystemExit) as refused:
        CLI.main(args)
    assert refused.value.code == 2
    assert "--" + option in capsys.readouterr().err


@pytest.mark.parametrize("additional", [False, True])
def test_installed_cli_forwards_selected_locations_and_suite(
    tmp_path, monkeypatch, captured_sandbox_inputs, additional
):
    observed = {}
    monkeypatch.setattr(CLI, "load_target_experiment", lambda path, **kw: SimpleNamespace(path=path, **kw))
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"admission": {"consumable": True}}))

    def run_stage(**kwargs):
        observed.update(kwargs)
        return record

    monkeypatch.setattr(CLI.authoring, "run_stage", run_stage)
    args = arguments(tmp_path)
    if additional:
        from test_compiler_libraries import selected

        selection, _ = selected(tmp_path)
        contract = tmp_path / "reviewed.json"
        contract.write_text(json.dumps(selection.contract.record()))
        args += [
            "--compiler-library-root",
            str(selection.root),
            "--compiler-library-contract",
            str(contract),
            "--compiler-library-contract-sha256",
            selection.contract.sha256,
        ]
    assert CLI.main(args) == 0
    for name in ("functional-runs-root", "stage-root", "source-root", "contract-root"):
        assert observed[name.replace("-", "_")] == tmp_path / name
    assert observed["suite"] == "fixture-study"
    assert observed["sandbox_inputs"] is captured_sandbox_inputs
    assert observed["target_experiment"].path == Path("target.yaml")
    assert observed["target_experiment"].source_root == tmp_path / "source-root"
    if additional:
        assert observed["compiler_library_selection"].record() == selection.record()
    else:
        assert "compiler_library_selection" not in observed


def test_installed_authoring_loads_real_descriptor_from_explicit_external_root(tmp_path, monkeypatch):
    from merlin.targetgen import target_experiment as TE

    source = tmp_path / "source-root"
    source.mkdir()
    (source / "target.yaml").write_text(json.dumps({"target": "synthetic", "capsule_corpus": "corpus/public"}))
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"admission": {"consumable": True}}))

    def forbidden(*args, **kwargs):
        pytest.fail("installed descriptor load attempted checkout discovery or process launch")

    def run_stage(**kwargs):
        target = kwargs["target_experiment"]
        assert target.source_root == source
        assert target.path == source / "target.yaml"
        assert target.capsule_corpus == source / "corpus/public"
        return record

    monkeypatch.setattr(TE, "repo_root", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(CLI.authoring, "run_stage", run_stage)
    monkeypatch.chdir(tmp_path)
    assert CLI.main(arguments(tmp_path)) == 0


def test_installed_cli_reopens_explicit_original_pass_owner(tmp_path, monkeypatch):
    from test_compiler_pass_state import original

    inputs, first = original.__wrapped__(tmp_path)
    from merlin_experiments.phase2.compiler_pass_preparation import CompilerPassPreparation

    from merlin.common.digest import sha256_file

    preparation = CompilerPassPreparation(
        "configuration_cleanup",
        baseline_policy={"enabled": False},
        candidate_policy={"enabled": True},
    )
    policies = tmp_path / "original-policies.json"
    policies.write_text(json.dumps(preparation.record()))
    library = inputs["library"]
    contract = tmp_path / "selected-library.json"
    contract.write_text(json.dumps(library.contract.record()))
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"admission": {"consumable": True}}))
    monkeypatch.setattr(CLI, "load_target_experiment", lambda path, **kw: SimpleNamespace(path=path, **kw))
    observed = {}

    def run_stage(**kwargs):
        observed.update(kwargs)
        return record

    monkeypatch.setattr(CLI.authoring, "run_stage", run_stage)
    args = arguments(tmp_path) + [
        "--compiler-library-root",
        str(library.root),
        "--compiler-library-contract",
        str(contract),
        "--compiler-library-contract-sha256",
        library.contract.sha256,
        "--compiler-pass-selection",
        str(inputs["path"]),
        "--compiler-pass-selection-sha256",
        inputs["sha256"],
        "--compiler-pass-admission-root",
        str(inputs["work_root"]),
        "--compiler-pass-preparation",
        str(policies),
        "--compiler-pass-preparation-sha256",
        sha256_file(policies),
    ]
    assert CLI.main(args) == 0
    assert observed["compiler_pass_selection"].record() == first.record()
    assert observed["compiler_pass_preparation"].record() == preparation.record()
    assert (inputs["work_root"] / "original-check-ran").is_file()
    assert not (tmp_path / "stage-root").exists()


def test_resource_admission_precedes_original_pass_factory(tmp_path, monkeypatch):
    from merlin_experiments.phase2 import compiler_pass_state as PASSES
    from merlin_experiments.phase2 import search_configuration as CONFIG
    from merlin_experiments.phase2 import search_resource_launch as RESOURCES

    boundary, calls = object(), []
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"admission": {"consumable": True}}))
    monkeypatch.setattr(CLI, "load_target_experiment", lambda path, **kw: SimpleNamespace(path=path, **kw))
    monkeypatch.setattr(CONFIG, "read_search_configuration", lambda path: object())

    def admit(*args, **kwargs):
        calls.append("resource")
        return boundary

    def restore(args, *, library, stage_root, external_boundary):
        assert external_boundary is boundary and calls == ["resource"]
        calls.append("original_factory")
        return None

    def run_stage(**kwargs):
        assert kwargs["external_resource_boundary"] is boundary
        calls.append("stage")
        return record

    monkeypatch.setattr(RESOURCES, "prepare_launch_boundary", admit)
    monkeypatch.setattr(PASSES, "selection_from_arguments", restore)
    monkeypatch.setattr(CLI.authoring, "run_stage", run_stage)
    assert CLI.main(arguments(tmp_path) + ["--search-configuration", "configuration.json"]) == 0
    assert calls == ["resource", "original_factory", "stage"]


def test_trusted_source_root_default_is_forwarded_without_installed_discovery(tmp_path, monkeypatch):
    observed = {}
    monkeypatch.setattr(CLI, "load_target_experiment", lambda path, **kw: SimpleNamespace(path=path, **kw))
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"admission": {"consumable": True}}))

    def run_stage(**kwargs):
        observed.update(kwargs)
        return record

    monkeypatch.setattr(CLI.authoring, "run_stage", run_stage)
    args = arguments(tmp_path)
    index = args.index("--source-root")
    del args[index : index + 2]
    assert CLI.main(args, source_root=tmp_path) == 0
    assert observed["source_root"] == tmp_path
    assert observed["target_experiment"].source_root == tmp_path


def test_component_cli_accepts_only_explicit_live_owner_handoff(tmp_path, monkeypatch):
    """Argument routing only; these objects cannot issue runtime qualification."""
    from merlin_experiments.phase2 import component_launch as CL
    from merlin_experiments.phase2 import component_launch_inputs as INPUTS

    origin, runtime, inputs = object(), object(), object()
    declaration = tmp_path / "launch.json"
    calls = []

    def load(path, *, phase1_origin, measurement_support):
        assert path == declaration and phase1_origin is origin and measurement_support is runtime
        calls.append("live-inputs")
        return inputs

    def qualify(actual, **kwargs):
        assert actual is inputs and kwargs == {"model": "fixture", "effort": "high"}
        calls.append("qualify")
        return SimpleNamespace(qualification_path=tmp_path / "unissued-diagnostic.json")

    monkeypatch.setattr(INPUTS, "load_component_launch_inputs", load)
    monkeypatch.setattr(CL, "qualify_component_launch", qualify)
    assert (
        CLI.main(
            [
                "--workflow",
                "component-only-v1",
                "--component-launch-inputs",
                str(declaration),
                "--qualification-only",
                "--model",
                "fixture",
            ],
            phase1_origin=origin,
            measurement_support=runtime,
        )
        == 0
    )
    assert calls == ["live-inputs", "qualify"]
