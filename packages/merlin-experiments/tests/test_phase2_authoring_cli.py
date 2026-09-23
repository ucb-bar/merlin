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


def test_installed_cli_forwards_selected_locations_and_suite(tmp_path, monkeypatch, captured_sandbox_inputs):
    observed = {}
    monkeypatch.setattr(CLI, "load_target_experiment", lambda path, **kw: SimpleNamespace(path=path, **kw))
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"admission": {"consumable": True}}))

    def run_stage(**kwargs):
        observed.update(kwargs)
        return record

    monkeypatch.setattr(CLI.authoring, "run_stage", run_stage)
    assert CLI.main(arguments(tmp_path)) == 0
    for name in ("functional-runs-root", "stage-root", "source-root", "contract-root"):
        assert observed[name.replace("-", "_")] == tmp_path / name
    assert observed["suite"] == "fixture-study"
    assert observed["sandbox_inputs"] is captured_sandbox_inputs
    assert observed["target_experiment"].path == Path("target.yaml")
    assert observed["target_experiment"].source_root == tmp_path / "source-root"


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
