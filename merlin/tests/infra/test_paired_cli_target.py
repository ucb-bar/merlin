"""Paired launch uses explicit target identity before creating measurement output."""

import importlib
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2 import paired_cli
from merlin_experiments.phase2 import paired_inputs as PI

from merlin.common.paths import merlin_dir


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(merlin_dir() / "experiments/gemmini_perf_bench/scripts"))
    return importlib.import_module("run_paired_perf_bench")


def arguments(tmp_path):
    return [
        "--functional-run-id",
        "functional",
        "--functional-submission-sha256",
        "a" * 64,
        "--candidate-record",
        str(tmp_path / "candidate.json"),
        "--corpus-root",
        str(tmp_path / "corpus"),
        "--corpus-manifest",
        str(tmp_path / "manifest.json"),
        "--corpus-manifest-sha256",
        "b" * 64,
        "--corpus-capsules-sha256",
        "c" * 64,
        "--phase",
        "tuning",
        "--gsim-certificate",
        str(tmp_path / "certificate.json"),
        "--gsim-certificate-sha256",
        "d" * 64,
        "--rtl-facts",
        str(tmp_path / "facts.json"),
        "--run-id",
        "measurement",
    ]


def test_descriptor_is_required_before_admission(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(PI, "load_paired_inputs", lambda *a, **k: pytest.fail("unexpected admission"))
    with pytest.raises(SystemExit) as error:
        runner.main(arguments(tmp_path))
    assert error.value.code == 2


@pytest.mark.parametrize("target", ["vector-fixture", "tensor-fixture"])
def test_output_collision_checked_under_selected_target(runner, tmp_path, monkeypatch, target):
    descriptor = tmp_path / "descriptor.yaml"
    observed = []

    def load(path, *, source_root):
        assert path == descriptor
        assert source_root == runner.repo_root()
        return SimpleNamespace(target=target)

    def runs(selected, suite):
        observed.append((selected, suite))
        return tmp_path / selected / suite

    (tmp_path / target / "perf-bench/measurement").mkdir(parents=True)
    monkeypatch.setattr(paired_cli, "load_target_experiment", load)
    monkeypatch.setattr(runner, "runs_root", runs)
    monkeypatch.setattr(PI, "load_paired_inputs", lambda *a, **k: pytest.fail("unexpected admission"))
    with pytest.raises(PC.CampaignGateError, match="must be fresh"):
        runner.main([*arguments(tmp_path), "--descriptor", str(descriptor)])
    assert observed == [(target, "capsule-bench"), (target, "perf-bench")]


@pytest.mark.parametrize("missing", ["measurement-root", "functional-runs-root", "contract-root", "source-root"])
def test_installed_measurement_requires_explicit_roots(tmp_path, monkeypatch, missing):
    monkeypatch.setattr(paired_cli, "load_target_experiment", lambda *a: pytest.fail("missing input reached admission"))
    argv = [*arguments(tmp_path), "--descriptor", str(tmp_path / "descriptor.yaml")]
    for name in ("measurement-root", "functional-runs-root", "contract-root", "source-root"):
        if name != missing:
            argv += [f"--{name}", str(tmp_path / name)]
    with pytest.raises(SystemExit) as error:
        paired_cli.main(argv)
    assert error.value.code == 2


def test_installed_measurement_forwards_roots_without_checkout_discovery(tmp_path, monkeypatch):
    from merlin.common import paths
    from merlin.targetgen import target_experiment

    roots = {
        name: tmp_path / name for name in ("measurement-root", "functional-runs-root", "contract-root", "source-root")
    }
    roots["source-root"].mkdir()
    descriptor = roots["source-root"] / "descriptor.yaml"
    descriptor.write_text("target: synthetic\ncapsule_corpus: public\n")
    monkeypatch.setattr(target_experiment, "repo_root", lambda: pytest.fail("descriptor discovered a checkout"))
    monkeypatch.setattr(paths, "repo_root", lambda: pytest.fail("discovered a checkout"))
    monkeypatch.setattr(paths, "merlin_dir", lambda: pytest.fail("discovered native resources"))

    class AdmittedInputs(Exception):
        pass

    def admit(*args, **kwargs):
        assert args[3].source_root == roots["source-root"]
        assert args[3].capsule_corpus == roots["source-root"] / "public"
        assert kwargs["functional_runs_root"] == roots["functional-runs-root"]
        assert not roots["measurement-root"].exists()
        raise AdmittedInputs

    monkeypatch.setattr(PI, "load_paired_inputs", admit)
    argv = [*arguments(tmp_path), "--descriptor", "descriptor.yaml"]
    for name, value in roots.items():
        argv += [f"--{name}", str(value)]
    with pytest.raises(AdmittedInputs):
        paired_cli.main(argv)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_installed_measurement_passes_contract_and_records_execution_refusal(tmp_path, monkeypatch, error):
    """Exercise the real CLI writer with synthetic admitted inputs and a refusing executor."""
    target = SimpleNamespace(target="synthetic")
    root, contract = tmp_path / "measurements", tmp_path / "contract"
    inputs = SimpleNamespace(
        functional=SimpleNamespace(run_id="functional"),
        baseline=tmp_path / "baseline",
        baseline_sha256="a" * 64,
        candidate_sha256="c" * 64,
        handoff=SimpleNamespace(record_sha256="b" * 64),
        gsim_certificate=SimpleNamespace(to_dict=lambda: {"synthetic": True}),
        corpus=SimpleNamespace(root=tmp_path / "corpus", manifest_sha256="d" * 64, capsules_sha256="e" * 64),
    )
    identity = paired_cli.ME.ResultIdentity("tuning", "baseline", "PK", "case", "gsim", "r000")
    plan = SimpleNamespace(expected=(identity,), declaration={"fixture": True}, declaration_sha256="f" * 64)
    monkeypatch.setattr(paired_cli, "load_target_experiment", lambda path, **kwargs: target)
    monkeypatch.setattr(PI, "load_paired_inputs", lambda *a, **kw: inputs)
    monkeypatch.setattr(PI, "identity_guard", lambda inputs: {"fixture": "unchanged"})
    monkeypatch.setattr(PC, "functional_fork", lambda functional: object())
    monkeypatch.setattr(PC, "check_fork", lambda *a: SimpleNamespace(to_dict=lambda: {"ok": True}))
    monkeypatch.setattr(paired_cli.PM, "build_measurement_plan", lambda inputs: plan)
    monkeypatch.setattr(paired_cli.PM, "schedule_fanout", lambda *a, **kw: {"workers": 1})
    monkeypatch.setattr(paired_cli.MS, "load_rtl_identity", lambda *a: {"fixture": True})
    calls = []

    def refuse(selected_plan, destination, **kwargs):
        assert selected_plan is plan
        assert destination == root / "measurement"
        assert kwargs["contract_root"] == contract
        assert kwargs["target_experiment"] is target
        calls.append(destination)
        raise error("synthetic unavailable engine")

    monkeypatch.setattr(paired_cli.PM, "execute_schedule", refuse)
    argv = [
        *arguments(tmp_path),
        "--source-root",
        str(tmp_path),
        "--descriptor",
        str(tmp_path / "descriptor.yaml"),
        "--measurement-root",
        str(root),
        "--contract-root",
        str(contract),
        "--functional-runs-root",
        str(tmp_path / "functional"),
    ]
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            paired_cli.main(argv)
    else:
        assert paired_cli.main(argv) == 2
    document = json.loads((root / "measurement/campaign_manifest.json").read_bytes())
    assert calls == [root / "measurement"]
    assert document["status"] == "NO_GO"
    assert document["refusal"] == f"{error.__name__}: synthetic unavailable engine"
    assert document["completion"]["complete"] is False
    assert document["raw_results"]["n_cells"] == 0
