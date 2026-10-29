"""Public source diagnostics do not enter the ordinary numerical grading route."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import public_scalability as suite
from merlin_experiments.phase1.feedback import public_scalability_inspection as inspection
from test_public_scalability import staged


@pytest.mark.parametrize("mode", ["compile_only", "candidate_diagnostics"])
def test_selected_public_inspection_reuses_source_owner_without_grading(tmp_path, monkeypatch, mode):
    from merlin_experiments.phase1.feedback import candidate_observations, compiler_profile_runtime

    root, _, te, context = staged(tmp_path)
    monkeypatch.setenv(suite.ENV, "uninstalled")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    suite.install(te, root)
    args = SimpleNamespace(sim="spike", tiers="L0,L1,L2", compile_only=mode == "compile_only",
        candidate_diagnostics=mode == "candidate_diagnostics", capsules="case0",
        compiler_bundle_manifest=tmp_path / "original-bundle.yaml", contract=tmp_path / "contract.yaml",
        capsules_root=tmp_path / "foreign-root")
    seen = []

    def observe(a, **kwargs):
        assert a is not args and a.capsules_root == root / "cases" and a.capsules == "case0"
        assert kwargs["context"] is context
        seen.append(mode)
        return 19

    monkeypatch.setattr(compiler_profile_runtime, "selfcheck_compile_only", observe)
    monkeypatch.setattr(compiler_profile_runtime, "selfcheck_diagnostics", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(candidate_observations, "diagnostics_cli", observe)
    monkeypatch.setattr(suite, "grade", lambda *a, **kw: pytest.fail("inspection entered numerical grade"))
    assert suite.cli(args, context=context) == 19
    assert seen == [mode] and args.capsules_root == tmp_path / "foreign-root"
    args.capsules = "not-selected"
    with pytest.raises(ValueError, match="outside the selected"):
        suite.cli(args, context=context)
    assert seen == [mode]


@pytest.mark.parametrize("change", [
    dict(candidate_diagnostics=True), dict(shape_coverage=True), dict(offload_census=True),
    dict(model_layers=True), dict(dynamic_census=True), dict(scale_probe=True), dict(caller_layout=True),
    dict(tuning_overrides={"x": 1}), dict(reference_request="trial"), dict(sim="gsim"), dict(tiers="L3"),
    dict(reference_experiment_request_id="trial"), dict(tuning_experiment={"id": "trial"}),
    dict(candidate_diagnostics="true"),
])
def test_source_inspection_cannot_widen_to_other_probes_variants_or_engines(change):
    request = dict(compile_only=True, sim="spike", tiers="L0,L1,L2")
    request.update(change)
    with pytest.raises(ValueError):
        inspection.requested(request)
    with pytest.raises(ValueError):
        suite.request_policy("spike", "L0,L1,L2", inspection=True)


def test_diagnostics_recheck_original_selected_suite_before_publication(tmp_path, monkeypatch, capsys):
    from merlin_experiments.phase1.feedback import compiler_profile_runtime

    root, _, te, context = staged(tmp_path)
    monkeypatch.setenv(suite.ENV, "uninstalled")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    suite.install(te, root)
    args = SimpleNamespace(sim="spike", tiers="", compile_only=True, candidate_diagnostics=False,
                           capsules="case0", capsules_root=None, out=str(tmp_path / "author-result.json"))

    def changed(a, **kw):
        Path(a.out).write_text('{"status":"observed"}')
        print('{"status":"observed"}')
        source = root / "public_sources/case0.mlir"
        source.chmod(0o644)
        source.write_text("changed")
        return 0

    monkeypatch.setattr(compiler_profile_runtime, "selfcheck_compile_only", changed)
    with pytest.raises(ValueError, match="changed"):
        suite.cli(args, context=context)
    assert not Path(args.out).exists() and not capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--compile-only", "--candidate-diagnostics"])
def test_standalone_author_client_transports_public_source_inspection(tmp_path, flag):
    import json
    import subprocess
    import sys

    from merlin_experiments.phase1.tools import selfcheck

    client = tmp_path / "agent_selfcheck.py"
    client.write_bytes(Path(selfcheck.__file__).read_bytes())
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission/compiler.py").write_text("# synthetic client transport only\n")
    result = subprocess.run([sys.executable, str(client), "--suite", "scalability", "--capsules", "case0",
        "--sim", "spike", "--tiers", "L0,L1,L2", flag, "--wait-budget", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert result.returncode == 3, result.stdout + result.stderr
    request_path, = (tmp_path / ".qa_channel").glob("req_*.json")
    request = json.loads(request_path.read_bytes())
    assert request["suite"] == "scalability" and request["capsules"] == "case0"
    assert inspection.requested(request) == flag.removeprefix("--").replace("-", "_")


@pytest.mark.parametrize("flag", ["--compile-only", "--candidate-diagnostics"])
def test_standalone_source_inspection_refuses_sweep_without_creating_grading_requests(tmp_path, flag):
    import subprocess
    import sys

    from merlin_experiments.phase1.tools import selfcheck

    client = tmp_path / "agent_selfcheck.py"
    client.write_bytes(Path(selfcheck.__file__).read_bytes())
    result = subprocess.run([sys.executable, str(client), "--suite", "scalability", "--capsules", "case0",
        "--sim", "spike", flag, "--sweep", "variants/a", "--wait-budget", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2 and "refuse inspection" in result.stderr
    assert not list(tmp_path.glob(".qa_channel/req_*.json"))
