"""Original public-only emission diagnostics, without simulator or admission."""

import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import candidate_observations as O


@pytest.mark.parametrize("change_source", [False, True])
def test_selected_public_roster_original_emit_and_no_answer_projection(tmp_path, monkeypatch, change_source):
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen import package_runtime as PR

    compiler = tmp_path / "compiler"
    compiler.mkdir()
    (compiler / "manifest.yaml").write_text("commands: {}\nbuild_options: {packing: true}\n")
    capsules = [{"name": name, "__dir__": str(tmp_path / name), "interface_mlir": "public.mlir", "label": "public",
                 "dtype": "i8", "expected": "PRIVATE_GOLDEN_SENTINEL"} for name in ("one", "two")]
    calls = []
    monkeypatch.setattr(O.CR, "discover_capsules", lambda root, **kwargs: capsules)
    monkeypatch.setattr(PR, "load_package", lambda path, **kw: SimpleNamespace(directory=path))
    monkeypatch.setattr(PR, "integrity_scan", lambda pkg: calls.append("integrity"))
    monkeypatch.setattr(PR, "build_package", lambda pkg, **kw: calls.append("original_build"))
    monkeypatch.setattr(CC, "make_run_paths", lambda root, name, **kw: SimpleNamespace(generated=root / name))
    def emit(package, submission, capsule, paths, **kwargs):
        calls.append(capsule["name"])
        paths.generated.mkdir(parents=True, exist_ok=True)
        paths.generated.joinpath("input.interface.mlir").write_text("module {}")
        paths.generated.joinpath("lowered.llvm.mlir").write_text("module { llvm.func @kernel() { llvm.return } }")
        paths.generated.joinpath("command_buffer.json").write_text(json.dumps({"commands": [], "tensors": {}}))
        if change_source:
            compiler.joinpath("compiler.py").write_text("changed during emission")
        return package, {}, ""
    monkeypatch.setattr(CC, "run_entrypoints", emit)
    report = O.emit_public_diagnostics(compiler, target="selected", capsules_root=tmp_path,
                                       contract=tmp_path / "contract", capsules="two", timeout=10)
    assert calls == ["integrity", "original_build", "two"]
    assert report["member_count"] == 1
    assert report["source_stability"] == ("UNKNOWN" if change_source else "unchanged")
    if change_source:
        assert report["members"][0]["status"] == "UNKNOWN_SOURCE_CHANGED"
        assert report["members"][0]["diagnostics"]["source_emission_join"]["status"] == "UNKNOWN"
    assert report["numerical_status"] == report["selected_rtl_status"] == "NOT_RUN"
    assert "PRIVATE_GOLDEN_SENTINEL" not in json.dumps(report)
    assert str(tmp_path) not in json.dumps(report)
    with pytest.raises(ValueError, match="unknown public"):
        O.emit_public_diagnostics(compiler, target="selected", capsules_root=tmp_path,
                                  contract=tmp_path, capsules="private", timeout=10)


def test_ambiguous_artifact_collection_does_not_choose_a_member(tmp_path):
    for name, body in (("one", "module {}"), ("two", "different source")):
        directory = tmp_path / name
        directory.mkdir()
        directory.joinpath("input.interface.mlir").write_text(body)
    report = O.observe_generated_artifacts(tmp_path)
    assert "source" in report["product_collection"]["unavailable_roles"]
    assert report["source"]["status"] == "UNKNOWN"


def test_public_per_output_numeric_policy_is_projected_without_answers(tmp_path):
    policy = {"schema": "merlin.numeric_policy.per_output.v1", "compare": "per_output", "outputs": {
        "integer": {"compare": "exact_int", "dtype": "i32"},
        "floating": {"compare": "tolerance_float", "dtype": "f32", "atol": 0.01, "rtol": 0.02}}}
    report = O.observe_generated_artifacts(tmp_path, numeric_context={"numeric_policy": policy,
                                          "expected": "PROTECTED_ANSWER"})
    assert report["numerical_context"]["selection"]["numeric_policy"] == policy
    assert report["numerical_context"]["numerical_verdict"] == "UNKNOWN"
    assert "PROTECTED_ANSWER" not in json.dumps(report)
    policy["outputs"]["integer"]["expected"] = "PROTECTED_ANSWER"
    report = O.observe_generated_artifacts(tmp_path, numeric_context={"numeric_policy": policy})
    assert report["numerical_context"]["selection"]["numeric_policy"] == {"status": "UNKNOWN"}


def test_standalone_client_and_broker_forward_the_new_closed_flag():
    root = Path(O.__file__).parents[1]
    client = ast.parse((root / "tools/selfcheck.py").read_text())
    imports = [node.module or "" for node in ast.walk(client) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith("merlin") for name in imports)
    from merlin_experiments.phase1.brokers.selfcheck import FORWARDED_PROBES
    assert FORWARDED_PROBES["candidate_diagnostics"] == "--candidate-diagnostics"


def test_observed_options_reopen_actual_completed_products_and_drift(tmp_path):
    from merlin.common import invocation_record as I

    product = tmp_path / "kernel.o"
    product.write_bytes(b"observed object")
    tool = tmp_path / "tool"
    tool.write_bytes(b"original synthetic tool identity")
    with I.observe(tmp_path, stage="object", argv=[str(tool), "-O2"], outputs=(product,), env={}) as receipt:
        receipt.complete(subprocess.CompletedProcess([], 0, b"", b""))
    report = O.observe_generated_artifacts(tmp_path)
    assert report["observed_codegen_options"]["status"] == "observed"
    assert report["observed_codegen_options"]["records"][0]["observed_flags"] == ["-O2"]
    product.write_bytes(b"changed object")
    assert O.observe_generated_artifacts(tmp_path)["observed_codegen_options"]["status"] == "UNKNOWN"


def test_original_selfcheck_compile_flag_avoids_default_engine_and_baseline(tmp_path, monkeypatch, capsys):
    from merlin_experiments.phase1.feedback import selfcheck as S

    monkeypatch.setattr(S, "resolve_context", lambda *args: SimpleNamespace(target="selected", repo=tmp_path))
    monkeypatch.setattr(S, "_default_sim", lambda *args: pytest.fail("compile diagnostics selected a simulator"))
    calls = []
    def observe(submission, **kwargs):
        calls.append((submission, kwargs))
        return {"schema": "merlin.public_candidate_diagnostics.v1", "numerical_status": "NOT_RUN"}
    monkeypatch.setattr(O, "emit_public_diagnostics", observe)
    result = S.main(["--candidate-diagnostics", "--submission", str(tmp_path / "unadmitted"),
                     "--capsules-root", str(tmp_path / "public"), "--capsules", "one", "--timeout", "10"])
    assert result == 0
    assert calls[0][1]["capsules"] == "one"
    assert json.loads(capsys.readouterr().out)["numerical_status"] == "NOT_RUN"
