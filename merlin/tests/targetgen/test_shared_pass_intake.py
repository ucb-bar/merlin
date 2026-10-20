"""Ordinary package intake honors an original opt-in without granting semantics."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_shared_pass_contract import manifest

from merlin.targetgen import capsule_common as C
from merlin.targetgen import package_runtime as P
from merlin.targetgen.oot_starterkit.passes import inspect_manifest, main
from merlin.targetgen.shared_pass_context import (
    active_shared_pass_requirement,
    scoped_shared_pass_requirement,
    selected_shared_pass_requirement,
)
from merlin.targetgen.shared_pass_contract import shared_pass_requirement


def package(tmp_path, *, declared=True):
    value = manifest()
    value.update(
        artifact_type="mlir_oot_target_backend",
        language="python",
        authoring={"mode": "hand_curated"},
        integrity_exempt=False,
        entrypoints={"tool": "driver.py"},
    )
    for name in (
        "parse",
        "lower_interface_to_target",
        "emit_command_buffer",
        "emit_target_artifact",
        "lower_target_to_llvm",
    ):
        value["commands"][name] = {"argv": ["{tool}", name, "{input_mlir}", "{output_json}"]}
    if not declared:
        value.pop("shared_pass_contract")
    path = tmp_path / "package"
    path.mkdir()
    (path / "driver.py").write_text("# inert intake fixture; never run\n")
    (path / "manifest.yaml").write_text(yaml.safe_dump(value))
    return path


def test_normal_loader_keeps_legacy_and_enforces_explicit_requirement(tmp_path):
    path = package(tmp_path, declared=False)
    assert P.load_package(path).target == "neutral"
    with pytest.raises(P.CertFailure, match="shared_pass_contract"):
        P.load_package(path, shared_pass_requirement=shared_pass_requirement())


def test_normal_loader_accepts_declarations_without_qualifying_them(tmp_path):
    path = package(tmp_path)
    result = P.load_package(path, shared_pass_requirement=shared_pass_requirement())
    assert result.target == "neutral"
    assert not hasattr(result, "shared_pass_qualification")


def test_original_scope_applies_to_default_loader_and_cannot_be_weakened(tmp_path):
    path = package(tmp_path, declared=False)
    requirement = shared_pass_requirement()
    assert active_shared_pass_requirement() is None
    with scoped_shared_pass_requirement(requirement):
        detached = active_shared_pass_requirement()
        detached["families"].clear()
        assert active_shared_pass_requirement() == requirement
        with pytest.raises(P.CertFailure, match="shared_pass_contract"):
            P.load_package(path)
        with scoped_shared_pass_requirement(requirement):
            assert selected_shared_pass_requirement() == requirement
        for weaker in (None, shared_pass_requirement(required_supported=["producer_packing"])):
            with pytest.raises(ValueError, match="original"):
                with scoped_shared_pass_requirement(weaker):
                    pytest.fail("scope replacement entered")
    assert active_shared_pass_requirement() is None
    assert P.load_package(path).target == "neutral"


def test_scope_restores_after_exception_and_rejects_malformed_policy():
    with pytest.raises(RuntimeError):
        with scoped_shared_pass_requirement(shared_pass_requirement()):
            raise RuntimeError("failed original compile")
    assert active_shared_pass_requirement() is None
    malformed = shared_pass_requirement()
    malformed["limits"]["max_repairs"] = True
    with pytest.raises(ValueError):
        selected_shared_pass_requirement(malformed)


def test_original_scoped_requirement_reaches_ordinary_lowering_before_any_command(tmp_path):
    path = package(tmp_path, declared=False)
    pkg = P.load_package(path)
    interface = tmp_path / "input.mlir"
    interface.write_text("module {}")
    calls = []
    with scoped_shared_pass_requirement(shared_pass_requirement()):
        with pytest.raises(ValueError, match="shared_pass_contract"):
            C.lower_interface(pkg, interface, tmp_path / "work", contract=None, timeout=3, invoke=calls.append)
    assert calls == []
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("change", ["contract", "command", "parse_command"])
def test_command_callback_cannot_change_selected_pass_interfaces(tmp_path, change):
    pkg = P.load_package(package(tmp_path))
    interface = tmp_path / "input.mlir"
    interface.write_text("module {}")
    calls = []

    def invoke(_pkg, command, *_args, **_kwargs):
        calls.append(command)
        if change == "contract":
            pkg.manifest["shared_pass_contract"]["families"][0]["reason"] = "late different declaration"
        elif change == "command":
            pkg.manifest["commands"]["emit_target_bundle"]["argv"].append("changed")
        else:
            pkg.manifest["commands"]["parse"]["argv"].append("changed")
        return SimpleNamespace(returncode=0, stdout="module {}", stderr="")

    with pytest.raises(ValueError, match="changed"):
        C.lower_interface(
            pkg,
            interface,
            tmp_path / "work",
            contract=None,
            timeout=3,
            invoke=invoke,
            shared_pass_requirement=shared_pass_requirement(),
        )
    assert calls == ["parse"]


def test_inspection_reports_exact_inputs_and_never_invokes_package(tmp_path, monkeypatch):
    path = package(tmp_path)
    monkeypatch.setattr(P, "run_entrypoint", lambda *a, **k: pytest.fail("inspection ran compiler"))
    report = inspect_manifest(path / "manifest.yaml")
    assert report["status"] == "declared_interfaces"
    assert report["changed_ir_consumption"] == "NOT_RUN"
    assert report["numerical_agreement"] == "NOT_RUN"
    assert report["performance"] == "NOT_MEASURED"
    assert Path(report["inputs"]["manifest"]["path"]) == path / "manifest.yaml"
    assert len(report["inputs"]["manifest"]["sha256"]) == 64


def test_cli_requirement_then_inspect_is_structural_only_and_preserves_inputs(tmp_path, capsys):
    path = package(tmp_path)
    before = (path / "manifest.yaml").read_bytes()
    requirement_path, report_path = tmp_path / "requirement.json", tmp_path / "inspection.json"
    assert main(["requirement", "--output", str(requirement_path)]) == 0
    assert (
        main(
            [
                "inspect",
                "--manifest",
                str(path / "manifest.yaml"),
                "--requirement",
                str(requirement_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    assert json.loads(report_path.read_text())["changed_ir_consumption"] == "NOT_RUN"
    assert (path / "manifest.yaml").read_bytes() == before
    with pytest.raises(FileExistsError):
        main(["requirement", "--output", str(requirement_path)])
    assert capsys.readouterr().out == ""


def test_cli_missing_or_weakened_contract_refuses_without_success_report(tmp_path, capsys):
    path = package(tmp_path, declared=False)
    with pytest.raises(SystemExit) as refused:
        main(["inspect", "--manifest", str(path / "manifest.yaml")])
    assert refused.value.code == 2
    assert "shared_pass_contract" in capsys.readouterr().err
    original = shared_pass_requirement()
    weakened = copy.deepcopy(original)
    weakened["families"].pop("producer_packing")
    with pytest.raises(ValueError):
        selected_shared_pass_requirement(weakened)
