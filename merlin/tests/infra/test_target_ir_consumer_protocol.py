"""Actual selected target-file transport, without assuming consumer semantics."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_selected_schedule_contracts import inputs

from merlin.common.digest import sha256_file
from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
from merlin.targetgen import capsule_common as CC
from merlin.targetgen.contract import schemas
from merlin.targetgen.package_runtime import CertFailure

CONSUMER = dict(
    schema="merlin.target_ir_consumer.v1",
    command="emit_target_bundle",
    input="selected_target_mlir",
    output="command_buffer_and_target_artifact",
)
TARGET = """module {
func.func @kernel() {
%slot = arith.constant 0 : i8
%value = arith.constant 3 : i32
%other = arith.constant 0 : i32
"neutral_isa.set"(%slot, %value, %other) : (i8, i32, i32) -> ()
"neutral_isa.set"(%slot, %value, %other) : (i8, i32, i32) -> ()
func.return
}
}
"""


def package(tmp_path):
    return SimpleNamespace(
        directory=tmp_path,
        tool=tmp_path / "compiler.py",
        manifest=dict(
            artifact_type="mlir_oot_target_backend",
            target="neutral",
            language="python",
            authoring=dict(mode="agent_generated_from_rtl_facts"),
            integrity_exempt=False,
            entrypoints=dict(tool="compiler.py"),
            target_ir_consumer=copy.deepcopy(CONSUMER),
            commands={
                name: dict(argv=["{tool}", "{input_mlir}", "{output_json}"])
                for name in (
                    "parse",
                    "lower_interface_to_target",
                    "emit_command_buffer",
                    "lower_target_to_llvm",
                    "emit_target_bundle",
                )
            },
        ),
    )


def selection(tmp_path, enabled=True):
    facts = tmp_path / "facts"
    facts.mkdir()
    selected, *_ = inputs(facts)
    return ConfigurationIRSelection(selected, "independent-neutral-persistent-state-v1", enabled=enabled)


def command_buffer():
    return dict(
        abi_version="0.1",
        target="neutral",
        tensors=dict(A=dict(shape=[1, 4], dtype="i8", role="input"), Y=dict(shape=[1, 4], dtype="i8", role="output")),
        commands=[dict(opcode="MOVEMENT", operands=dict(src="A", dst="Y"))],
    )


def run(tmp_path, selected, *, mutation=None, pkg=None, **kwargs):
    pkg = pkg or package(tmp_path)
    source = tmp_path / "interface.mlir"
    source.write_text("module {}\n")
    generated = tmp_path / "generated"
    calls = []

    def invoke(_pkg, name, inp, output=None, *, timeout):
        calls.append((name, inp))
        if mutation is not None:
            mutation(name, inp, pkg, selected)
        if name == "parse":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if name == "lower_interface_to_target":
            return SimpleNamespace(returncode=0, stdout=TARGET, stderr="")
        assert name == "emit_target_bundle" and inp == generated / "selected.target.mlir"
        module = selected.parse(inp.read_text())
        count = sum(op.name == "neutral_isa.set" for op in module.walk())
        output.write_text(json.dumps(command_buffer()))
        return SimpleNamespace(returncode=0, stdout=f"; actual consumer received {count} setup writes\n", stderr="")

    answer = CC.lower_interface(
        pkg, source, generated, contract=None, timeout=5, invoke=invoke, target_configuration=selected, **kwargs
    )
    return pkg, generated, calls, answer


def test_optional_consumer_uses_existing_manifest_schema(tmp_path):
    schemas.validate_manifest(package(tmp_path).manifest)


def test_package_target_cannot_borrow_a_different_selected_isa(tmp_path):
    selected, pkg = selection(tmp_path, enabled=False), package(tmp_path)
    pkg.manifest["target"] = "other"
    with pytest.raises(ValueError, match="facts and package target disagree"):
        run(tmp_path, selected, pkg=pkg)
    assert not (tmp_path / "generated").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_exact_selected_target_reaches_bundle_and_retains_original(tmp_path, enabled):
    selected = selection(tmp_path, enabled)
    memo = {"unrelated": "original"}
    pkg, generated, calls, (cb, artifact) = run(tmp_path, selected, overlap=True, memo=memo)
    assert [name for name, _ in calls] == ["parse", "lower_interface_to_target", "emit_target_bundle"]
    assert memo == {"unrelated": "original"}
    assert (generated / "lowered.target.mlir").read_text() == TARGET
    assert artifact == f"; actual consumer received {1 if enabled else 2} setup writes\n"
    assert cb == command_buffer()
    receipt = generated / "target_configuration.json"
    report = CC.revalidate_target_configuration(pkg, generated, selected, expected_receipt_sha256=sha256_file(receipt))
    assert report["files"]["selected_target"]["path"] == str((generated / "selected.target.mlir").resolve())
    assert report["files"]["target_artifact"]["sha256"] == sha256_file(generated / "lowered.llvm.mlir")
    assert report["configuration"]["verification"]["removed_operations"] == int(enabled)


@pytest.mark.parametrize("fault", ["absent", "version", "missing_input", "missing_output", "unknown", "abi"])
def test_selected_route_requires_declared_consumer_before_any_invocation(tmp_path, fault):
    selected, pkg = selection(tmp_path, enabled=False), package(tmp_path)
    if fault == "absent":
        pkg.manifest.pop("target_ir_consumer")
    elif fault == "version":
        pkg.manifest["target_ir_consumer"]["schema"] = "merlin.target_ir_consumer.v99"
    elif fault in ("missing_input", "missing_output"):
        absent = "{input_mlir}" if fault == "missing_input" else "{output_json}"
        pkg.manifest["commands"]["emit_target_bundle"]["argv"].remove(absent)
    elif fault == "unknown":
        pkg.manifest["target_ir_consumer"]["unreviewed"] = True
    else:
        pkg.manifest["abi_version"] = "0.2"
    with pytest.raises(CertFailure, match="explicit.*merlin.target_ir_consumer"):
        run(tmp_path, selected, pkg=pkg)
    assert not (tmp_path / "generated").exists()


@pytest.mark.parametrize("fault", ["selection", "consumer", "input", "target", "selected"])
def test_emitter_cannot_change_original_selection_or_consumed_files(tmp_path, fault):
    def mutation(name, inp, pkg, selected):
        if fault == "selection" and name == "parse":
            object.__setattr__(selected, "enabled", False)
        elif fault == "consumer" and name == "parse":
            pkg.manifest["commands"]["emit_target_bundle"]["argv"].append("--changed")
        elif fault == "input" and name == "parse":
            inp.write_text("module { }\n")
        elif fault == "target" and name == "emit_target_bundle":
            (inp.parent / "lowered.target.mlir").write_text("module {}\n")
        elif fault == "selected" and name == "emit_target_bundle":
            inp.write_text("module {}\n")

    with pytest.raises(ValueError, match="changed|reselect|selection"):
        run(tmp_path, selection(tmp_path), mutation=mutation)
    assert not (tmp_path / "generated/target_configuration.json").exists()


@pytest.mark.parametrize("role", ["receipt", "selected_target", "target_artifact", "command_buffer"])
def test_retained_receipt_digest_and_all_file_pins_are_checked(tmp_path, role):
    selected = selection(tmp_path)
    pkg, generated, _, _ = run(tmp_path, selected)
    receipt = generated / "target_configuration.json"
    expected = sha256_file(receipt)
    report = json.loads(receipt.read_text())
    path = receipt if role == "receipt" else Path(report["files"][role]["path"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        CC.revalidate_target_configuration(pkg, generated, selected, expected_receipt_sha256=expected)


@pytest.mark.parametrize("name", ["selected.target.mlir", "../escape.mlir", "command_buffer.json"])
def test_selected_artifact_cannot_overwrite_original_inputs_or_escape(tmp_path, name):
    with pytest.raises(ValueError, match="overlaps"):
        run(tmp_path, selection(tmp_path), artifact_name=name)
