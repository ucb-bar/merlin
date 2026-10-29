"""Actual package emitter processes; no native compilation or target execution.

Only invocation/file custody is checked. Every pass-family declaration remains
unknown; this test grants no effects or supported optimization qualification.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import test_shared_pass_consumption as probe
import test_target_configuration_native as native
from test_selected_schedule_contracts import inputs

from merlin.common.digest import sha256_file
from merlin.targetgen import artifact_scale
from merlin.targetgen import capsule_common as CC
from merlin.targetgen import package_runtime as P
from merlin.targetgen.shared_pass_contract import shared_pass_requirement


def check_actual_effects(package, generated, artifact):
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin
    from xdsl.parser import Parser

    from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
    from merlin.targetgen.isa_dialect_gen import build_dialect
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    paths = json.loads((package.directory / "selection.json").read_text())
    selected = read_scheduling_contracts(*paths, target="neutral")
    context = Context()
    for dialect in (builtin.Builtin, arith.Arith, build_dialect(selected.isa)):
        context.load_dialect(dialect)
    target = Parser(context, (generated / "selected.target.mlir").read_text()).parse_module()
    llvm = Parser(make_llvm_context(), artifact).parse_module()
    return probe.check_emitted_calls(target, llvm)


@pytest.fixture
def common():
    return CC


def fixture(tmp_path, *, version=2):
    facts = tmp_path / "facts"
    facts.mkdir()
    selected, *_ = inputs(facts, width=4, endian="<")
    package = native.package(tmp_path, selected, 32)
    requirement = shared_pass_requirement(version=version)
    contract = probe.declared_contract(requirement)
    contract["schema"] = f"merlin.shared_pass_contract.v{version}"
    if version == 2:
        for boundary in contract["boundaries"].values():
            boundary["contracts"].update(resources="neutral-fixture-bounds-v1", completion="synchronous-C-v1")
    package.manifest["shared_pass_contract"] = contract
    (package.directory / "manifest.yaml").write_text(json.dumps(package.manifest))
    interface = tmp_path / "source.mlir"
    interface.write_text("module {}\n")
    return package, requirement, interface


def test_actual_ordinary_v2_consumer_gets_lowered_target_not_interface(tmp_path, common, monkeypatch):
    package, requirement, interface = fixture(tmp_path)
    events = []

    class Owner:
        def process_lower_interface_v1(self, *args, **kwargs):
            raise AssertionError("v2 must not delegate through legacy process protocol")

    monkeypatch.setattr(P, "active_package_executor", lambda: Owner())

    original_runner = P.run_entrypoint

    def invoke(pkg, command, source, *args, **kwargs):
        events.append((command, Path(source).name))
        return original_runner(pkg, command, source, *args, **kwargs)

    monkeypatch.setattr(P, "run_entrypoint", invoke)
    generated = tmp_path / "generated"
    memo = {("legacy", "unused"): ("forged", {}, "forged")}
    _, artifact = common.lower_interface(
        package,
        interface,
        generated,
        contract=None,
        timeout=30,
        overlap=True,
        memo=memo,
        shared_pass_requirement=requirement,
    )
    assert events == [
        ("parse", "input.interface.mlir"),
        ("lower_interface_to_target", "input.interface.mlir"),
        ("emit_target_bundle", "selected.target.mlir"),
    ]
    assert len(memo) == 1
    original_target = generated / "lowered.target.mlir"
    selected_target = generated / "selected.target.mlir"
    assert original_target.read_bytes() == selected_target.read_bytes() != interface.read_bytes()
    consumption = json.loads((generated / "consumption.json").read_text())
    assert consumption["input_path"] == str(selected_target)
    assert consumption["input_sha256"] == sha256_file(selected_target)
    assert consumption["actual_commands"] == [list(command) for command in native.COMMANDS]
    receipt = json.loads((generated / "shared_pass_target.json").read_text())
    assert receipt["requirement"] == requirement
    assert len(receipt["files"]) == 6
    assert receipt["files"]["target_artifact"]["sha256"] == sha256_file(generated / "lowered.llvm.mlir")
    assert artifact == (generated / "lowered.llvm.mlir").read_text()
    # Independently reconstruct the actual emitted operation/constant DAG.
    assert check_actual_effects(package, generated, artifact) == 10


@pytest.mark.parametrize("role", ["interface_source", "original_target", "selected_target"])
def test_emitter_mutating_any_frozen_input_refuses(tmp_path, common, role):
    package, requirement, interface = fixture(tmp_path)
    generated = tmp_path / "generated"

    def invoke(pkg, command, source, *args, **kwargs):
        result = P.run_entrypoint(pkg, command, source, *args, **kwargs)
        if command == "emit_target_bundle":
            path = {
                "interface_source": interface,
                "original_target": generated / "lowered.target.mlir",
                "selected_target": generated / "selected.target.mlir",
            }[role]
            path.write_bytes(path.read_bytes() + b"\n")
        return result

    with pytest.raises(ValueError, match="input or consumed output changed"):
        common.lower_interface(
            package, interface, generated, contract=None, timeout=30, invoke=invoke, shared_pass_requirement=requirement
        )


@pytest.mark.parametrize("role", ["command_buffer.json", "lowered.llvm.mlir"])
def test_last_output_observer_cannot_clobber_consumed_products(tmp_path, common, monkeypatch, role):
    package, requirement, interface = fixture(tmp_path)
    generated = tmp_path / "generated"
    original_refusal = artifact_scale.refusal

    def changed(artifact):
        result = original_refusal(artifact)
        path = generated / role
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(artifact_scale, "refusal", changed)
    with pytest.raises(ValueError, match="input or consumed output changed"):
        common.lower_interface(
            package, interface, generated, contract=None, timeout=30, shared_pass_requirement=requirement
        )


def test_emitter_replacing_selected_input_with_source_is_refused(tmp_path, common):
    package, requirement, interface = fixture(tmp_path)
    driver = package.directory / "driver.py"
    driver.write_text(
        driver.read_text().replace(
            "module = Parser(context, source.read_text()).parse_module()",
            "module = Parser(context, source.with_name('input.interface.mlir').read_text()).parse_module()",
        )
    )
    # Exit0 and structurally valid output are insufficient: the independent
    # emitted-effects check detects dropping the actual lowered-target work.
    _, artifact = common.lower_interface(
        package, interface, tmp_path / "generated", contract=None, timeout=30, shared_pass_requirement=requirement
    )
    assert artifact.count("llvm.call @neutral_set") != 10
    with pytest.raises(AssertionError, match="emitted effects differ"):
        check_actual_effects(package, tmp_path / "generated", artifact)


def test_v1_ordinary_legacy_command_route_is_unchanged(tmp_path, common):
    package, requirement, interface = fixture(tmp_path, version=1)
    events = []

    def invoke(pkg, command, source, *args, **kwargs):
        events.append((command, Path(source).name))
        if command == "emit_command_buffer":
            raise RuntimeError("legacy sentinel")
        return P.run_entrypoint(pkg, command, source, *args, **kwargs)

    with pytest.raises(RuntimeError, match="legacy sentinel"):
        common.lower_interface(
            package,
            interface,
            tmp_path / "generated",
            contract=None,
            timeout=30,
            invoke=invoke,
            shared_pass_requirement=requirement,
        )
    assert events[-1] == ("emit_command_buffer", "input.interface.mlir")


@pytest.mark.parametrize("indirect", [False, True])
def test_v2_fresh_direct_products_required_before_callback(tmp_path, common, indirect):
    package, requirement, interface = fixture(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "selected.target.mlir").write_text("stale")
    if indirect:
        alias = tmp_path / "alias"
        alias.symlink_to(generated, target_is_directory=True)
        generated = alias
    with pytest.raises(ValueError, match="fresh retained|direct private"):
        common.lower_interface(
            package,
            interface,
            generated,
            contract=None,
            timeout=30,
            invoke=lambda *a, **k: pytest.fail("must refuse before callback"),
            shared_pass_requirement=requirement,
        )


@pytest.mark.parametrize("name", ["input.interface.mlir", "selected.target.mlir"])
def test_source_aliasing_reserved_inputs_refuses_before_any_write(tmp_path, common, name):
    package, requirement, interface = fixture(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir()
    reserved = generated / name
    reserved.hardlink_to(interface)
    original = interface.read_bytes()
    with pytest.raises(ValueError, match="fresh retained"):
        common.lower_interface(
            package,
            reserved,
            generated,
            contract=None,
            timeout=30,
            invoke=lambda *a, **k: pytest.fail("no callback before refusal"),
            shared_pass_requirement=requirement,
        )
    assert interface.read_bytes() == reserved.read_bytes() == original


@pytest.mark.parametrize("name", ["lowered.target.mlir", "selected.target.mlir", "lowered.llvm.mlir"])
def test_callback_cannot_plant_alias_for_later_host_write(tmp_path, common, name):
    package, requirement, interface = fixture(tmp_path)
    generated = tmp_path / "generated"
    original = interface.read_bytes()

    def invoke(pkg, command, source, *args, **kwargs):
        result = P.run_entrypoint(pkg, command, source, *args, **kwargs)
        stage = "emit_target_bundle" if name == "lowered.llvm.mlir" else "lower_interface_to_target"
        if command == stage:
            (generated / name).hardlink_to(interface)
        return result

    with pytest.raises(ValueError, match="fresh host-owned"):
        common.lower_interface(
            package, interface, generated, contract=None, timeout=30, invoke=invoke, shared_pass_requirement=requirement
        )
    assert interface.read_bytes() == original


def test_emitter_cannot_replace_identity_copy_with_same_byte_alias(tmp_path, common):
    package, requirement, interface = fixture(tmp_path)
    generated = tmp_path / "generated"

    def invoke(pkg, command, source, *args, **kwargs):
        result = P.run_entrypoint(pkg, command, source, *args, **kwargs)
        if command == "emit_target_bundle":
            Path(source).unlink()
            Path(source).hardlink_to(generated / "lowered.target.mlir")
        return result

    with pytest.raises(ValueError, match="must not alias"):
        common.lower_interface(
            package, interface, generated, contract=None, timeout=30, invoke=invoke, shared_pass_requirement=requirement
        )


@pytest.mark.parametrize("alias,stage", [("symlink", "emitter"), ("hardlink", "last_observer")])
def test_final_receipt_alias_refuses_without_clobber(tmp_path, common, monkeypatch, alias, stage):
    package, requirement, interface = fixture(tmp_path)
    generated = tmp_path / "generated"
    original = interface.read_bytes()
    original_refusal = artifact_scale.refusal
    expected_cb = []

    def plant():
        destination = generated / "shared_pass_target.json"
        target = interface if alias == "symlink" else generated / "command_buffer.json"
        if alias == "symlink":
            destination.symlink_to(target)
        else:
            expected_cb.append(target.read_bytes())
            destination.hardlink_to(target)

    def invoke(pkg, command, source, *args, **kwargs):
        result = P.run_entrypoint(pkg, command, source, *args, **kwargs)
        if command == "emit_target_bundle" and stage == "emitter":
            plant()
        return result

    def last_observer(artifact):
        result = original_refusal(artifact)
        if stage == "last_observer":
            plant()
        return result

    monkeypatch.setattr(artifact_scale, "refusal", last_observer)
    with pytest.raises(ValueError, match="fresh host-owned receipt"):
        common.lower_interface(
            package,
            interface,
            generated,
            contract=None,
            timeout=30,
            invoke=invoke,
            shared_pass_requirement=requirement,
        )
    assert interface.read_bytes() == original
    if expected_cb:
        assert (generated / "command_buffer.json").read_bytes() == expected_cb[0]


def test_imported_owner_compatibility_symlink_does_not_constrain_artifacts(tmp_path, common, monkeypatch):
    alias = tmp_path / "imported_common.py"
    alias.symlink_to(Path(common.__file__).resolve())
    spec = importlib.util.spec_from_file_location("merlin.targetgen._linked_v2_common", alias)
    linked = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(linked)
    assert Path(linked.__file__).is_symlink()
    test_actual_ordinary_v2_consumer_gets_lowered_target_not_interface(tmp_path, linked, monkeypatch)
