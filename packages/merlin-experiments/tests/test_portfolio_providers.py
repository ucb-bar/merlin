"""Selected backend capabilities assemble inert providers without target-name imports."""

import importlib
import socket
import subprocess
from dataclasses import fields
from types import ModuleType

import pytest
from merlin_experiments.phase2 import portfolio_providers as P


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("provider assembly cannot launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)


def forbidden(*args, **kwargs):
    pytest.fail("assembly executed a runtime capability")


def backend(tmp_path, *, movement=True, source_pair=True):
    selected = ModuleType("synthetic_selected_backend")
    if source_pair:
        for name in ("short_program_environment", "prepare_short_program_build", "prepare_short_program_execution"):
            setattr(selected, name, forbidden)
    abi = ModuleType("selected_witness_capability")
    abi.derive_native_witness_abi = lambda *, target: {
        "native_layout": forbidden,
        "expected_symbol": "selected_symbol",
        "abi_provenance": {"target": target},
    }
    selected.host_witness_abi = abi
    primitive = ModuleType("selected_primitive_capability")
    source = tmp_path / "original_primitive_owner.py"
    source.write_text("# selected primitive capability provenance\n")
    primitive.__file__ = str(source)

    def prepare(*, include_operand_movement=False):
        forbidden()

    primitive.prepare_primitive_probe = prepare if movement else forbidden
    for name in ("execute_prepared_primitive", "isolated_primitive_signature", "runtime_elf_digest"):
        setattr(primitive, name, forbidden)
    selected.primitive_probe = primitive
    return selected


def assemble(tmp_path, selected, **options):
    return P.assemble(
        target="different_from_backend_name",
        backend=selected,
        output=tmp_path / "providers",
        **{
            "semantic_only": False,
            "probe_interface": None,
            "probe_runtime_receipt": None,
            "profile_counters": False,
            **options,
        },
    )


def test_absent_optional_capabilities_are_unavailable(tmp_path):
    result = assemble(tmp_path, ModuleType("empty_backend"))
    assert all(getattr(result, field.name) is None for field in fields(result))


@pytest.mark.parametrize("capability", ["host_witness_abi", "primitive_probe"])
@pytest.mark.parametrize("value", [None, {}, "target_spelled_module"])
def test_explicit_malformed_capability_is_not_treated_as_absent(tmp_path, capability, value):
    selected = ModuleType("bad_backend")
    setattr(selected, capability, value)
    with pytest.raises(ValueError):
        assemble(tmp_path, selected)


@pytest.mark.parametrize("capability", ["host_witness_abi", "primitive_probe"])
def test_declared_lazy_import_failure_propagates(tmp_path, capability):
    selected = ModuleType("broken_backend")

    def lookup(name):
        if name == capability:
            raise ModuleNotFoundError("declared dependency unavailable", name="missing_dependency")
        raise AttributeError(name)

    selected.__getattr__ = lookup
    with pytest.raises(ModuleNotFoundError, match="declared dependency unavailable"):
        assemble(tmp_path, selected)


@pytest.mark.parametrize("semantic_only", [False, True])
def test_real_inert_constructors_preserve_selected_adapter_and_semantics(tmp_path, monkeypatch, semantic_only):
    selected = backend(tmp_path)
    original_source = selected.primitive_probe.__file__
    interface, receipt = tmp_path / "interface.mlir", tmp_path / "runtime.json"
    interface.write_text("// synthetic interface\n")
    receipt.write_text("{}")
    monkeypatch.setattr(importlib, "import_module", lambda *a, **k: pytest.fail("target-spelled discovery"))
    result = assemble(
        tmp_path,
        selected,
        semantic_only=semantic_only,
        probe_interface=interface,
        probe_runtime_receipt=receipt,
        profile_counters=True,
    )
    assert result.provider.adapter is selected.primitive_probe
    assert result.provider.adapter.__file__ == original_source
    assert result.provider.profile_counters is True
    assert result.context_provider.adapter is selected.primitive_probe
    assert result.paired_context_provider.adapter is selected.primitive_probe
    assert result.semantic_provider is not None
    assert result.semantic_provider.physical.abi_provenance == {"target": "different_from_backend_name"}
    assert result.semantic_provider.legacy is not None
    assert (result.source_pair_provider is None) is semantic_only
    assert (result.semantic_provider.lane_migration is None) is semantic_only
    if not semantic_only:
        assert result.source_pair_provider.adapter is selected
        assert result.semantic_provider.lane_migration.runtime_provider is result.source_pair_provider
    assert not (tmp_path / "providers").exists()


def test_operand_movement_parameter_controls_context_availability(tmp_path):
    selected = backend(tmp_path, movement=False)
    result = assemble(tmp_path, selected)
    assert result.context_provider is result.paired_context_provider is None
    assert result.semantic_provider is not None


def test_requested_probe_cannot_silently_disappear(tmp_path):
    with pytest.raises(ValueError):
        assemble(
            tmp_path,
            ModuleType("empty_backend"),
            probe_interface=tmp_path / "interface",
            probe_runtime_receipt=tmp_path / "receipt",
        )


@pytest.mark.parametrize(
    "missing",
    [
        "prepare_primitive_probe",
        "execute_prepared_primitive",
        "isolated_primitive_signature",
        "runtime_elf_digest",
        "__file__",
    ],
)
def test_incomplete_declared_primitive_capability_refuses(tmp_path, missing):
    selected = backend(tmp_path)
    delattr(selected.primitive_probe, missing)
    with pytest.raises(ValueError):
        assemble(tmp_path, selected)


def test_missing_declared_witness_deriver_refuses(tmp_path):
    selected = backend(tmp_path)
    selected.host_witness_abi.derive_native_witness_abi = None
    with pytest.raises(ValueError):
        assemble(tmp_path, selected)
