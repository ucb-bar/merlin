"""Explicit source observations cannot replace semantic or runtime authority."""

from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.targetgen.contract.build_service import file_digest
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.source_observation import ExplicitSourceObservation

_SOURCE = """module { func.func @main(%a: tensor<1x1xi8>) -> tensor<1x1xi8> {
%d = tensor.empty() : tensor<1x1xi8>
%copy = linalg.copy ins(%a : tensor<1x1xi8>) outs(%d : tensor<1x1xi8>) -> tensor<1x1xi8>
func.return %copy : tensor<1x1xi8> } }"""
_LLVM = """module { llvm.func @entry(%a: !llvm.ptr, %b: !llvm.ptr) {
%input = llvm.ptrtoint %a : !llvm.ptr to i64
llvm.inline_asm has_side_effects "opaque $0", "r,~{memory}" %input : (i64) -> ()
llvm.return } }"""
_ABI = CompileOnlySourceAbi((CompileOnlyTensor("a", (1, 1), "i8"),), (CompileOnlyTensor("result", (1, 1), "i8"),))
_CB = {
    "abi_version": "0.1",
    "operand_naming": "positional",
    "tensors": {
        "arg0": {"shape": [1, 1], "dtype": "i8", "role": "input"},
        "out0": {"shape": [1, 1], "dtype": "i8", "role": "output"},
    },
    "kernel_abi": {
        "kind": "whole_program",
        "args": [{"tensor": "arg0", "access": "read"}, {"tensor": "out0", "access": "write"}],
        "outputs": ["out0"],
    },
}


class Reader:
    """Synthetic reader for attribution tests only, not semantics or a role."""

    def __init__(self):
        self.selection = "fixed diagnostic source reader"
        self.calls = 0
        self.mutation = None

    def verify(self):
        return None

    def record(self):
        return {"selection": self.selection, "scope": "synthetic observation only"}

    def observe(self, **kwargs):
        self.calls += 1
        if self.mutation == "callback":
            self.observe = self.substituted
        if self.mutation == "selection":
            self.selection = "substituted configuration"
        if self.mutation == "product":
            kwargs["lowered_mlir"].write_text("module {}")
        return {"observed_pointer_origin": kwargs["dataflow"].argument_origin(2), "claimed_role": "PASS"}

    def substituted(self, **kwargs):
        return {"substituted": True}


def prepare(tmp_path, *, original_abi=_ABI):
    import inspect
    import json

    source = tmp_path / "original.mlir"
    lowered = tmp_path / "emitted.mlir"
    command = tmp_path / "command_buffer.json"
    source.write_text(_SOURCE)
    lowered.write_text(_LLVM)
    command.write_text(json.dumps(_CB))
    owner = Path(inspect.getsourcefile(Reader))
    reader = Reader()
    service = ExplicitSourceObservation(
        "diagnostic-target",
        source,
        original_abi,
        64,
        100,
        reader,
        tuple((str(path), file_digest(path)) for path in (source, owner)),
    )
    arguments = {
        "source": source,
        "lowered_mlir": lowered,
        "command_buffer": _CB,
        "command_buffer_path": command,
        "entry_symbol": "entry",
        "evidence_root": tmp_path,
    }
    return service, reader, arguments


def test_actual_reader_call_consumes_original_emitted_bytes_and_cannot_mint_labels(tmp_path):
    service, reader, args = prepare(tmp_path)
    before = service.sha256
    result = service.observe(**args)
    assert result["observations"]["observed_pointer_origin"] == 0
    assert result["observations"]["claimed_role"] == "PASS"
    assert "source_equivalence" in result["unknown"] and "runtime" in result["unknown"]
    assert reader.calls == 1 and service.sha256 == before
    records = [I.verify(path) for path in tmp_path.rglob("invocation.json")]
    assert len(records) == 1 and records[0]["stage"] == "explicit_original_source_dataflow_observation"
    assert {row["path"] for row in records[0]["inputs"]} == {
        str(args["source"]),
        str(args["lowered_mlir"]),
        str(args["command_buffer_path"]),
    }
    assert any(row["path"] == str(tmp_path / "source_observation.json") for row in records[0]["outputs"])


@pytest.mark.parametrize("mutation", ["callback", "selection"])
def test_same_source_callback_or_configuration_substitution_refuses(tmp_path, mutation):
    service, reader, args = prepare(tmp_path)
    reader.mutation = mutation
    with pytest.raises(ValueError, match="changed"):
        service.observe(**args)
    assert reader.calls == 1


def test_source_byte_identity_and_complete_abi_are_not_reader_labels(tmp_path):
    service, reader, args = prepare(tmp_path)
    changed = tmp_path / "changed.mlir"
    changed.write_text(_SOURCE.replace("func.return %copy", "func.return %a"))
    with pytest.raises(ValueError, match="original program bytes"):
        service.observe(**{**args, "source": changed})
    assert reader.calls == 0


def test_changed_command_buffer_and_pointer_roster_refuse_before_reader(tmp_path):
    service, reader, args = prepare(tmp_path)
    with pytest.raises(ValueError, match="actual emitted command buffer"):
        service.observe(**{**args, "command_buffer": {}})
    args["lowered_mlir"].write_text(_LLVM.replace(", %b: !llvm.ptr", ""))
    with pytest.raises(ValueError, match="pointer slots"):
        service.observe(**args)
    assert reader.calls == 0


def test_forged_original_tensor_type_refuses_without_calling_selected_reader(tmp_path):
    bad = CompileOnlySourceAbi((CompileOnlyTensor("a", (1, 2), "i8"),), _ABI.outputs)
    with pytest.raises(ValueError, match="complete ordered tensor types"):
        prepare(tmp_path, original_abi=bad)


def test_unknown_emitted_dispatch_is_not_silently_observed(tmp_path):
    service, reader, args = prepare(tmp_path)
    args["lowered_mlir"].write_text(_LLVM.replace("module {", "module { llvm.func @opaque(!llvm.ptr) "))
    with pytest.raises(ValueError, match="without external dispatch"):
        service.observe(**args)
    assert reader.calls == 0


def test_source_drift_and_missing_method_membership_refuse(tmp_path):
    service, reader, args = prepare(tmp_path)
    args["source"].write_text(_SOURCE + "\n")
    with pytest.raises(ValueError, match="omits|changed"):
        service.verify()


def test_reader_cannot_overwrite_actual_lowered_input_during_observation(tmp_path):
    service, reader, args = prepare(tmp_path)
    reader.mutation = "product"
    with pytest.raises(ValueError, match="inputs changed"):
        service.observe(**args)
    assert reader.calls == 1


def test_selected_method_code_replacement_refuses_before_invocation(tmp_path):
    service, reader, args = prepare(tmp_path)
    original = Reader.observe.__code__
    try:
        Reader.observe.__code__ = Reader.substituted.__code__
        with pytest.raises(ValueError, match="callback identity"):
            service.observe(**args)
        assert reader.calls == 0
    finally:
        Reader.observe.__code__ = original


def test_repeated_publication_preserves_original_observation(tmp_path):
    service, reader, args = prepare(tmp_path)
    service.observe(**args)
    before = (tmp_path / "source_observation.json").read_bytes()
    with pytest.raises(ValueError, match="new private product"):
        service.observe(**args)
    assert (tmp_path / "source_observation.json").read_bytes() == before and reader.calls == 1
