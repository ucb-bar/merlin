"""Source-plan transport and LLVM metadata regressions; no execution workload."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from merlin.targetgen.contract.source_ownership import SourceOwnershipError, source_owned_mutables
from merlin.targetgen.oot_starterkit.plan import source_operation_inventory, validate_mixed_program_plan
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context


INTERFACE = '''module attributes {merlin_iface.version = "0.1", merlin_iface.target = "example", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<3x4xi8>
  %A = merlin_iface.tensor {name = "A", role = "input"} : tensor<2x3xi8>
  %packed = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<3x4xi8>) -> !merlin_iface.resident
  %acc = merlin_iface.matmul %A, %packed : (tensor<2x3xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y = merlin_iface.commit %acc {name = "Y", epilogue = ["relu"], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<2x4xi32>
  merlin_iface.evict %packed : (!merlin_iface.resident) -> ()
}
'''.encode()


def _neutral():
    from merlin_experiments.phase1.neutral_program_inputs import encode_source
    return encode_source(INTERFACE)


def _buffer(raw):
    return {
        "tensors": {"W": {"shape": [3, 4], "dtype": "i8", "role": "weight"},
                    "A": {"shape": [2, 3], "dtype": "i8", "role": "input"},
                    "tmp": {"shape": [2, 4], "dtype": "i32", "role": "intermediate"},
                    "Y": {"shape": [2, 4], "dtype": "i32", "role": "output"}},
        "kernel_abi": {"kind": "whole_program", "args": [
            {"tensor": "W", "access": "read"}, {"tensor": "A", "access": "read"},
            {"tensor": "tmp", "access": "write"}, {"tensor": "Y", "access": "write"}], "outputs": ["Y"]},
        "params": {"global_program_plan": {
            "schema": "mixed_program_plan_v1", "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_op_count": 2, "entry_bindings": ["W", "A"], "output_bindings": ["Y"],
            "source_values": [{"op_index": 0, "result_index": 0, "tensor": "tmp"},
                              {"op_index": 1, "result_index": 0, "tensor": "Y"}],
            "compiler_temporaries": [], "prologue_instruction_range": [0, 1],
            "epilogue_instruction_range": [3, 4], "schedule_instruction_count": 4,
            "tasks": [
                {"task_index": 0, "kind": "host", "host_reason": "Mathematical producer.",
                 "source_op_indices": [0], "instruction_start": 1, "instruction_end": 2,
                 "reads": ["W", "A"], "writes": ["tmp"]},
                {"task_index": 1, "kind": "host", "host_reason": "Mathematical consumer.",
                 "source_op_indices": [1], "instruction_start": 2, "instruction_end": 3,
                 "reads": ["tmp"], "writes": ["Y"]}]}}}


LOWERED = '''builtin.module {
  llvm.module_flags [#llvm.mlir.module_flag<error, "target-abi", "lp64d">, #llvm.mlir.module_flag<append_unique, "example-features", ["extension-a", "extension-b"]>]
  llvm.func @kernel(%w: !llvm.ptr, %a: !llvm.ptr, %tmp: !llvm.ptr, %y: !llvm.ptr) {
    %i = llvm.mlir.constant(0 : i64) : i64
    %p = llvm.getelementptr %tmp[%i] {merlin.global_task = 0 : i64, merlin.source_op_index = 0 : i64} : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %q = llvm.getelementptr %y[%i] {merlin.global_task = 1 : i64, merlin.source_op_index = 1 : i64} : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.return
  }
}'''


def test_neutral_transport_typed_inventory_and_complete_scratch_plan():
    raw = _neutral()
    inventory = source_operation_inventory(raw)
    assert inventory["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert [op["operation"] for op in inventory["operations"]] == ["numerical_program.matmul", "numerical_program.relu"]
    assert inventory["operations"][0]["operands"][0]["source"] == {"arg_index": 1}
    assert inventory["operations"][1]["operands"][0]["source"] == {"op_index": 0, "result_index": 0}
    assert inventory["returns"][0]["source"] == {"op_index": 1, "result_index": 0}
    cb = _buffer(raw)
    assert validate_mixed_program_plan(raw, cb, LOWERED)["ok"]
    assert source_owned_mutables(raw, cb, lowered_mlir=LOWERED) == ("tmp",)


@pytest.mark.parametrize("mutation", ["hash", "owner", "shape", "early-read", "output", "entry"])
def test_neutral_transport_does_not_relax_source_ownership(mutation):
    raw, cb = _neutral(), _buffer(_neutral())
    plan = cb["params"]["global_program_plan"]
    if mutation == "hash": plan["source_sha256"] = "0" * 64
    elif mutation == "owner": plan["source_values"][0]["op_index"] = 1
    elif mutation == "shape": cb["tensors"]["tmp"]["shape"] = [1, 8]
    elif mutation == "early-read": plan["tasks"][0]["reads"].append("tmp")
    elif mutation == "output": plan["output_bindings"] = ["tmp"]
    elif mutation == "entry": plan["entry_bindings"].reverse()
    with pytest.raises(SourceOwnershipError):
        source_owned_mutables(raw, cb, lowered_mlir=LOWERED)


@pytest.mark.parametrize("bad_source", [INTERFACE, b"not a source program", b'{"schema":"numerical_program.v1"}'])
def test_missing_plan_is_diagnosed_before_source_parser_or_private_source_echo(bad_source):
    checked = validate_mixed_program_plan(bad_source, {"params": {}})
    assert checked == {"ok": False, "findings": ["params.global_program_plan is required"]}


def test_interface_uses_owning_grammar_and_keeps_lifetime_source_indices():
    inventory = source_operation_inventory(INTERFACE)
    assert inventory["source_sha256"] == hashlib.sha256(INTERFACE).hexdigest()
    assert inventory["arguments"] == [{"shape": [3, 4], "dtype": "i8"}, {"shape": [2, 3], "dtype": "i8"}]
    assert [op["operation"] for op in inventory["operations"]] == [
        "merlin_iface.resident_pack", "merlin_iface.matmul", "merlin_iface.commit", "merlin_iface.evict"]
    assert inventory["returns"] == [{"shape": [2, 4], "dtype": "i32", "source": {"op_index": 2, "result_index": 0}}]
    with pytest.raises(ValueError):
        source_operation_inventory(INTERFACE.replace(b"merlin_iface.evict", b"merlin_iface.unknown"))
    with pytest.raises(ValueError):
        source_operation_inventory(INTERFACE.replace(b"  merlin_iface.evict", b"  arith.constant 1 : i32\n  merlin_iface.evict"))


def test_interface_complete_plan_retains_typed_scratch_ownership():
    cb = _buffer(INTERFACE)
    plan = cb["params"]["global_program_plan"]
    plan.update(source_op_count=4, schedule_instruction_count=3, epilogue_instruction_range=[2, 3],
                source_values=[{"op_index": 2, "result_index": 0, "tensor": "Y"}],
                compiler_temporaries=[{"tensor": "tmp", "source_op_index": 2, "source_result_index": 0,
                                       "purpose": "Source-owned output staging."}],
                tasks=[{"task_index": 0, "kind": "host", "host_reason": "Whole mathematical program.",
                        "source_op_indices": [0, 1, 2, 3], "instruction_start": 1, "instruction_end": 2,
                        "reads": ["W", "A"], "writes": ["tmp", "Y"]}])
    assert source_owned_mutables(INTERFACE, cb) == ("tmp",)
    # A residency handle's opaque type cannot authorize a tensor allocation.
    plan["compiler_temporaries"][0]["source_op_index"] = 0
    with pytest.raises(SourceOwnershipError, match="shape/dtype"):
        source_owned_mutables(INTERFACE, cb)


@pytest.mark.parametrize("change", ["undefined", "duplicate-id", "output-type", "opaque-field", "duplicate-key"])
def test_numerical_inventory_refuses_ambiguous_or_untyped_provenance(change):
    doc = json.loads(_neutral())
    if change == "undefined": doc["expressions"][0]["operands"][0] = "future"
    elif change == "duplicate-id": doc["expressions"][0]["id"] = doc["inputs"][0]["id"]
    elif change == "output-type": doc["outputs"][0]["shape"] = [8]
    elif change == "opaque-field": doc["ownership"] = {"trust_me": True}
    raw = json.dumps(doc).encode()
    if change == "duplicate-key": raw = raw.replace(b'"schema":', b'"schema":"numerical_program.v1","schema":', 1)
    with pytest.raises(ValueError):
        source_operation_inventory(raw)


def test_llvm23_module_flags_roundtrip_preserves_metadata_and_task_checks():
    from xdsl.parser import Parser
    module = Parser(make_llvm_context(), LOWERED).parse_module()
    module.verify()
    emitted = str(module)
    assert '"target-abi", "lp64d"' in emitted
    assert '"example-features", ["extension-a", "extension-b"]' in emitted
    assert str(Parser(make_llvm_context(), emitted).parse_module()) == emitted
    cb = _buffer(_neutral())
    assert validate_mixed_program_plan(_neutral(), cb, emitted)["ok"]
    wrong_owner = emitted.replace("merlin.source_op_index = 1", "merlin.source_op_index = 0")
    assert not validate_mixed_program_plan(_neutral(), cb, wrong_owner)["ok"]
    wrong_arity = emitted.replace(", %y: !llvm.ptr", "")
    assert not validate_mixed_program_plan(_neutral(), cb, wrong_arity)["ok"]


def test_llvm_module_flags_accepts_no_executable_or_unrelated_metadata():
    from xdsl.parser import Parser
    from xdsl.utils.exceptions import VerifyException
    with pytest.raises(VerifyException, match="module_flag metadata"):
        Parser(make_llvm_context(), 'builtin.module { llvm.module_flags [#other.flag<1>] }').parse_module().verify()
    with pytest.raises(VerifyException, match="module-level metadata"):
        Parser(make_llvm_context(), '''builtin.module { llvm.func @bad() {
          llvm.module_flags [#llvm.mlir.module_flag<error, "key", "value">]
          llvm.return
        } }''').parse_module().verify()


@pytest.mark.parametrize("transport", ["neutral_math_v1", "shared_interface_v1", None])
def test_private_oracle_uses_original_selected_source_and_frozen_transport(tmp_path, monkeypatch, transport):
    from contextlib import contextmanager
    from merlin.targetgen import capsule_lane_evidence as L
    from merlin.targetgen.contract import source_ownership as O
    from merlin_experiments.phase1.feedback import snapshot_execution as S

    path = tmp_path / "program.mlir"
    path.write_bytes(INTERFACE)
    cb = _buffer(_neutral())
    # The owner is trusted host selection. A submitted transport declaration
    # below must have no influence on which source identity is validated.
    owner = None if transport is None else SimpleNamespace(owner=SimpleNamespace(bundle={
        "author_profile": {"disclosure": {"input_format": transport}}}))
    monkeypatch.setattr(S, "active_snapshot", lambda: owner)
    captured = []

    @contextmanager
    def scope(source, command_buffer, **kwargs):
        captured.append(source)
        yield ()

    monkeypatch.setattr(O, "source_mutable_scope", scope)
    cb["params"]["input_format"] = "candidate_fake_transport"
    with L.oracle_source_scope({"interface_mlir": path.name, "__dir__": str(tmp_path)}, cb, LOWERED):
        pass
    assert captured == [_neutral() if transport == "neutral_math_v1" else INTERFACE]


def test_private_neutral_oracle_scope_validates_real_plan_and_rejects_original_hash(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_lane_evidence as L
    from merlin.targetgen.contract.source_ownership import active_source_owned_mutables
    from merlin_experiments.phase1.feedback import snapshot_execution as S

    path = tmp_path / "program.mlir"
    path.write_bytes(INTERFACE)
    monkeypatch.setattr(S, "active_snapshot", lambda: SimpleNamespace(owner=SimpleNamespace(bundle={
        "author_profile": {"disclosure": {"input_format": "neutral_math_v1"}}})))
    capsule = {"interface_mlir": path.name, "__dir__": str(tmp_path)}
    cb = _buffer(_neutral())
    assert active_source_owned_mutables(cb) == ()
    with L.oracle_source_scope(capsule, cb, LOWERED):
        assert active_source_owned_mutables(cb) == ("tmp",)
    assert active_source_owned_mutables(cb) == ()
    cb["params"]["global_program_plan"]["source_sha256"] = hashlib.sha256(INTERFACE).hexdigest()
    with pytest.raises(SourceOwnershipError, match="exact source bytes"):
        with L.oracle_source_scope(capsule, cb, LOWERED):
            pytest.fail("the private original hash must not substitute for delivered neutral bytes")
