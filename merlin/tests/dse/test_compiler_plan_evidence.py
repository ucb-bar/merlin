"""Structural compiler evidence cannot certify arithmetic or timing."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan
from merlin.perf.model_placement import prepare_captured_source


SOURCE = '''builtin.module {
  func.func @forward() -> tensor<2xi32> {
    %0 = arith.constant dense<[3, 7]> : tensor<2xi32>
    func.return %0 : tensor<2xi32>
  }
}'''
LOWERED = '''builtin.module {
  llvm.func @kernel(%0: !llvm.ptr) {
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.getelementptr %0[%1] {merlin.global_task = 0 : i64} : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.return
  }
}'''


def _buffer():
    return {
        "tensors": {"result": {"shape": [2], "dtype": "i32", "role": "output"}},
        "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "result"}],
                       "outputs": ["result"]},
        "params": {"global_program_plan": {
            "schema": "mixed_program_plan_v1",
            "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
            "source_op_count": 1,
            "tasks": [{"task_index": 0, "kind": "host", "source_op_indices": [0],
                       "instruction_start": 1, "instruction_end": 2,
                       "reads": [], "writes": ["result"]}],
            "schedule_instruction_count": 3,
            "prologue_instruction_range": [0, 1], "epilogue_instruction_range": [2, 3],
            "source_values": [{"op_index": 0, "result_index": 0, "tensor": "result"}],
            "entry_bindings": [], "output_bindings": ["result"],
        }},
    }


def _verify(buffer=None, lowered=LOWERED):
    return verify_compiler_global_plan(source_text=SOURCE, lowered_text=lowered,
                                       command_buffer=_buffer() if buffer is None else buffer,
                                       candidate_sha256="a" * 64)


def test_structural_coverage_is_explicitly_not_semantic_equivalence():
    # This mock has task ownership but does not implement the source constant. Therefore the
    # structural receipt MUST NOT license a correct-program or model-speedup assertion.
    result = _verify()
    assert result["status"] == "verified"
    assert result["numeric_equivalence"] == "requires independent mechanism witnesses"
    assert result["timing"] == "UNKNOWN"
    assert result["shared_immutable_constants"] == 1
    assert result["host_activity"]["artifact_sha256"] == hashlib.sha256(LOWERED.encode()).hexdigest()
    assert result["host_activity"]["top_allocations_by_static_payload"] == []
    assert result["host_activity"]["top_buffers_by_scalar_memory_payload"] == []
    assert result["host_activity"]["task_activity_coverage"] == "complete"
    assert "0" in {row["task"] for row in result["host_activity"]["tasks"]}
    # Legacy plans remain structurally valid, but representation-dependent transforms must
    # fail closed until the compiler emits an exact physical storage contract.
    metadata = result["source_plan_metadata"]
    assert metadata["status"] == "partial"
    assert metadata["physical_storage"]["status"] == "partial"
    assert metadata["physical_storage"]["exact_physical_representations"] == 0
    assert metadata["physical_storage"]["unknown"][0]["reason"] == (
        "no verified physical storage encoding")


@pytest.mark.parametrize("mutation", [
    lambda b: b["params"]["global_program_plan"].update(source_op_count=2),
    lambda b: b["params"]["global_program_plan"].update(source_sha256="b" * 64),
    lambda b: b["params"]["global_program_plan"]["tasks"][0].update(source_op_indices=[0, 0]),
    lambda b: b["params"]["global_program_plan"]["tasks"][0].update(instruction_start=0),
    lambda b: b["params"]["global_program_plan"].update(source_values=[None]),
    lambda b: b["params"]["global_program_plan"].update(entry_bindings=["result"]),
    lambda b: b["params"]["global_program_plan"]["tasks"][0].update(reads=[[]]),
    lambda b: b["params"]["global_program_plan"]["tasks"][0].update(kind="contraction"),
    lambda b: b["kernel_abi"].update(args=[]),
    lambda b: b["kernel_abi"].update(outputs=[]),
    lambda b: b["tensors"]["result"].update(shape=[4]),
])
def test_invalid_or_incomplete_plan_is_refused(mutation):
    buffer = deepcopy(_buffer())
    mutation(buffer)
    assert _verify(buffer)["status"] == "refused"


def test_unowned_nonconstant_cannot_hide_in_shared_constant_exception():
    assert _verify(lowered=LOWERED.replace("{merlin.global_task = 0 : i64}", ""))["status"] == "refused"


def test_materialized_orphan_cannot_hide_as_an_unused_kernel_argument():
    buffer = deepcopy(_buffer())
    buffer["tensors"]["orphan"] = {"shape": [2], "dtype": "i32", "role": "input"}
    buffer["kernel_abi"]["args"].insert(0, {"tensor": "orphan"})
    lowered = LOWERED.replace("@kernel(%0: !llvm.ptr)",
                              "@kernel(%orphan: !llvm.ptr, %0: !llvm.ptr)")

    result = _verify(buffer, lowered)

    assert result["status"] == "refused"
    assert "materialized tensors lack complete source or compiler-temporary provenance" in result[
        "problems"]


def test_missing_protocol_is_unknown():
    assert _verify({})["status"] == "UNKNOWN"


def test_absent_transition_is_not_claimed_as_a_copy_proof():
    result = _verify()
    assert result["physical_transition_evidence"]["status"] == "not_declared"
    assert result["timing"] == "UNKNOWN"


@pytest.mark.parametrize("declaration", [[{"id": "fake"}], {"id": "fake"}, None])
def test_transition_declaration_cannot_bypass_actual_address_checks(declaration):
    buffer = _buffer()
    buffer["params"]["global_program_plan"]["physical_transitions"] = declaration
    result = _verify(buffer)
    assert result["status"] == "refused"
    assert result["physical_transition_evidence"]["status"] == "refused"


def test_undeclared_emitted_transition_is_not_silently_ignored():
    lowered = LOWERED.replace("merlin.global_task = 0 : i64",
                              'merlin.global_task = 0 : i64, merlin.global_transition = "hidden"')
    result = _verify(lowered=lowered)
    assert result["status"] == "refused"
    assert result["physical_transition_evidence"]["status"] == "refused"


def test_host_parsed_artifact_is_reused_but_still_verified(monkeypatch):
    from merlin.frontends import linalg_mlir
    from xdsl.dialects.llvm import LLVM
    context = linalg_mlir.make_context()
    context.load_dialect(LLVM)
    parsed = linalg_mlir.parse_mlir_text(LOWERED, context)
    original_parse = linalg_mlir.parse_mlir_text
    parsed_texts = []

    def parse(text, ctx=None):
        parsed_texts.append(text)
        return original_parse(text, ctx)

    monkeypatch.setattr(linalg_mlir, "parse_mlir_text", parse)
    result = verify_compiler_global_plan(source_text=SOURCE, lowered_text=LOWERED,
                                        command_buffer=_buffer(), candidate_sha256="a" * 64,
                                        parsed_lowered_module=parsed)
    assert result["status"] == "verified"
    assert parsed_texts == [SOURCE]


def test_prepared_source_analysis_is_evidence_equivalent(tmp_path: Path):
    source = tmp_path / "model.mlir"
    source.write_text(SOURCE)
    prepared = prepare_captured_source(source)

    reused = verify_compiler_global_plan(
        source_text=SOURCE, lowered_text=LOWERED, command_buffer=_buffer(),
        candidate_sha256="a" * 64, prepared_source_analysis=prepared)

    assert reused == _verify()


def test_prepared_source_analysis_refuses_source_identity_drift(tmp_path: Path):
    source = tmp_path / "model.mlir"
    source.write_text(SOURCE)
    prepared = prepare_captured_source(source)

    with pytest.raises(ValueError, match="does not match source bytes"):
        verify_compiler_global_plan(
            source_text=SOURCE + "\n", lowered_text=LOWERED, command_buffer=_buffer(),
            candidate_sha256="a" * 64, prepared_source_analysis=prepared)


def test_explicit_compiler_scratch_is_structural_not_dtype_semantics():
    buffer = _buffer()
    buffer["tensors"]["scratch"] = {"shape": [2], "dtype": "i64", "role": "intermediate"}
    buffer["kernel_abi"]["args"].append({"tensor": "scratch"})
    plan = buffer["params"]["global_program_plan"]
    plan["compiler_temporaries"] = [{"tensor": "scratch", "source_op_index": 0,
                                    "source_result_index": 0, "purpose": "wide_intermediate"}]
    plan["tasks"][0]["writes"].append("scratch")
    lowered = LOWERED.replace("@kernel(%0: !llvm.ptr)", "@kernel(%0: !llvm.ptr, %scratch: !llvm.ptr)")
    assert _verify(buffer, lowered)["status"] == "verified"
    plan["tasks"][0]["writes"].remove("scratch")
    assert _verify(buffer, lowered)["status"] == "refused"


def test_emitted_task_reordering_is_refused_even_if_receipt_order_is_valid():
    source = SOURCE.replace("func.return %0", "%1 = tensor.cast %0 : tensor<2xi32> to tensor<2xi32>\n    func.return %1")
    buffer = _buffer()
    plan = buffer["params"]["global_program_plan"]
    plan.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(), source_op_count=2,
                schedule_instruction_count=4, epilogue_instruction_range=[3, 4])
    plan["source_values"].append({"op_index": 1, "result_index": 0, "tensor": "result"})
    plan["tasks"].append({"task_index": 1, "kind": "host", "source_op_indices": [1],
                          "instruction_start": 2, "instruction_end": 3,
                          "reads": ["result"], "writes": ["result"]})
    lowered = LOWERED.replace("%2 = llvm.getelementptr", "%early = llvm.getelementptr %0[%1] {merlin.global_task = 1 : i64} : (!llvm.ptr, i64) -> !llvm.ptr, i32\n    %2 = llvm.getelementptr")
    result = verify_compiler_global_plan(source_text=source, lowered_text=lowered,
                                        command_buffer=buffer, candidate_sha256="a" * 64)
    assert result["status"] == "refused"
    assert "kernel block reverses scheduled task order" in result["problems"]


def _encoded_buffer():
    from merlin.perf.storage_encoding import GroupedAxesStorage
    buffer = _buffer()
    buffer["tensors"]["result"]["shape"] = [1, 2]
    buffer["params"]["storage_encodings"] = {
        "result": GroupedAxesStorage((2,), "i32", ((), (0,)), (1, 2), (4, 1), 4).to_dict()}
    return buffer


def test_explicit_encoding_verifies_address_contract_not_consumer_semantics():
    result = _verify(_encoded_buffer())
    assert result["status"] == "verified"
    encoding = result["storage_encodings"]["result"]
    assert encoding["logical_strides_elements"] == [1]
    assert encoding["emitted_consumer_addressing"] == "requires artifact-bound address evidence"
    assert encoding["caller_materialization"] == "requires artifact-bound pack/view evidence"
    # The deliberately incomplete mock kernel still cannot become numerically qualified.
    assert result["numeric_equivalence"] == "requires independent mechanism witnesses"
    assert result["timing"] == "UNKNOWN"
    joined = result["source_plan_metadata"]["physical_storage"]
    assert joined["status"] == "complete"
    assert joined["materialized_source_values"] == joined["exact_physical_representations"] == 1
    assert joined["unknown"] == []
    assert joined["rows"][0]["source_origin"] == {
        "kind": "source_result", "source_operation_id": 0, "result_index": 0}
    assert joined["rows"][0]["logical"] == {"shape": [2], "dtype": "i32"}
    assert joined["rows"][0]["physical_tensor"]["shape"] == [1, 2]
    assert joined["rows"][0]["encoding"].startswith("grouped_axes_storage_v1@sha256:")
    assert joined["rows"][0]["layout"].startswith("static_strided_elements_v1@sha256:")
    assert joined["rows"][0]["layout_contract"] == {
        "axis_groups": [[], [0]], "physical_shape": [1, 2],
        "strides_elements": [4, 1], "storage_elements": 4, "offset_elements": 0,
    }


@pytest.mark.parametrize("mutation", [
    lambda b: b["params"].update(storage_encodings=[]),
    lambda b: b["params"]["storage_encodings"].update(missing={}),
    lambda b: b["params"]["storage_encodings"]["result"].update(dtype="i8"),
    lambda b: b["params"]["storage_encodings"]["result"].update(logical_shape=[1, 2], axis_groups=[[0], [1]]),
    lambda b: b["params"]["storage_encodings"]["result"].update(physical_shape=[2, 1], axis_groups=[[0], []]),
    lambda b: b["params"]["storage_encodings"]["result"].update(storage_elements=1),
    lambda b: b["params"]["storage_encodings"]["result"].update(axis_groups=[[], []]),
    lambda b: b["tensors"]["result"].update(dtype="i8"),
])
def test_invalid_encoding_cannot_waive_source_type_or_storage(mutation):
    buffer = _encoded_buffer()
    mutation(buffer)
    assert _verify(buffer)["status"] == "refused"


def test_scalar_rank_lifting_needs_explicit_encoding():
    from merlin.perf.storage_encoding import GroupedAxesStorage
    source = SOURCE.replace("tensor<2xi32>", "tensor<i32>").replace("dense<[3, 7]>", "dense<3>")
    buffer = _buffer()
    buffer["tensors"]["result"]["shape"] = [1]
    buffer["params"]["global_program_plan"]["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()

    def check():
        return verify_compiler_global_plan(source_text=source, lowered_text=LOWERED,
                                            command_buffer=buffer, candidate_sha256="a" * 64)

    assert check()["status"] == "refused"
    buffer["params"]["storage_encodings"] = {
        "result": GroupedAxesStorage((), "i32", ((),), (1,), (1,), 4).to_dict()}
    assert check()["status"] == "verified"


def test_inherited_family_does_not_turn_host_scalar_work_into_a_contraction():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from xdsl.dialects.builtin import StringAttr
    module = parse_mlir_text(SOURCE)
    constant = next(op for op in module.walk() if op.name == "arith.constant")
    constant.attributes["prov.family"] = StringAttr("contraction")
    source = str(module)
    buffer = _buffer()
    buffer["params"]["global_program_plan"]["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    result = verify_compiler_global_plan(source_text=source, lowered_text=LOWERED,
                                        command_buffer=buffer, candidate_sha256="a" * 64)
    assert result["status"] == "verified"
    assert result["execution_placement"].startswith("UNVERIFIED")
    buffer["params"]["global_program_plan"]["tasks"][0]["kind"] = "contraction"
    assert verify_compiler_global_plan(source_text=source, lowered_text=LOWERED,
                                      command_buffer=buffer, candidate_sha256="a" * 64)["status"] == "refused"


MAC_SOURCE = '''builtin.module {
  func.func @forward(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>, %c: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %r = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d2,d1)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<2x2xi32>, tensor<2x2xi32>) outs(%c : tensor<2x2xi32>) {
    ^bb0(%x: i32, %y: i32, %z: i32):
      %mul = arith.muli %x, %y : i32
      %add = arith.addi %z, %mul : i32
      linalg.yield %add : i32
    } -> tensor<2x2xi32>
    func.return %r : tensor<2x2xi32>
  }
}'''


@pytest.mark.parametrize("kind", ["host", "contraction", "convolution", "adapter_declared_kind"])
@pytest.mark.parametrize("dtype", ["i32", "f32"])
def test_actual_mac_structure_is_not_a_hardware_placement_claim(kind, dtype):
    source = MAC_SOURCE if dtype == "i32" else MAC_SOURCE.replace("i32", "f32").replace("arith.muli", "arith.mulf").replace("arith.addi", "arith.addf")
    buffer = _buffer()
    buffer["tensors"] = {name: {"shape": [2, 2], "dtype": dtype}
                         for name in ("a", "b", "c", "result")}
    buffer["kernel_abi"]["args"] = [{"tensor": name} for name in buffer["tensors"]]
    plan = buffer["params"]["global_program_plan"]
    plan.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(), entry_bindings=["a", "b", "c"])
    plan["tasks"][0].update(kind=kind, reads=["a", "b", "c"])
    lowered = LOWERED.replace("@kernel(%0: !llvm.ptr)",
                             "@kernel(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %0: !llvm.ptr)")
    result = verify_compiler_global_plan(source_text=source, lowered_text=lowered,
                                        command_buffer=buffer, candidate_sha256="a" * 64)
    assert result["status"] == "verified"
    assert result["declared_task_kinds"] == {0: kind}
    assert result["execution_placement"].startswith("UNVERIFIED")


@pytest.mark.parametrize("change", [
    ("linalg.yield %add", "linalg.yield %z"),
    ("arith.addi %z, %mul", "arith.addi %x, %mul"),
    ('"reduction"', '"parallel"'),
    ("arith.muli %x, %y", "arith.muli %x, %x"),
])
def test_mac_source_kind_requires_yielded_recurrence_not_just_opcode_presence(change):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.perf.compiler_plan_evidence import _source_has_multiply_accumulate
    module = parse_mlir_text(MAC_SOURCE.replace(*change))
    op = next(op for op in module.walk() if op.name == "linalg.generic")
    assert not _source_has_multiply_accumulate(op)


def test_source_metadata_finds_exact_integer_narrow_chain_and_task_owner():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.perf.compiler_plan_evidence import _source_has_multiply_accumulate
    from merlin.perf.source_plan_metadata import build_source_plan_metadata
    from merlin.perf.storage_encoding import GroupedAxesStorage
    from merlin.xdsl_dialects.lowering.dispatch_program import lower_model_to_dispatch_program

    source = Path(
        "merlin/contract/capsules/_perf/PB01_no_island_m16k32h16n16/"
        "capsule.interface.mlir").read_text()
    module = parse_mlir_text(source)
    _outlined, graph = lower_model_to_dispatch_program(module, prune=False)
    function = next(op for op in module.body.block.ops if op.name == "func.func")
    block = function.body.block
    operations = [op for op in block.ops if op.name != "func.return"]
    values = {
        block.args[0]: "activation", block.args[1]: "weight0", block.args[2]: "weight1",
        operations[3].results[0]: "accumulator", operations[7].results[0]: "narrow",
        operations[11].results[0]: "result",
    }

    def contract(value):
        shape = list(value.type.get_shape())
        dtype = str(value.type.get_element_type())
        encoding = GroupedAxesStorage(
            tuple(shape), dtype, tuple((axis,) for axis in range(len(shape))), tuple(shape),
            tuple([shape[1], 1]), shape[0] * shape[1]).to_dict()
        return {"contract": encoding,
                "proof_scope": "bounded address map",
                "caller_materialization": "requires evidence",
                "emitted_consumer_addressing": "requires evidence"}

    tensors = {name: {"shape": list(value.type.get_shape()),
                      "dtype": str(value.type.get_element_type()), "role": "intermediate"}
               for value, name in values.items()}
    encodings = {name: contract(value) for value, name in values.items()}
    owners = {index: 0 if index <= 3 else 1 if index <= 7 else 2
              for index in range(len(operations))}
    metadata = build_source_plan_metadata(
        block=block, operations=operations, graph=graph, owners=owners,
        task_kinds={0: "contraction", 1: "host", 2: "contraction"},
        value_tensors=values, tensors=tensors, storage_encodings=encodings,
        is_contraction=_source_has_multiply_accumulate)

    assert metadata["status"] == "verified"
    assert metadata["physical_storage"]["status"] == "complete"
    assert metadata["physical_storage"]["exact_physical_representations"] == 6
    epilogues = metadata["integer_epilogue_ownership"]
    assert epilogues["contraction_roots"] == 2
    assert epilogues["classification_counts"] == {
        "complete_integer_epilogue": 1,
        "partial_integer_epilogue": 0,
        "unclassified": 1,
    }
    chain = next(root for root in epilogues["roots"]
                 if root["classification"] == "complete_integer_epilogue")
    assert chain["producer_source_operation_id"] == 3
    assert chain["source_operation_ids"] == [3, 7]
    assert chain["accumulator_source_buffer"] is not None
    assert chain["stages"][0]["stage"] == "narrow_store"
    assert chain["stages"][0]["operation"] == "saturating_integer_narrow"
    assert chain["stages"][0]["task_index"] == 1
    assert chain["stages"][0]["task_kind"] == "host"
    assert chain["stages"][0]["classification_source"] == "exact_integer_scalar_dag"
    assert chain["stages"][0]["inputs"][0]["role"] == "accumulator"
    assert chain["stages"][0]["inputs"][0]["relation"] == "exact"
    assert len(chain["stages"][0]["semantic_sha256"]) == 64


def test_source_metadata_does_not_call_full_shape_add_a_bias():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.perf.compiler_plan_evidence import _source_has_multiply_accumulate
    from merlin.perf.source_plan_metadata import build_source_plan_metadata
    from merlin.xdsl_dialects.lowering.dispatch_program import lower_model_to_dispatch_program

    source = MAC_SOURCE.replace(
        "func.return %r : tensor<2x2xi32>",
        '''%empty = tensor.empty() : tensor<2x2xi32>
    %sum = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%r, %c : tensor<2x2xi32>, tensor<2x2xi32>) outs(%empty : tensor<2x2xi32>) {
    ^bb1(%x: i32, %y: i32, %unused: i32):
      %added = arith.addi %x, %y : i32
      linalg.yield %added : i32
    } -> tensor<2x2xi32>
    func.return %sum : tensor<2x2xi32>''')
    module = parse_mlir_text(source)
    _outlined, graph = lower_model_to_dispatch_program(module, prune=False)
    function = next(op for op in module.body.block.ops if op.name == "func.func")
    block = function.body.block
    operations = [op for op in block.ops if op.name != "func.return"]
    metadata = build_source_plan_metadata(
        block=block, operations=operations, graph=graph,
        owners={index: 0 for index in range(len(operations))}, task_kinds={0: "host"},
        value_tensors={}, tensors={}, storage_encodings={},
        is_contraction=_source_has_multiply_accumulate)

    root = metadata["integer_epilogue_ownership"]["roots"][0]
    assert root["classification"] == "unclassified"
    assert root["stages"] == []
    assert root["reasons"] == [
        "integer pointwise scalar DAG has no exact supported epilogue identity"]
