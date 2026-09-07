"""A movement context retains competing commands instead of laundering isolated timing."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from merlin.perf.context_probe import extract_queued_movement_context
from merlin.kernels.decode import rocc
from merlin.kernels import endpoints
from merlin.targetgen.rocc import decode


@pytest.fixture
def context_source(monkeypatch):
    isa = {"CONFIG_SUBTYPE": {0: "CONFIG_EX", 1: "CONFIG_LD"}, "revision": "test-abi"}
    monkeypatch.setattr(decode, "isa_constants", lambda target: isa)
    monkeypatch.setattr(rocc, "funct_table_for", lambda target: {
        "names": {0: "ex_config", 1: "load_config", 2: "load", 3: "stage", 4: "compute"}})
    monkeypatch.setattr(endpoints, "endpoints_for", lambda target: [SimpleNamespace(
        name="declared_endpoint", engine="unsplit_engine", source="test-derived-abi",
        roles_of=lambda name: {"load": ("operand_load",), "compute": ("accumulate",),
                               "stage": ("weight_load",), "ex_config": ("configure",),
                               "load_config": ("configure",)}.get(name, ()))])

    def constant(value):
        return {"kind": "const", "raw": value}

    def load(address, argument):
        pointer = {"kind": "argbase", "arg_index": argument, "offset": 0}
        return {"class": "MVIN", "funct": 2, "rs1": pointer, "rs2": constant(address),
                "decoded": {"spad_addr": address, "rows": 4, "cols": 4, "dram": pointer}}

    trace = {"instructions": [
        {"class": "FENCE", "decoded": {}},
        {"class": "CONFIG_EX", "funct": 0, "rs1": constant(1), "rs2": constant(2),
         "decoded": {"subtype": "EX"}},
        {"class": "CONFIG_LD", "funct": 1, "rs1": constant(3), "rs2": constant(4),
         "decoded": {"subtype": "LD", "stride": 4}},
        load(0, 0), load(8, 1), load(16, 2),
        {"class": "PRELOAD", "funct": 3, "rs1": constant(5), "rs2": constant(6),
         "decoded": {"weight_spad": 8, "c_addr": 0, "accumulate": False}},
        {"class": "COMPUTE_PRELOADED", "funct": 4, "rs1": constant(7), "rs2": constant(8),
         "decoded": {"a_spad": 0}},
    ]}
    buffer = {"kernel_abi": {"args": [{"tensor": name} for name in ("a", "b", "other")]},
              "tensors": {name: {"dtype": "i8"} for name in ("a", "b", "other")}}
    text = "host-retained source fixture"
    kwargs = {"target": "test-device", "artifact_text": text,
              "artifact_sha256": hashlib.sha256(text.encode()).hexdigest(), "command_buffer": buffer}
    return trace, kwargs


def test_context_keeps_actual_competing_transfer_and_operand_loads_in_roi(context_source):
    trace, kwargs = context_source
    result = extract_queued_movement_context(trace, **kwargs)
    motif = result["motifs"][0]
    assert motif["instruction_indices"] == [3, 4, 5, 6, 7]
    assert motif["queued_competing_movement_indices"] == [5]
    assert motif["competing_movements_disjoint_from_operand_rows"] == [5]
    assert [(edge["before"], edge["after"]) for edge in motif["dependencies"]] == [(3, 7), (4, 7)]
    assert motif["command_empty_entry_observed"]
    assert motif["transfers"][2]["tensor"] == "other"
    assert motif["transfers"][2]["load_configuration"]["stride"] == 4
    assert motif["resource_bindings"][-1]["endpoints"][0]["declared_engine"] == "unsplit_engine"
    assert not motif["calibration_admissible"] and not motif["physical_parallelism_proven"]
    assert motif["in_context_cycles"] is None and motif["overlap_observed"] is None
    assert any("task/source mapping" in item for item in motif["state_missing"])


def test_wrong_artifact_identity_fails(context_source):
    trace, kwargs = context_source
    kwargs["artifact_text"] += "changed"
    with pytest.raises(ValueError, match="artifact hash"):
        extract_queued_movement_context(trace, **kwargs)


def test_isolated_primitive_is_not_relabelled_as_contention_context(context_source):
    trace, kwargs = context_source
    del trace["instructions"][5]
    result = extract_queued_movement_context(trace, **kwargs)
    assert not result["motifs"]
    assert any("no additional queued movement" in reason for row in result["unsupported"] for reason in row["missing"])


def test_unresolved_state_or_excessive_window_is_not_admitted(context_source):
    trace, kwargs = context_source
    limited = extract_queued_movement_context(trace, max_commands=4, **kwargs)
    assert not limited["motifs"]
    trace["instructions"][5]["decoded"]["spad_addr"] = 1
    overwritten = extract_queued_movement_context(trace, **kwargs)
    assert not overwritten["motifs"]
    assert any("partially overwritten" in reason for row in overwritten["unsupported"] for reason in row["missing"])


def test_exact_movement_encoding_changes_context_identity(context_source):
    trace, kwargs = context_source
    first = extract_queued_movement_context(trace, **kwargs)["motifs"][0]
    changed = deepcopy(trace)
    changed["instructions"][5]["rs1"]["offset"] = 64
    second = extract_queued_movement_context(changed, **kwargs)["motifs"][0]
    assert first["primitive_domain_digest"] == second["primitive_domain_digest"]
    assert first["context_shape_sha256"] != second["context_shape_sha256"]


def test_source_queued_entry_remains_unresolved(context_source):
    trace, kwargs = context_source
    trace["instructions"].insert(3, deepcopy(trace["instructions"][5]))
    result = extract_queued_movement_context(trace, **kwargs)
    motif = result["motifs"][0]
    assert not motif["command_empty_entry_observed"]
    assert any("queued commands preceding" in reason for reason in motif["state_missing"])


def test_interleaved_host_store_is_not_dropped_from_context(context_source, monkeypatch):
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp, i32
    from xdsl.ir import Block, Region
    trace, kwargs = context_source
    block = Block(arg_types=[llvm.LLVMPointerType(), i32])
    for index, row in enumerate(trace["instructions"]):
        if index == 6:
            block.add_op(llvm.StoreOp(block.args[1], block.args[0]))
        block.add_op(llvm.InlineAsmOp(row["class"], "", [], [], has_side_effects=True))
    block.add_op(llvm.ReturnOp())
    module = ModuleOp([llvm.FuncOp("context", llvm.LLVMFunctionType([llvm.LLVMPointerType(), i32]),
                                  body=Region([block]))])
    # Real IR work/command interleaving is inspected; decoding is independent of this host-work check.
    monkeypatch.setattr(decode, "decode_module", lambda *args, **kwargs: trace)
    result = extract_queued_movement_context(trace, parsed_module=module, **kwargs)
    assert any("non-command host work" in reason and "llvm.store" in reason
               for reason in result["motifs"][0]["state_missing"])
