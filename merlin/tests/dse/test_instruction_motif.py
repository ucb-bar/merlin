from copy import deepcopy

from merlin.perf.instruction_motif import initialized_compute_primitives
from merlin.targetgen.rocc import decode


def primitive_trace():
    def constant(value):
        return {"kind": "const", "raw": value}
    return {"instructions": [
        {"class": "CONFIG_EX", "funct": 10, "rs1": constant(7), "rs2": constant(9),
         "decoded": {"subtype": "EX"}},
        {"class": "MVIN", "decoded": {"spad_addr": 0, "rows": 4, "cols": 4}},
        {"class": "MVIN", "decoded": {"spad_addr": 8, "rows": 4, "cols": 4}},
        {"class": "PRELOAD", "funct": 11, "rs1": constant(18), "rs2": constant(19),
         "decoded": {"weight_spad": 8, "c_addr": 0, "accumulate": False}},
        {"class": "COMPUTE_PRELOADED", "funct": 12, "rs1": constant(20), "rs2": constant(21),
         "decoded": {"a_spad": 0}},
    ]}


def configure(monkeypatch, selector=10):
    monkeypatch.setattr(decode, "isa_constants", lambda target: {
        "CONFIG_SUBTYPE": {selector: "CONFIG_EX"}, "CUSTOM_OPCODE": selector})


def test_actual_compute_payload_and_initialized_operands_define_domain(monkeypatch):
    configure(monkeypatch)
    trace = primitive_trace()
    first = initialized_compute_primitives(trace, target="test-device")[0]
    assert not first["missing"]
    assert first["domain_digest"]
    assert not first["calibration_admissible"]
    assert first["in_context_cycles"] is None
    assert first["initialization_provenance"][0]["producer_index"] == 1
    other = deepcopy(trace)
    other["instructions"][-1]["rs1"]["raw"] += 1
    second = initialized_compute_primitives(other, target="test-device")[0]
    assert first["domain_digest"] != second["domain_digest"]


def test_partial_overwrite_blocks_initialized_tile_proof(monkeypatch):
    configure(monkeypatch)
    trace = primitive_trace()
    trace["instructions"].insert(3, {"class": "MVIN", "decoded": {
        "spad_addr": 9, "rows": 1, "cols": 4}})
    result = initialized_compute_primitives(trace, target="test-device")[0]
    assert any("partially overwritten" in missing for missing in result["missing"])
    assert result["domain_digest"] is None


def test_accumulation_requires_initial_value_witness(monkeypatch):
    configure(monkeypatch)
    trace = primitive_trace()
    trace["instructions"][-2]["decoded"]["accumulate"] = True
    result = initialized_compute_primitives(trace, target="test-device")[0]
    assert any("accumulator initial" in missing for missing in result["missing"])
    assert result["domain_digest"] is None


def test_retained_weight_and_absent_load_do_not_count_as_initialized(monkeypatch):
    configure(monkeypatch)
    trace = primitive_trace()
    trace["instructions"][-2]["decoded"]["weight_spad"] = decode.GARBAGE
    result = initialized_compute_primitives(trace, target="test-device")[0]
    assert any("retained internal state" in missing for missing in result["missing"])
    trace = primitive_trace()
    del trace["instructions"][1]
    result = initialized_compute_primitives(trace, target="test-device")[0]
    assert any("no observed producer" in missing for missing in result["missing"])


def test_target_revision_changes_domain_and_absent_configuration_refuses(monkeypatch):
    configure(monkeypatch)
    first = initialized_compute_primitives(primitive_trace(), target="test-device")[0]
    configure(monkeypatch, selector=100)
    second = initialized_compute_primitives(primitive_trace(), target="test-device")[0]
    assert first["domain_digest"] != second["domain_digest"]
    trace = primitive_trace()
    del trace["instructions"][0]
    result = initialized_compute_primitives(trace, target="test-device")[0]
    assert any("configuration is unresolved" in missing for missing in result["missing"])


def test_unknown_instruction_invalidates_prior_initialization(monkeypatch):
    configure(monkeypatch)
    trace = primitive_trace()
    trace["instructions"].insert(3, {"class": "UNKNOWN", "decoded": {}})
    result = initialized_compute_primitives(trace, target="test-device")[0]
    assert result["domain_digest"] is None
    assert any("no observed producer" in missing for missing in result["missing"])


def compiled_task_fixture(monkeypatch, *, operand_capacity=8, unknown_host=False):
    """Real LLVM SSA/assembly parse against a small independently supplied target fact set."""
    from types import SimpleNamespace
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i64
    from xdsl.ir import Block, Region
    from merlin.targetgen import address_space
    from merlin.targetgen.address_space import Store
    from merlin.perf.instruction_motif import extract_task_instruction_motif

    isa = {"DIM": 4, "CUSTOM_OPCODE": 11, "FUNCT3": 3,
           "ACC_I8": 1 << 31, "C_ACC": (1 << 31) | (1 << 29),
           "ACC_ACCUM": 1 << 30, "FULL_C_BIT": 1 << 29,
           "CONFIG_SUBTYPE": {0: "CONFIG_EX", 1: "CONFIG_LD", 2: "CONFIG_ST"},
           "FUNCT_CLASS": {0: "CONFIG", 2: "MVIN", 3: "MVOUT", 4: "COMPUTE_PRELOADED",
                           6: "PRELOAD"}}
    monkeypatch.setattr(decode, "isa_constants", lambda target: isa)
    stores = [Store("operand-sram", operand_capacity * 4, operand_capacity, 4, "i8", 8,
                    4, operand_capacity, 1),
              Store("result-sram", 128, 8, 4, "i32", 32, 16, 8, 1)]
    monkeypatch.setattr(address_space, "derive_address_space", lambda target: SimpleNamespace(stores=stores))
    ptr = llvm.LLVMPointerType()
    block = Block(arg_types=[ptr, ptr, ptr])
    def add(op):
        op.attributes["merlin.global_task"] = IntegerAttr(0, i64)
        block.add_op(op)
        return op.results[0] if op.results else None
    def const(n):
        return add(llvm.ConstantOp(IntegerAttr(n, i64), i64))
    def fence():
        add(llvm.InlineAsmOp("fence", "~{memory}", [], [], has_side_effects=True))
    def issue(funct, left, right):
        add(llvm.InlineAsmOp(f".insn r {isa['CUSTOM_OPCODE']}, 3, {funct}, x0, $0, $1",
                            "r,r", [left, right], [], has_side_effects=True))
    def packed(addr):
        return const((4 << 48) | (4 << 32) | addr)
    fence()
    issue(0, const(0), const(0))
    issue(0, const(1), const(4))
    issue(2, add(llvm.PtrToIntOp(block.args[0], i64)), packed(0))
    issue(2, add(llvm.PtrToIntOp(block.args[1], i64)), packed(4))
    issue(6, packed(4), packed(isa["ACC_I8"]))
    issue(4, packed(0), packed(decode.GARBAGE))
    issue(0, const(2), const(4))
    issue(3, add(llvm.PtrToIntOp(block.args[2], i64)), packed(isa["ACC_I8"]))
    if unknown_host:
        add(llvm.LoadOp(block.args[0], i64))
    fence()
    block.add_op(llvm.ReturnOp())
    module = ModuleOp([llvm.FuncOp("kernel", llvm.LLVMFunctionType([ptr, ptr, ptr]),
                                   body=Region([block]))])
    module.verify()
    cb = {"params": {"global_program_plan": {"tasks": [{"task_index": 0, "kind": "contraction",
                                                           "reads": ["A", "W"], "writes": ["Y"]}]}},
          "kernel_abi": {"args": [{"tensor": name} for name in ("A", "W", "Y")]},
          "tensors": {name: {"dtype": "i8", "shape": [4, 4]} for name in ("A", "W", "Y")}}
    return extract_task_instruction_motif(artifact=str(module).encode(), command_buffer=cb,
                                          target="derived-small-engine", task_index=0)


def test_structural_llvm_extractor_derives_capacity_and_keeps_timing_unknown(monkeypatch):
    result = compiled_task_fixture(monkeypatch)
    assert result.signature is not None, result.missing
    assert result.signature.to_dict()["capacity_regime"] == {
        "operand-sram": "fits_single", "result-sram": "fits_double"}
    assert result.facts["device_boundary_fences"] == {"entry": True, "exit": True}
    assert result.timing_context_missing


def test_extractor_rejects_out_of_capacity_or_hidden_host_work(monkeypatch):
    result = compiled_task_fixture(monkeypatch, operand_capacity=6)
    assert result.signature is None
    assert any("exceeds derived" in item for item in result.missing)
    result = compiled_task_fixture(monkeypatch, unknown_host=True)
    assert result.signature is None
    assert any("non-command work" in item for item in result.missing)
