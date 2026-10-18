"""Batched copy proof agrees with exact scalar execution and rejects changed effects."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.perf.host_cfg_activity import analyze_host_cfg_activity
from merlin.targetgen.rocc.decode import _parse_module
from merlin.verify.conv_input_preparation import prove_input_preparation
from merlin.verify.host_compute_audit import _functions
from merlin.verify.input_copy_storage import CopyStorage


def _program(rows, columns, *, padding=True):
    return f"""module {{
      llvm.func @kernel(%input: !llvm.ptr, %weight: !llvm.ptr, %output: !llvm.ptr) {{
        %zero = llvm.mlir.constant(0) : i64
        %one = llvm.mlir.constant(1) : i64
        %rows = llvm.mlir.constant({rows}) : i64
        %columns = llvm.mlir.constant({columns}) : i64
        %padding = llvm.mlir.constant({columns - (1 if padding else 0)}) : i64
        %size = llvm.mlir.constant({rows * columns}) : i64
        %bytezero = llvm.mlir.constant(0 : i8) : i8
        %scratch = llvm.alloca %size x i8 : (i64) -> !llvm.ptr
        llvm.br ^outer(%zero : i64)
      ^outer(%i: i64):
        %row = llvm.mul %i, %columns : i64
        llvm.br ^inner(%zero : i64)
      ^inner(%j: i64):
        %offset = llvm.add %row, %j : i64
        %destination = llvm.getelementptr %scratch[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %valid = llvm.icmp "ult" %j, %padding : i64
        llvm.cond_br %valid, ^copy, ^pad
      ^copy:
        %source = llvm.getelementptr %input[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %byte = llvm.load %source : !llvm.ptr -> i8
        llvm.br ^latch(%byte : i8)
      ^pad:
        llvm.br ^latch(%bytezero : i8)
      ^latch(%chosen: i8):
        llvm.store %chosen, %destination : i8, !llvm.ptr
        %next = llvm.add %j, %one : i64
        %done = llvm.icmp "eq" %next, %columns : i64
        llvm.cond_br %done, ^outer_latch, ^inner(%next : i64)
      ^outer_latch:
        %nextrow = llvm.add %i, %one : i64
        %finished = llvm.icmp "eq" %nextrow, %rows : i64
        llvm.cond_br %finished, ^transfer, ^outer(%nextrow : i64)
      ^transfer:
        %address = llvm.ptrtoint %scratch : !llvm.ptr to i64
        "llvm.inline_asm"(%size, %size) <{{asm_string = ".insn r 0x7b, 3, 0, x0, $0, $1",
          constraints = "r,r,~{{memory}}", has_side_effects}}> : (i64, i64) -> ()
        "llvm.inline_asm"(%address, %size) <{{asm_string = ".insn r 0x7b, 3, 2, x0, $0, $1",
          constraints = "r,r,~{{memory}}", has_side_effects}}> : (i64, i64) -> ()
        llvm.return
      }}
    }}"""


def _proof(monkeypatch, text, size, *, batch=True, weight_bytes=1):
    from merlin.targetgen.rocc import decode

    observer = SimpleNamespace(
        word_bits=64,
        isa_constants=lambda _: {"CUSTOM_OPCODE": 0x7B},
        decode_instruction=lambda funct, rs1, rs2, _: (
            ("CONFIG_LD", {"stride": size, "state_id": 0}) if funct == 0
            else ("CONFIG_ST", {"out_stride_bytes": size * 4}) if funct == 4
            else ("MVOUT", {"dram": rs1, "rows": 1, "cols": size, "readout": "i32"}) if funct == 3
            else ("MVIN", {"dram": rs1, "rows": 1, "cols": size})
        ),
    )
    monkeypatch.setattr(decode, "_semantics", lambda _: SimpleNamespace(operand_observer=observer))
    function = _functions(_parse_module(text))["kernel"]
    context = {"call_buffers": [
        {"role": "input_0", "shape": [size], "dtype": "i8"},
        {"role": "weight_0", "shape": [weight_bytes, 1], "dtype": "i8"},
        {"role": "out_0", "shape": [size, 1], "dtype": "i32"},
    ]}
    return prove_input_preparation(
        function, analyze_host_cfg_activity(function), context, target="fixture", batch_copies=batch
    )


@pytest.mark.parametrize("rows,columns,padding", [(7, 257, True), (9, 131, False), (2, 509, True)])
def test_batched_rectangular_copy_preserves_exact_scalar_counts(monkeypatch, rows, columns, padding):
    text = _program(rows, columns, padding=padding)
    batch = _proof(monkeypatch, text, rows * columns)
    scalar = _proof(monkeypatch, text, rows * columns, batch=False)
    assert batch["status"] == scalar["status"] == "proven", (batch, scalar)
    assert batch.pop("batched_copy_regions") == 1
    assert scalar.pop("batched_copy_regions") == 0
    assert batch == scalar


def test_large_copy_fits_bounded_batching_without_expanding_scalar_budget(monkeypatch):
    rows, columns = 256, 512
    text = _program(rows, columns)
    result = _proof(monkeypatch, text, rows * columns)
    assert result["status"] == "proven" and result["batched_copy_regions"] == 1, result
    scalar = _proof(monkeypatch, text, rows * columns, batch=False)
    assert scalar["status"] == "unproven" and "finite scalar" in scalar["detail"]


def test_vectorization_does_not_remove_the_cumulative_payload_proof_budget(monkeypatch):
    from merlin.verify import conv_input_preparation

    monkeypatch.setattr(conv_input_preparation, "MAX_OPERATIONS", 2000)
    result = _proof(monkeypatch, _program(7, 257), 7 * 257)
    assert result["status"] == "unproven" and "finite exhaustive copy proof budget" in result["detail"]


def test_latch_testing_current_induction_preserves_exact_trip_count(monkeypatch):
    rows, columns = 7, 257
    text = _program(rows, columns).replace(
        "%rows =", f"%lastrow = llvm.mlir.constant({rows - 1}) : i64\n %rows ="
    ).replace("%columns =", f"%lastcol = llvm.mlir.constant({columns - 1}) : i64\n %columns =")
    text = text.replace('llvm.icmp "eq" %next, %columns', 'llvm.icmp "eq" %j, %lastcol')
    text = text.replace('llvm.icmp "eq" %nextrow, %rows', 'llvm.icmp "eq" %i, %lastrow')
    result = _proof(monkeypatch, text, rows * columns)
    scalar = _proof(monkeypatch, text, rows * columns, batch=False)
    assert result["status"] == scalar["status"] == "proven", (result, scalar)
    assert result.pop("batched_copy_regions") == 1
    scalar.pop("batched_copy_regions")
    assert result == scalar


def test_disjoint_sibling_loops_share_axes_without_multiplying_iteration_domains(monkeypatch):
    rows, columns = 7, 257
    text = _program(rows, columns)
    start, end = text.index("      ^inner("), text.index("      ^outer_latch:")
    prefix, body, suffix = text[:start], text[start:end], text[end:]

    def renamed(tag):
        renamed = body.replace("%bytezero", "BYTE_ZERO_OUTSIDE_LOOP")
        for name in ("j", "offset", "destination", "valid", "source", "byte", "chosen", "next", "done"):
            renamed = renamed.replace("%" + name, "%" + name + tag)
        for name in ("inner", "copy", "pad", "latch"):
            renamed = renamed.replace("^" + name, "^" + name + tag)
        return renamed.replace("BYTE_ZERO_OUTSIDE_LOOP", "%bytezero")

    prefix = prefix.replace("llvm.br ^inner(%zero : i64)", """
      %choice = llvm.icmp "ult" %i, %one : i64
      llvm.cond_br %choice, ^innerA(%zero : i64), ^innerB(%zero : i64)""")
    text = prefix + renamed("A") + renamed("B") + suffix
    result = _proof(monkeypatch, text, rows * columns)
    scalar = _proof(monkeypatch, text, rows * columns, batch=False)
    assert result["status"] == scalar["status"] == "proven", (result, scalar)
    assert result.pop("batched_copy_regions") == 1
    scalar.pop("batched_copy_regions")
    assert result == scalar


def test_weights_can_be_repacked_only_as_unchanged_consumed_bytes(monkeypatch):
    rows, columns = 7, 257
    text = _program(rows, columns).replace("%input[%offset]", "%weight[%offset]")
    result = _proof(monkeypatch, text, rows * columns, weight_bytes=rows * columns)
    assert result["status"] == "proven", result
    changed = text.replace("llvm.br ^latch(%byte : i8)",
                           "%changed = llvm.add %byte, %byte : i8\n llvm.br ^latch(%changed : i8)")
    assert _proof(monkeypatch, changed, rows * columns, weight_bytes=rows * columns)["status"] == "unproven"
    memory = CopyStorage({0: 100, 1: 50, "scratch": 2})
    memory["scratch", 0] = (0, 7)
    memory["scratch", 1] = (1, 7)
    assert memory["scratch", 0] == (0, 7)
    assert memory["scratch", 1] == (1, 7)


@pytest.mark.parametrize("mutation", [
    lambda s: s.replace("llvm.store %chosen, %destination", "llvm.store %chosen, %output"),
    lambda s: s.replace("%input[%offset]", "%input[%size]"),
    lambda s: s.replace("%scratch[%offset]", "%scratch[%zero]"),
    lambda s: s.replace("llvm.br ^latch(%byte : i8)",
                        "%changed = llvm.add %byte, %byte : i8\n llvm.br ^latch(%changed : i8)"),
    lambda s: s.replace('llvm.icmp "ult" %j, %padding : i64',
                        'llvm.icmp "ult" %j, %size : i64').replace("%input[%offset]", "%weight[%offset]"),
])
def test_batched_copy_rejects_tensor_compute_wrong_buffers_bounds_and_conflicting_writes(monkeypatch, mutation):
    result = _proof(monkeypatch, mutation(_program(7, 257)), 7 * 257)
    assert result["status"] == "unproven", result


def test_byte_initialization_and_consumption_are_bound_to_each_write_epoch():
    memory = CopyStorage({"scratch": 4})
    memory["scratch", 0] = (0, 7)
    with pytest.raises(ValueError, match="overwritten before"):
        memory["scratch", 0] = (0, 8)
    memory.consume("scratch", np.array([0]))
    memory.require_consumed()
    memory["scratch", 0] = (0, 8)
    with pytest.raises(ValueError, match="never feeds"):
        memory.require_consumed()
    with pytest.raises(ValueError, match="uninitialized"):
        memory.consume("scratch", np.array([1]))


def test_repeated_load_rows_do_not_expand_a_cartesian_byte_roster():
    memory = CopyStorage({"scratch": 16})
    for i in range(16):
        memory["scratch", i] = (0, i)
    memory.consume_rows("scratch", 0, 1000000000, 16, 0)
    memory.require_consumed()


def test_scratch_forwarding_tracks_epochs_and_requires_a_real_consumed_destination():
    memory = CopyStorage({0: 8, 1: 8, 2: 32, "first": 4, "second": 4})
    memory["first", 0] = (0, 3)
    origins, reference = memory.read_batch("first", np.array([0]))
    # Reading alone cannot discharge the original write.
    with pytest.raises(ValueError, match="never feeds"):
        memory.require_consumed()
    memory.store_batch("second", np.array([0]), origins, {})
    memory.forward(reference)
    # Forwarding alone still leaves a required final accelerator consumer.
    with pytest.raises(ValueError, match="never feeds"):
        memory.require_consumed()
    memory.consume("second", np.array([0]))
    memory.require_consumed()
    memory["first", 0] = (0, 3)
    with pytest.raises(ValueError, match="changed source write epoch"):
        memory.forward(reference)
    with pytest.raises(ValueError, match="never feeds"):
        memory.require_consumed()


@pytest.mark.parametrize("write_first", [False, True])
def test_batched_scratch_reads_and_writes_cannot_be_reordered(write_first):
    memory = CopyStorage({"scratch": 4})
    memory["scratch", 0] = (0, 3)
    memory.consume("scratch", np.array([0]))
    written, read = {}, {}
    if write_first:
        memory.store_batch("scratch", np.array([0]), np.array([3]), written, read)
        with pytest.raises(ValueError, match="conflicting scratch reads"):
            memory.read_batch("scratch", np.array([0]), written, read)
    else:
        memory.read_batch("scratch", np.array([0]), written, read)
        with pytest.raises(ValueError, match="conflicting scratch reads"):
            memory.store_batch("scratch", np.array([0]), np.array([3]), written, read)


def test_result_publication_requires_actual_decoded_writes_and_an_intervening_fence():
    memory = CopyStorage({0: 4, 1: 4, 2: 16, "tile": 16})
    with pytest.raises(ValueError, match="accelerator result bytes"):
        memory.publish_output(np.arange(4), np.arange(4))
    memory.result_rows("tile", 0, 2, 8, 8)
    with pytest.raises(ValueError, match="before the command fence"):
        memory.read_batch("tile", np.arange(16))
    memory.pending_results.clear()
    origins, reference = memory.read_batch("tile", np.arange(16))
    with pytest.raises(ValueError, match="output-result bytes"):
        memory.consume("tile", np.arange(16))
    memory.publish_output(np.arange(16), origins)
    memory.forward(reference)
    memory.require_consumed()
    with pytest.raises(ValueError, match="repeats a destination"):
        memory.publish_output(np.arange(16), origins)


@pytest.mark.parametrize("pad_forward", [False, True])
def test_two_stage_input_packing_matches_scalar_execution(monkeypatch, pad_forward):
    rows, columns = 7, 257
    text = _program(rows, columns)
    text = text.replace("llvm.br ^outer(%zero : i64)", """
        %staging = llvm.alloca %size x i8 : (i64) -> !llvm.ptr
        llvm.br ^outer(%zero : i64)""")
    text = text.replace('^transfer, ^outer(', '^repack, ^outer(')
    text = text.replace("      ^transfer:", """
      ^repack:
        llvm.br ^move(%zero : i64)
      ^move(%index: i64):
        %from = llvm.getelementptr %scratch[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %to = llvm.getelementptr %staging[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %copied = llvm.load %from : !llvm.ptr -> i8
        llvm.store %copied, %to : i8, !llvm.ptr
        %incremented = llvm.add %index, %one : i64
        %last = llvm.icmp "eq" %incremented, %size : i64
        llvm.cond_br %last, ^transfer, ^move(%incremented : i64)
      ^transfer:""")
    text = text.replace("%address = llvm.ptrtoint %scratch", "%address = llvm.ptrtoint %staging")
    if pad_forward:
        text = text.replace("%staging =", "%lastvalid = llvm.sub %size, %one : i64\n %staging =")
        text = text.replace("%copied = llvm.load %from : !llvm.ptr -> i8\n        llvm.store %copied, %to", """
        %movevalid = llvm.icmp "ult" %index, %lastvalid : i64
        llvm.cond_br %movevalid, ^forward, ^fill
      ^forward:
        %copied = llvm.load %from : !llvm.ptr -> i8
        llvm.br ^move_latch(%copied : i8)
      ^fill:
        llvm.br ^move_latch(%bytezero : i8)
      ^move_latch(%forwarded: i8):
        llvm.store %forwarded, %to""")
    result = _proof(monkeypatch, text, rows * columns)
    scalar = _proof(monkeypatch, text, rows * columns, batch=False)
    assert result["status"] == scalar["status"] == "proven", (result, scalar)
    assert result.pop("batched_copy_regions") == 2
    scalar.pop("batched_copy_regions")
    assert result == scalar
    if pad_forward:
        # Dropping one more lane would strand a real original-byte copy.
        changed = text.replace('%lastvalid = llvm.sub %size, %one : i64',
                               '%shorter = llvm.sub %size, %one : i64\n %lastvalid = llvm.sub %shorter, %one : i64')
        for batch in (True, False):
            refused = _proof(monkeypatch, changed, rows * columns, batch=batch)
            assert refused["status"] == "unproven" and "never feeds" in refused["detail"], refused


def _result_program():
    text = _program(7, 257)
    text = text.replace("llvm.br ^outer(%zero : i64)", """
        %four = llvm.mlir.constant(4) : i64
        %resultsize = llvm.mul %size, %four : i64
        %resulttile = llvm.alloca %resultsize x i8 : (i64) -> !llvm.ptr
        llvm.br ^outer(%zero : i64)""")
    return text.replace("        llvm.return", """
        %resultaddress = llvm.ptrtoint %resulttile : !llvm.ptr to i64
        "llvm.inline_asm"(%resultsize, %resultsize) <{asm_string = ".insn r 0x7b, 3, 4, x0, $0, $1",
          constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()
        "llvm.inline_asm"(%resultaddress, %resultsize) <{asm_string = ".insn r 0x7b, 3, 3, x0, $0, $1",
          constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()
        "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects}> : () -> ()
        llvm.br ^publish(%zero : i64)
      ^publish(%element: i64):
        %position = llvm.mul %element, %four : i64
        %resultfrom = llvm.getelementptr %resulttile[%position] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %resultto = llvm.getelementptr %output[%position] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %resultbyte = llvm.load %resultfrom : !llvm.ptr -> i32
        llvm.store %resultbyte, %resultto : i32, !llvm.ptr
        %nextresult = llvm.add %element, %one : i64
        %endresult = llvm.icmp "eq" %nextresult, %size : i64
        llvm.cond_br %endresult, ^finish, ^publish(%nextresult : i64)
      ^finish:
        llvm.return""")


def test_result_tile_publication_matches_scalar_execution(monkeypatch):
    result = _proof(monkeypatch, _result_program(), 7 * 257)
    scalar = _proof(monkeypatch, _result_program(), 7 * 257, batch=False)
    assert result["status"] == scalar["status"] == "proven", (result, scalar)
    assert result.pop("batched_copy_regions") == 2
    scalar.pop("batched_copy_regions")
    assert result == scalar


@pytest.mark.parametrize("mutation", [
    lambda s: s.replace('"llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects}> : () -> ()', ''),
    lambda s: s.replace('%resulttile[%position]', '%input[%position]'),
    lambda s: s.replace('llvm.store %resultbyte, %resultto',
                        '%changedresult = llvm.add %resultbyte, %resultbyte : i32\n llvm.store %changedresult, %resultto'),
    lambda s: s.replace('%output[%position]', '%output[%zero]'),
])
def test_output_transport_still_rejects_missing_fences_wrong_sources_arithmetic_and_overlap(monkeypatch, mutation):
    assert _proof(monkeypatch, mutation(_result_program()), 7 * 257)["status"] == "unproven"


def test_fixed_width_payload_copy_preserves_every_byte(monkeypatch):
    rows, columns = 9, 131
    text = _program(rows, columns)
    # Keep the original byte loop, but copy adjacent two-byte words at disjoint
    # even addresses and transfer their complete initialized union.
    text = text.replace(f"llvm.mlir.constant({rows * columns}) : i64",
                        f"llvm.mlir.constant({rows * columns * 2}) : i64")
    text = text.replace("%one =", "%two = llvm.mlir.constant(2) : i64\n %one =")
    text = text.replace("%offset = llvm.add %row, %j : i64",
                        "%element = llvm.add %row, %j : i64\n %offset = llvm.mul %element, %two : i64")
    text = text.replace("%bytezero = llvm.mlir.constant(0 : i8) : i8",
                        "%bytezero = llvm.mlir.constant(0 : i16) : i16")
    text = text.replace("llvm.load %source : !llvm.ptr -> i8", "llvm.load %source : !llvm.ptr -> i16")
    text = text.replace("^latch(%byte : i8)", "^latch(%byte : i16)")
    text = text.replace("^latch(%bytezero : i8)", "^latch(%bytezero : i16)")
    text = text.replace("^latch(%chosen: i8)", "^latch(%chosen: i16)")
    text = text.replace("llvm.store %chosen, %destination : i8", "llvm.store %chosen, %destination : i16")
    result = _proof(monkeypatch, text, rows * columns * 2)
    scalar = _proof(monkeypatch, text, rows * columns * 2, batch=False)
    assert result["status"] == scalar["status"] == "proven", (result, scalar)
    assert result.pop("batched_copy_regions") == 1
    scalar.pop("batched_copy_regions")
    assert result == scalar


def test_only_padding_free_byte_array_gep_layouts_are_derived():
    from xdsl.dialects.builtin import i8, i32
    from xdsl.dialects.llvm import LLVMArrayType

    from merlin.verify.input_copy_storage import byte_gep_stride

    assert byte_gep_stride(LLVMArrayType(4, i8)) == 4
    assert byte_gep_stride(LLVMArrayType(2, LLVMArrayType(3, i8))) == 6
    with pytest.raises(ValueError, match="pointer layout"):
        byte_gep_stride(LLVMArrayType(4, i32))


def test_integer_batching_matches_scalar_widths_and_proves_poison_flags():
    from merlin.perf.constant_control_cfg import integer_operation
    from merlin.verify.input_copy_integer import integer

    text = """module { llvm.func @test(%a: i8, %b: i8) {
      %sum = llvm.add %a, %b : i8
      %compare = llvm.icmp "ult" %a, %b : i8
      %cast = llvm.zext %a : i8 to i64
      llvm.return
    }}"""
    function = _functions(_parse_module(text))["test"]
    operations = [op for op in function.walk() if op.name in {"llvm.add", "llvm.icmp", "llvm.zext"}]
    left = np.array([-128, -1, 0, 1, 127], dtype=np.int64)
    right = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    for op in operations:
        args = [left, right][:len(op.operands)]
        actual = integer(op, args, np.ones(5, dtype=np.bool_))
        expected = [integer_operation(op, [int(arg[i]) for arg in args]) for i in range(5)]
        assert actual.tolist() == expected
    from xdsl.dialects.builtin import IntegerAttr

    addition = deepcopy(operations[0])
    addition.properties["overflowFlags"] = IntegerAttr(1, 32)
    with pytest.raises(ValueError, match="signed no-wrap"):
        integer(addition, [left, right], np.ones(5, dtype=np.bool_))


def test_disjoint_or_requires_proven_nonoverlapping_bits_in_every_active_lane():
    from merlin.perf.constant_control_cfg import integer_operation
    from merlin.verify.input_copy_integer import integer

    module = _parse_module("""module { llvm.func @test(%a: i64, %b: i64) {
      %x = "llvm.or"(%a, %b) <{isDisjoint}> : (i64, i64) -> i64
      llvm.return
    }}""")
    op = next(op for op in module.walk() if op.name == "llvm.or")
    assert integer_operation(op, [16, 3]) == 19
    with pytest.raises(ValueError, match="disjoint-or"):
        integer_operation(op, [16, 16])
    a, b = np.array([16, 16]), np.array([3, 16])
    with pytest.raises(ValueError, match="disjoint-or"):
        integer(op, [a, b], np.array([True, True]))
    assert integer(op, [a, b], np.array([True, False]))[0] == 19


def test_vectorized_minimum_and_cast_flags_match_exact_scalar_semantics():
    from xdsl.dialects.builtin import IntegerAttr, UnitAttr

    from merlin.perf.constant_control_cfg import integer_operation
    from merlin.verify.input_copy_integer import integer

    text = """module { llvm.func @test(%a: i32, %b: i32) {
      %minimum = "llvm.intr.umin"(%a, %b) : (i32, i32) -> i32
      %narrow = llvm.trunc %a : i32 to i8
      %wide = llvm.zext %a : i32 to i64
      llvm.return
    }}"""
    function = _functions(_parse_module(text))["test"]
    minimum, truncation, extension = [op for op in function.walk() if op.results]
    a, b = np.array([-1, 0, 127, 256]), np.array([1, 3, 128, 512])
    actual = integer(minimum, [a, b], np.ones(4, dtype=bool))
    assert actual.tolist() == [integer_operation(minimum, [int(x), int(y)]) for x, y in zip(a, b, strict=True)]
    truncation.properties["overflowFlags"] = IntegerAttr(3, 32)
    with pytest.raises(ValueError, match="poison flags"):
        integer(truncation, [a], np.ones(4, dtype=bool))
    assert integer(truncation, [a], np.array([False, True, True, False])).tolist()[1:3] == [0, 127]
    extension.properties["nonNeg"] = UnitAttr()
    with pytest.raises(ValueError, match="nonnegative poison"):
        integer(extension, [a], np.ones(4, dtype=bool))
    assert integer(extension, [a], np.array([False, True, True, True])).tolist()[1:] == [0, 127, 256]
