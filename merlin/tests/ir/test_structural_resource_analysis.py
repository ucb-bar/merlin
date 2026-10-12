"""Actual IR analysis and allocation/emission mutations, independent of solver formulae."""

from __future__ import annotations

import builtins
from dataclasses import replace

import pytest
from xdsl.dialects import builtin, func, scf
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, traits_def
from xdsl.traits import MemoryWriteEffect

from merlin.semantic_compiler.allocate import StorageBank, check_assignment, live_ranges
from merlin.semantic_compiler.allocation_types import CandidateGraph, Reservation, Value
from merlin.semantic_compiler.rules import AddressConstraint
from merlin.transforms.analyses import IRIndex, analyze_ir, may_alias, validate_analysis
from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.resource_allocation import (
    ATTRIBUTE,
    InstructionConstraints,
    ResourceBinding,
    allocate_instruction_resources,
    emit_resource_allocation,
    validate_allocated_ir,
)


def _scalar_module():
    return parse_ir("""builtin.module {
      func.func @forward(%x: i8, %y: i8) -> (i8, i8) {
        %a = arith.addi %x, %y : i8
        %b = arith.muli %a, %y : i8
        %c = arith.subi %a, %x : i8
        return %b, %c : i8, i8
      }
    }""")


def _model(module, registers=3):
    index = IRIndex(module)
    entry = next(op for op in module.walk() if isinstance(op, func.FuncOp)).body.block
    banks = (StorageBank("inputs", "boundary", 2, "slot"), StorageBank("temporaries", "registers", registers, "slot"))
    bindings = tuple(ResourceBinding(index.id(value), "inputs", 1) for value in entry.args)
    bindings += tuple(ResourceBinding(index.id(value), "temporaries", 1) for op in entry.ops for value in op.results)
    return {"entry": "forward", "bindings": bindings, "banks": banks}


def _replace_row(emitted, ordinal, **changes):
    metadata = dict(emitted.attributes[ATTRIBUTE].data)
    rows = list(metadata["values"])
    row = dict(rows[ordinal].data)
    row.update(changes)
    rows[ordinal] = builtin.DictionaryAttr(row)
    metadata["values"] = builtin.ArrayAttr(rows)
    emitted.attributes[ATTRIBUTE] = builtin.DictionaryAttr(metadata)


def test_analysis_liveness_has_independent_expected_endpoints_and_pressure():
    module = _scalar_module()
    analysis = analyze_ir(module)
    index = IRIndex(module)
    entry = next(op for op in module.walk() if isinstance(op, func.FuncOp)).body.block
    block = next(row for row in analysis.blocks if row.block == index.block_ids[entry])
    expected = {
        index.id(entry.args[0]): (-1, 2),
        index.id(entry.args[1]): (-1, 1),
        index.id(entry.ops.first.results[0]): (0, 2),
    }
    actual = {interval.value: (interval.start, interval.end) for interval in block.live_intervals}
    for value, bounds in expected.items():
        assert actual[value] == bounds
    assert block.peak_live_values == 4
    assert block.peak_known_logical_bytes == 4
    assert block.working_set_known_logical_bytes == 0
    assert not block.unknown_sizes
    assert analysis.blocks[0].peak_live_values == 0  # symbol declarations do not execute their bodies
    validate_analysis(module, analysis)
    with pytest.raises(ValueError, match="actual current IR"):
        validate_analysis(module, replace(analysis, source_sha256="wrong"))


def test_alias_views_external_arguments_and_fresh_allocations_are_distinguished():
    module = parse_ir("""builtin.module {
      func.func @forward(%x: memref<4xi8>, %y: memref<4xi8>) {
        %a = memref.alloc() : memref<4xi8>
        %b = memref.alloc() : memref<4xi8>
        %v = "memref.cast"(%a) : (memref<4xi8>) -> memref<?xi8>
        return
      }
    }""")
    analysis, index = analyze_ir(module), IRIndex(module)
    entry = next(op for op in module.walk() if isinstance(op, func.FuncOp)).body.block
    values = [
        analysis.value(index.id(value)) for value in (*entry.args, *(op.results[0] for op in entry.ops if op.results))
    ]
    x, y, a, b, view = values
    assert may_alias(x, y)
    assert may_alias(a, view)
    assert not may_alias(a, b)
    assert not may_alias(a, x)
    assert view.alias_roots == a.alias_roots
    assert view.logical_bytes is None
    block = next(row for row in analysis.blocks if row.block == index.block_ids[entry])
    assert block.working_set_known_logical_bytes == 16


def test_unknown_call_extends_buffer_lifetime_and_effect_dependencies():
    module = parse_ir("""builtin.module {
      func.func private @escape(memref<4xi8>)
      func.func @forward() {
        %a = memref.alloc() : memref<4xi8>
        %b = memref.alloc() : memref<4xi8>
        %i = arith.constant 0 : index
        %v = arith.constant 7 : i8
        memref.store %v, %a[%i] : memref<4xi8>
        memref.store %v, %b[%i] : memref<4xi8>
        call @escape(%a) : (memref<4xi8>) -> ()
        memref.store %v, %b[%i] : memref<4xi8>
        return
      }
    }""")
    analysis, index = analyze_ir(module), IRIndex(module)
    entry = next(op for op in module.walk() if isinstance(op, func.FuncOp) and op.sym_name.data == "forward").body.block
    block = next(row for row in analysis.blocks if row.block == index.block_ids[entry])
    a = analysis.value(index.id(entry.ops.first.results[0]))
    assert a.escapes
    interval = next(row for row in block.live_intervals if row.value == a.id)
    assert interval.end == len(tuple(entry.ops))
    assert (4, 5) not in block.dependencies  # stores to independently allocated buffers
    assert (4, 6) in block.dependencies and (5, 6) in block.dependencies
    assert (6, 7) in block.dependencies
    assert index.operation_ids[tuple(entry.ops)[6]] in block.unknown_effects
    assert block.unknown_sizes  # index representation was not supplied
    known_width = analyze_ir(module, index_bits=32)
    known_block = next(row for row in known_width.blocks if row.block == index.block_ids[entry])
    assert not known_block.unknown_sizes


def test_loop_carried_buffer_can_alias_its_enclosing_fresh_allocation():
    module = parse_ir("""builtin.module {
      func.func @forward() {
        %a = memref.alloc() : memref<4xi8>
        %lo = arith.constant 0 : index
        %hi = arith.constant 2 : index
        %step = arith.constant 1 : index
        %v = arith.constant 7 : i8
        memref.store %v, %a[%lo] : memref<4xi8>
        %result = scf.for %iv = %lo to %hi step %step iter_args(%carried = %a) -> (memref<4xi8>) {
          %loaded = memref.load %carried[%iv] : memref<4xi8>
          memref.store %loaded, %carried[%iv] : memref<4xi8>
          scf.yield %carried : memref<4xi8>
        }
        memref.dealloc %a : memref<4xi8>
        return
      }
    }""")
    analysis, index = analyze_ir(module), IRIndex(module)
    entry = next(op for op in module.walk() if isinstance(op, func.FuncOp)).body.block
    loop = next(op for op in module.walk() if isinstance(op, scf.ForOp))
    allocated = analysis.value(index.id(entry.ops.first.results[0]))
    carried = analysis.value(index.id(loop.body.block.args[1]))
    assert allocated.alias_kind == "fresh"
    assert carried.alias_kind == "unknown"
    assert may_alias(allocated, carried)
    assert may_alias(allocated, analysis.value(index.id(loop.results[0])))
    block = next(row for row in analysis.blocks if row.block == index.block_ids[entry])
    assert (5, 6) in block.dependencies  # enclosing store must precede carried-buffer read


def test_allocator_reuses_existing_solver_and_actual_annotation_checker():
    module = _scalar_module()
    original = ir_text(module)
    model = _model(module)
    allocation = allocate_instruction_resources(module, **model)
    assert allocation.status == "feasible"
    emitted = emit_resource_allocation(module, allocation, **model)
    assert ir_text(module) == original
    validate_allocated_ir(module, parse_ir(ir_text(emitted)), **model)
    assert emitted.attributes[ATTRIBUTE].data["source_sha256"].data == allocation.source_sha256
    assert allocation.logical_outputs == tuple(binding.value for binding in model["bindings"][-2:])
    assert allocate_instruction_resources(module, **_model(module, registers=2)).status == "infeasible_candidate"


@pytest.mark.parametrize("fault", ["overlap", "capacity", "lifetime", "bank", "source", "geometry"])
def test_actual_emitted_assignment_faults_are_rejected(fault):
    module = _scalar_module()
    model = _model(module)
    allocation = allocate_instruction_resources(module, **model)
    emitted = emit_resource_allocation(module, allocation, **model)
    rows = tuple(emitted.attributes[ATTRIBUTE].data["values"])
    if fault == "overlap":
        _replace_row(emitted, 4, address=rows[3].data["address"])
    elif fault == "capacity":
        _replace_row(emitted, 4, address=builtin.IntegerAttr(100, builtin.i32))
    elif fault == "lifetime":
        _replace_row(emitted, 4, live_end=builtin.IntegerAttr(0, builtin.i32))
    elif fault == "bank":
        _replace_row(emitted, 4, storage=builtin.StringAttr("inputs"))
    elif fault == "source":
        addition = next(op for op in emitted.walk() if op.name == "arith.addi")
        addition.operands = [addition.lhs, addition.lhs]
    else:
        model = _model(module, registers=4)
    with pytest.raises(ValueError):
        validate_allocated_ir(module, emitted, **model)


def test_unknown_effects_aliases_and_missing_binding_do_not_become_feasible():
    unknown = parse_ir("""builtin.module {
      func.func @forward(%x: i8) -> i8 {
        "foreign.opaque"(%x) : (i8) -> ()
        return %x : i8
      }
    }""")
    model = _model(unknown)
    result = allocate_instruction_resources(unknown, **model)
    assert result.status == "unknown" and "unknown effects" in result.reason
    assert not result.addresses
    pointer = parse_ir("builtin.module { func.func @forward(%x: memref<4xi8>) { return } }")
    assert allocate_instruction_resources(pointer, **_model(pointer)).status == "unknown"
    scalar = _scalar_module()
    incomplete = _model(scalar)
    incomplete["bindings"] = incomplete["bindings"][:-1]
    assert allocate_instruction_resources(scalar, **incomplete).status == "unknown"


@irdl_op_definition
class _StateWrite(IRDLOperation):
    name = "public_probe.state_write"
    value = operand_def(builtin.i8)
    traits = traits_def(MemoryWriteEffect())


def test_void_state_instruction_is_scheduled_and_pins_its_operand():
    module = _scalar_module()
    entry = next(op for op in module.walk() if isinstance(op, func.FuncOp)).body.block
    ops = tuple(entry.ops)
    state = _StateWrite.build(operands=[ops[0].results[0]])
    entry.insert_op_before(state, ops[-1])
    model = _model(module, registers=4)
    unknown_completion = allocate_instruction_resources(module, **model)
    assert unknown_completion.status == "unknown" and "completion" in unknown_completion.reason
    index = IRIndex(module)
    model["instruction_constraints"] = (InstructionConstraints(index.operation_ids[state], completion_offset=1),)
    allocation = allocate_instruction_resources(module, **model)
    assert allocation.status == "feasible"
    assert index.id(ops[0].results[0]) in allocation.retained_effect_operands
    assert len(allocation.order) == 4
    emit_resource_allocation(module, allocation, **model).verify()


def test_void_instruction_address_constraints_are_checked_in_actual_emission():
    module = _scalar_module()
    entry = next(op for op in module.walk() if isinstance(op, func.FuncOp)).body.block
    ops = tuple(entry.ops)
    state = _StateWrite.build(operands=[ops[0].results[0]])
    entry.insert_op_before(state, ops[-1])
    index = IRIndex(module)
    model = _model(module, registers=4)
    model["reservations"] = (Reservation("temporaries", 0, 1),)
    model["instruction_constraints"] = (
        InstructionConstraints(
            index.operation_ids[state], validity=(AddressConstraint("aligned", "in0", value=2),), completion_offset=1
        ),
    )
    allocation = allocate_instruction_resources(module, **model)
    assert allocation.status == "feasible"
    emitted = emit_resource_allocation(module, allocation, **model)
    _replace_row(emitted, 2, address=builtin.IntegerAttr(1, builtin.i32))
    with pytest.raises(ValueError, match="alignment"):
        validate_allocated_ir(module, emitted, **model)
    model["instruction_constraints"] = (
        InstructionConstraints(
            index.operation_ids[state], validity=(AddressConstraint("aligned", "out", value=2),), completion_offset=1
        ),
    )
    assert allocate_instruction_resources(module, **model).status == "unknown"


def test_index_representation_requires_reviewed_width_and_is_bound_to_assignment():
    module = parse_ir("""builtin.module {
      func.func @forward(%x: index) -> index {
        %one = arith.constant 1 : index
        %result = arith.addi %x, %one : index
        return %result : index
      }
    }""")
    model = _model(module)
    assert allocate_instruction_resources(module, **model).status == "unknown"
    model["index_bits"] = 32
    allocation = allocate_instruction_resources(module, **model)
    assert allocation.status == "feasible"
    emitted = emit_resource_allocation(module, allocation, **model)
    model["index_bits"] = 64
    with pytest.raises(ValueError, match="constraints"):
        validate_allocated_ir(module, emitted, **model)


def test_event_nodes_consume_no_storage_but_extend_real_operand_lifetime():
    graph = CandidateGraph(
        (Value(0, "input", "r", 1, (), "x", "input"), Value(1, "state", "r", 0, (0,), None, "event")), ()
    )
    banks = (StorageBank("r", "physical", 1, "slot"),)
    assert live_ranges(graph, (1,))[0] == (0, 1)
    assert check_assignment(graph, (1,), {0: 0, 1: 0}, banks) == (True, "")
    with pytest.raises(ValueError, match="no storage"):
        Value(1, "state", "r", 1, (0,), None, "event")


def test_reservations_and_aliased_storage_are_checked_in_actual_assignments():
    module = _scalar_module()
    model = _model(module, registers=4)
    model["reservations"] = (Reservation("temporaries", 0, 1),)
    allocation = allocate_instruction_resources(module, **model)
    assert allocation.status == "feasible"
    emitted = emit_resource_allocation(module, allocation, **model)
    _replace_row(emitted, 4, address=builtin.IntegerAttr(0, builtin.i32))
    with pytest.raises(ValueError, match="reserved"):
        validate_allocated_ir(module, emitted, **model)
    model = _model(module)
    model["banks"] = (StorageBank("inputs", "physical", 2, "slot"), StorageBank("temporaries", "physical", 3, "slot"))
    assert allocate_instruction_resources(module, **model).status == "infeasible_candidate"


def test_analysis_does_not_import_a_solver(monkeypatch):
    original = builtins.__import__

    def denied(name, *args, **kwargs):
        if name == "z3" or name.startswith("z3."):
            raise AssertionError("analysis unexpectedly imported a placement solver")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", denied)
    assert analyze_ir(_scalar_module()).values


def test_nested_loop_captures_extend_outer_liveness_without_global_symbol_uses():
    module = parse_ir("""builtin.module {
      func.func @forward(%x: memref<4xi8>) {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %four = arith.constant 4 : index
        scf.for %i = %zero to %four step %one {
          %v = memref.load %x[%i] : memref<4xi8>
          memref.store %v, %x[%i] : memref<4xi8>
        }
        return
      }
    }""")
    analysis, index = analyze_ir(module, index_bits=32), IRIndex(module)
    function = next(op for op in module.walk() if isinstance(op, func.FuncOp))
    entry = function.body.block
    block = next(row for row in analysis.blocks if row.block == index.block_ids[entry])
    identity = index.id(entry.args[0])
    assert next(row for row in block.live_intervals if row.value == identity).end == 3
    assert not analysis.blocks[0].live_intervals
    inner = next(row for row in analysis.blocks if row.block not in (0, block.block))
    assert any(row.value == identity and row.start == -1 for row in inner.live_intervals)


def test_fixed_boundaries_bind_actual_emitted_output_order():
    module = _scalar_module()
    model = _model(module)
    model["fixed_inputs"] = {binding.value: i for i, binding in enumerate(model["bindings"][:2])}
    model["fixed_outputs"] = (2, 1)
    allocation = allocate_instruction_resources(module, **model)
    assert allocation.status == "feasible"
    emitted = emit_resource_allocation(module, allocation, **model)
    _replace_row(emitted, 3, address=builtin.IntegerAttr(0, builtin.i32))
    with pytest.raises(ValueError, match="fixed external"):
        validate_allocated_ir(module, emitted, **model)


def test_huge_static_working_set_does_not_enumerate_elements():
    module = parse_ir("builtin.module { func.func @forward(%x: memref<100000000003xi32>) { return } }")
    analysis = analyze_ir(module)
    assert len(analysis.values) == 1
    assert analysis.values[0].logical_bytes == 400000000012
    assert max(block.working_set_known_logical_bytes for block in analysis.blocks) == 400000000012
