"""Guarded upstream transformations preserve integer programs and opaque effects."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from xdsl.dialects import arith, scf
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.scf import ScfFunctions

from merlin.transforms.certificates import ir_text, make_change, parse_ir
from merlin.transforms.pure_scalar import eliminate_scalar_redundancy, hoist_scalar_invariants
from merlin.transforms.scalar_witness import SCALAR_SCOPES, validate_scalar_change


def _execute(module, values):
    interpreter = Interpreter(module)
    for implementation in (ArithFunctions(), FuncFunctions(), ScfFunctions()):
        interpreter.register_implementations(implementation)
    return interpreter.call_op("kernel", values)


def _redundant(width=8):
    return parse_ir(f"""builtin.module {{
      func.func @kernel(%a:i{width}, %b:i{width}) -> i{width} {{
        %first = arith.addi %a, %b : i{width}
        %second = arith.addi %a, %b : i{width}
        %dead_first = arith.muli %a, %b : i{width}
        %dead_second = arith.subi %dead_first, %a : i{width}
        %result = arith.muli %first, %second : i{width}
        func.return %result : i{width}
      }}
    }}""")


def _loop(width=8, lower=0, upper=7, step=2):
    return parse_ir(f"""builtin.module {{
      func.func @kernel(%a:i{width}, %b:i{width}) -> i{width} {{
        %lower = arith.constant {lower} : index
        %upper = arith.constant {upper} : index
        %step = arith.constant {step} : index
        %result = scf.for %i = %lower to %upper step %step iter_args(%acc = %a) -> (i{width}) {{
          %first = arith.addi %a, %b : i{width}
          %second = arith.muli %first, %b : i{width}
          %value = arith.subi %acc, %second : i{width}
          scf.yield %value : i{width}
        }}
        func.return %result : i{width}
      }}
    }}""")


def _check_actual_change(original, transform, kind):
    before = ir_text(original)
    modified = original.clone()
    witnesses = transform(modified)
    change = make_change(kind, before, modified, {}, witnesses, SCALAR_SCOPES[kind])
    validate_scalar_change(change)
    return modified, witnesses, change


@pytest.mark.parametrize("width", [1, 8, 16, 32, 128])
def test_cse_dce_reuses_upstream_mechanisms_and_preserves_wrapping(width):
    original = _redundant(width)
    modified, witnesses, _ = _check_actual_change(
        original,
        eliminate_scalar_redundancy,
        "scalar_redundancy_elimination",
    )
    assert [op.name for op in modified.walk() if isinstance(op, arith.AddiOp)] == ["arith.addi"]
    assert {w.rule for w in witnesses} == {"cse", "dce"}
    assert sum(isinstance(op, arith.MuliOp) for op in modified.walk()) == 1
    edge = (1 << (width - 1)) - 1
    for values in ((0, 0), (edge, edge), (-edge - 1, 1), (-1, -1)):
        assert _execute(original, values) == _execute(modified, values)
    assert not eliminate_scalar_redundancy(modified)


@pytest.mark.parametrize("lower,upper,step", [(0, 0, 1), (3, 3, 2), (0, 7, 2), (2, 9, 3), (-3, 4, 2)])
def test_licm_total_speculation_handles_zero_trip_and_nondivisible_bounds(lower, upper, step):
    original = _loop(lower=lower, upper=upper, step=step)
    modified, witnesses, _ = _check_actual_change(
        original,
        hoist_scalar_invariants,
        "scalar_loop_invariant_motion",
    )
    assert len(witnesses) == 2 and all(w.rule == "licm" for w in witnesses)
    loop = next(op for op in modified.walk() if isinstance(op, scf.ForOp))
    assert not any(isinstance(op, arith.AddiOp | arith.MuliOp) for op in loop.body.block.ops)
    assert any(isinstance(op, arith.SubiOp) for op in loop.body.block.ops)
    for values in ((127, 127), (-128, 1), (-1, 17)):
        assert _execute(original, values) == _execute(modified, values)
    assert not hoist_scalar_invariants(modified)


def test_large_loop_licm_space_depends_on_ir_not_iterations():
    original = _loop(upper=10**12 + 3, step=7)
    modified, witnesses, _ = _check_actual_change(
        original,
        hoist_scalar_invariants,
        "scalar_loop_invariant_motion",
    )
    assert len(witnesses) == 2
    assert len(ir_text(modified)) < 2500
    assert len(tuple(modified.walk())) == len(tuple(original.walk()))


def test_dynamic_loop_licm_preserves_variants_and_unknown_effects():
    original = parse_ir("""builtin.module {
      func.func @kernel(%upper:index, %a:i8, %b:i8) -> i8 {
        %l = arith.constant 0 : index
        %s = arith.constant 1 : index
        %r = scf.for %i = %l to %upper step %s iter_args(%acc=%a) -> (i8) {
          %sum = arith.addi %a, %b : i8
          "opaque.observe"(%sum, %i) : (i8,index) -> ()
          %next = arith.addi %acc, %sum : i8
          scf.yield %next : i8
        }
        func.return %r : i8
      }
    }""")
    modified, witnesses, _ = _check_actual_change(
        original,
        hoist_scalar_invariants,
        "scalar_loop_invariant_motion",
    )
    assert len(witnesses) == 1
    loop = next(op for op in modified.walk() if isinstance(op, scf.ForOp))
    assert [op.name for op in loop.body.block.ops] == ["builtin.unregistered", "arith.addi", "scf.yield"]


def test_unknown_consumers_keep_inputs_live_and_unknown_producers_distinct():
    original = parse_ir("""builtin.module {
      func.func @kernel(%a:i8, %b:i8) -> i8 {
        %sum = arith.addi %a, %b : i8
        %duplicate = arith.addi %a, %b : i8
        "opaque.observe"(%duplicate) : (i8) -> ()
        %read1 = "opaque.read"() : () -> i8
        %read2 = "opaque.read"() : () -> i8
        "opaque.observe"(%sum, %read1, %read2) : (i8,i8,i8) -> ()
        func.return %a : i8
      }
    }""")
    modified, witnesses, _ = _check_actual_change(
        original,
        eliminate_scalar_redundancy,
        "scalar_redundancy_elimination",
    )
    opaque = [op for op in modified.walk() if op.name == "builtin.unregistered"]
    assert len(opaque) == 4
    assert len(witnesses) == 1 and witnesses[0].rule == "cse"
    assert sum(isinstance(op, arith.AddiOp) for op in modified.walk()) == 1


@pytest.mark.parametrize("metadata", ["overflow<nsw>", "overflow<nuw>", "overflow<nsw,nuw>", "{unmodeled}"])
def test_poison_and_unmodeled_metadata_are_retained(metadata):
    source = f"""builtin.module {{
      func.func @kernel(%a:i8, %b:i8) -> i8 {{
        %x = arith.addi %a, %b {metadata} : i8
        %y = arith.addi %a, %b {metadata} : i8
        func.return %a : i8
      }}
    }}"""
    module = parse_ir(source)
    before = ir_text(module)
    assert not eliminate_scalar_redundancy(module)
    assert ir_text(module) == before


def test_licm_does_not_speculate_division_float_index_or_opaque_ops():
    source = """builtin.module {
      func.func @kernel(%a:i8, %b:i8, %x:f32, %y:f32) {
        %l = arith.constant 0 : index
        %u = arith.constant 0 : index
        %s = arith.constant 1 : index
        scf.for %i = %l to %u step %s {
          %division = arith.divsi %a, %b : i8
          %float = arith.addf %x, %y : f32
          %index = arith.addi %l, %u : index
          %unknown = "opaque.compute"(%a) : (i8) -> i8
          scf.yield
        }
        func.return
      }
    }"""
    module = parse_ir(source)
    before = ir_text(module)
    assert not hoist_scalar_invariants(module)
    assert ir_text(module) == before


@pytest.mark.parametrize("fault", ["target", "witness", "scope", "parameter"])
def test_actual_ir_validator_rejects_forged_scalar_changes(fault):
    original = _redundant()
    _, _, change = _check_actual_change(original, eliminate_scalar_redundancy, "scalar_redundancy_elimination")
    if fault == "target":
        target = parse_ir(change.after_ir)
        arithmetic = next(op for op in target.walk() if isinstance(op, arith.MuliOp))
        arithmetic.operands = (arithmetic.operands[0], target.body.block.first_op.body.block.args[0])
        change = replace(
            change,
            after_ir=ir_text(target),
            certificate=replace(
                change.certificate,
                after_ir_sha256=hashlib.sha256(ir_text(target).encode()).hexdigest(),
            ),
        )
    elif fault == "witness":
        first, *rest = change.certificate.witnesses
        change = replace(
            change, certificate=replace(change.certificate, witnesses=(replace(first, representative_index=999), *rest))
        )
    elif fault == "scope":
        change = replace(change, certificate=replace(change.certificate, scope=("candidate_assertion",)))
    else:
        change = replace(change, parameters={"assume_pure": True})
    with pytest.raises(ValueError):
        validate_scalar_change(change)


def test_nested_licm_replays_each_crossed_loop_and_preserves_carried_state():
    original = parse_ir("""builtin.module {
      func.func @kernel(%a:i8, %b:i8) -> i8 {
        %l = arith.constant 0 : index
        %u = arith.constant 3 : index
        %s = arith.constant 1 : index
        %outer = scf.for %i = %l to %u step %s iter_args(%acc=%a) -> (i8) {
          %inner = scf.for %j = %l to %u step %s iter_args(%state=%acc) -> (i8) {
            %invariant = arith.muli %a, %b : i8
            %variant = arith.addi %state, %invariant : i8
            scf.yield %variant : i8
          }
          scf.yield %inner : i8
        }
        func.return %outer : i8
      }
    }""")
    modified, witnesses, _ = _check_actual_change(
        original,
        hoist_scalar_invariants,
        "scalar_loop_invariant_motion",
    )
    assert len(witnesses) == 2
    assert witnesses[0].operation_index == witnesses[1].operation_index
    assert witnesses[0].loop_index != witnesses[1].loop_index
    assert _execute(original, (127, 17)) == _execute(modified, (127, 17))


def test_no_hoisting_crosses_an_opaque_region_boundary():
    original = parse_ir("""builtin.module {
      func.func @kernel(%a:i8, %b:i8) {
        "opaque.region"() ({
          %l = arith.constant 0 : index
          %u = arith.constant 2 : index
          %s = arith.constant 1 : index
          scf.for %i = %l to %u step %s {
            %sum = arith.addi %a, %b : i8
            "opaque.observe"(%sum) : (i8) -> ()
            scf.yield
          }
          "opaque.yield"() : () -> ()
        }) : () -> ()
        func.return
      }
    }""")
    modified, witnesses, _ = _check_actual_change(
        original,
        hoist_scalar_invariants,
        "scalar_loop_invariant_motion",
    )
    assert len(witnesses) == 1
    sum_op = next(op for op in modified.walk() if isinstance(op, arith.AddiOp))
    assert sum_op.parent_op().name == "builtin.unregistered"
