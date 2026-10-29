"""Reject excessive proof expansion before constructing any huge lane range."""

import builtins

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_unroll_inline, compiler_unroll_tail
from merlin.verify.loop_transform_intake import Unavailable


def _bounded_range(*arguments):
    if any(value > 100000 for value in arguments):
        pytest.fail("unbounded lane range reached before resource refusal")
    return builtins.range(*arguments)


def _identity_source(trips):
    return f"""builtin.module {{
      func.func @forward(%x:i32) -> i32 {{
        %lb = arith.constant 0 : index
        %ub = arith.constant {trips} : index
        %st = arith.constant 1 : index
        %r = scf.for %i = %lb to %ub step %st iter_args(%c = %x) -> (i32) {{
          scf.yield %c : i32
        }}
        func.return %r : i32
      }}
    }}"""


def test_huge_inline_identity_refuses_before_any_lane_expansion(monkeypatch):
    from merlin.verify import compiler_loop_resources, lean_graph_pipeline_inline_emit

    factor = (1 << 62) - 8
    before = _identity_source(factor)
    after = _identity_source(0).replace("func.return %r : i32", "func.return %x : i32")
    module = parse_ir(after)
    from xdsl.dialects import scf
    from xdsl.rewriter import Rewriter

    loop = next(op for op in module.walk() if type(op) is scf.ForOp)
    Rewriter.replace_op(loop, [], list(loop.iter_args))
    monkeypatch.setattr(lean_graph_pipeline_inline_emit, "range", _bounded_range, raising=False)
    monkeypatch.setattr(compiler_loop_resources, "range", _bounded_range, raising=False)
    with pytest.raises(Unavailable, match="lane expansion work"):
        compiler_unroll_inline.prepare_query(before, ir_text(module), factor=factor)


def test_huge_bulk_identity_refuses_before_statement_lane_list(monkeypatch):
    from merlin.verify import compiler_loop_resources, lean_graph_state_pipeline_emit

    monkeypatch.setattr(lean_graph_state_pipeline_emit, "range", _bounded_range, raising=False)
    monkeypatch.setattr(compiler_loop_resources, "range", _bounded_range, raising=False)
    identity = {"body": [{"ret": [0]}]}
    with pytest.raises(Unavailable, match="lane expansion work"):
        lean_graph_state_pipeline_emit.body_group_text(
            {"index_operations": []},
            {2: identity, 3: identity},
            {2: [], 3: []},
            1,
            1 << 60,
            1,
            "",
            max_calls=None,
        )


def test_carried_expression_growth_is_charged_before_symbolic_strings():
    from merlin.verify.compiler_loop_resources import check_lanes

    doubling = [{"pure": {"kind": "add", "width": 32}, "arguments": [1, 1]}, {"ret": [0]}]
    with pytest.raises(Unavailable, match="expression resource"):
        check_lanes({"index_operations": []}, doubling, repetitions=1000, environment=[], carry=["c0"])


def test_empty_mode_still_charges_suffix_case_expansion():
    from merlin.verify.compiler_loop_resources import check_lanes

    suffix = [{"effect": 0, "arguments": [], "outputs": []}] * 256 + [{"ret": []}]
    with pytest.raises(Unavailable, match="case expansion"):
        check_lanes(
            {"index_operations": []}, [{"ret": []}], repetitions=0, environment=[], carry=[], continuation=suffix
        )


def test_huge_factor_with_one_actual_lane_remains_supported():
    factor = (1 << 63) - 8
    module = parse_ir(_identity_source(1))
    before = ir_text(module)
    unroll_ordered_loops(module, (factor,), index_bits=64)
    prepared = compiler_unroll_inline.prepare_query(before, ir_text(module), factor=factor)
    assert prepared["capture"]["changed"][0]["count"] == 1
    assert prepared["capture"]["factor"] == factor


def test_factor_seven_with_nontrivial_carried_body_remains_supported():
    from test_compiler_unroll_modes import pair

    prepared = compiler_unroll_tail.prepare_query(*pair(8, 7, carries=3, pure=True), factor=7)
    assert prepared["capture"]["changed"][0]["count"] == 8
    assert prepared["capture"]["changed"][0]["carries"] == 3
