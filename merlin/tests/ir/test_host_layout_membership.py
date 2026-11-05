"""Actual layout construction must match its checked work and growth contract."""

import hashlib

import pytest
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Region

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import host_layout_realization
from merlin.llvmlower.graph_optimizations import (
    GraphOptimizationContract,
    _check_layout_output,
    _layout_bodies,
    optimize_host_graph,
)

SOURCE = """module {
func.func @kernel(%x:tensor<2x3xi32>)->tensor<2x3xi32> {
%copy_init=tensor.empty():tensor<2x3xi32>
%copy=linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>],
 iterator_types=["parallel","parallel"]} ins(%x:tensor<2x3xi32>) outs(%copy_init:tensor<2x3xi32>) {
 ^bb0(%value:i32,%unused:i32):
 linalg.yield %value:i32
}->tensor<2x3xi32>
%init=tensor.empty():tensor<3x2xi32>
%t=linalg.transpose ins(%copy:tensor<2x3xi32>) outs(%init:tensor<3x2xi32>) permutation=[1,0]
%out=tensor.empty():tensor<3x2xi32>
%a=linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>],
 iterator_types=["parallel","parallel"]} ins(%t:tensor<3x2xi32>) outs(%out:tensor<3x2xi32>) {
 ^bb0(%v:i32,%old:i32):
 %one=arith.constant 1:i32
 %sum=arith.addi %v,%one:i32
 linalg.yield %sum:i32
}->tensor<3x2xi32>
%back_init=tensor.empty():tensor<2x3xi32>
%back=linalg.transpose ins(%a:tensor<3x2xi32>) outs(%back_init:tensor<2x3xi32>) permutation=[1,0]
func.return %back:tensor<2x3xi32>
}}
"""


def _contract(**choices):
    return GraphOptimizationContract(
        hashlib.sha256(SOURCE.encode()).hexdigest(),
        "kernel",
        propagate_layout=True,
        physical_layout=True,
        independent_element_order_unobserved=True,
        **choices,
    )


def test_unreported_generic_copy_cannot_inherit_layout_realization(monkeypatch):
    original = parse_mlir_text(SOURCE)
    template = next(op for op in original.walk() if type(op) is GenericOp)
    real_rewrite = host_layout_realization.rewrite_module
    observed = []

    def corrupt(module, **kwargs):
        result = real_rewrite(module, **kwargs)
        returned = next(op for op in module.walk() if op.name == "func.return")
        value = returned.operands[0]
        empty = EmptyOp([], value.type)
        extra = template.clone()
        extra.operands = (value, empty.results[0])
        returned.parent.insert_op_before(empty, returned)
        returned.parent.insert_op_before(extra, returned)
        returned.operands = (extra.results[0],)
        # Identity coordinates, no scalar arithmetic, and an already admitted
        # scalar-body schema make the older semantic/work checks pass. This
        # still introduces a full tensor materialization outside the plan.
        observed.append(_check_layout_output(original, module, _layout_bodies(original)))
        return result

    monkeypatch.setattr(host_layout_realization, "rewrite_module", corrupt)
    with pytest.raises(ValueError, match="operation membership.*unreported work"):
        optimize_host_graph(SOURCE, _contract())
    assert len(observed) == 1
    assert observed[0]["before_scalar_evaluations"] == observed[0]["after_scalar_evaluations"] == 12


def test_growth_limit_is_checked_before_scalar_region_clone(monkeypatch):
    clone = Region.clone
    cloned = []

    def observe(region):
        if type(region.parent) is GenericOp:
            cloned.append(region)
        return clone(region)

    monkeypatch.setattr(Region, "clone", observe)
    optimize_host_graph(SOURCE, _contract())
    assert cloned  # Prove this interception reaches the actual scalar emitter.
    cloned.clear()

    def forbid(region):
        if type(region.parent) is GenericOp:
            cloned.append(region)
            pytest.fail("layout scalar body was cloned before checking its construction budget")
        return clone(region)

    monkeypatch.setattr(Region, "clone", forbid)
    with pytest.raises(ValueError, match="compilation-space cost contract"):
        optimize_host_graph(SOURCE, _contract(maximum_added_operations=1))
    assert not cloned
