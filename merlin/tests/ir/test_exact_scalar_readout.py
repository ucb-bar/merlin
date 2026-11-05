"""Consume exact readout IR at every transition, not just its proof receipt."""
import numpy as np
import pytest

from merlin.compile.scheduling.dense_contracts import DenseIntegerReadout
from merlin.compile.scheduling.fusion_semantics import scalar_graph,ScalarBudgetExceeded
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.integer_readout import emit_exact_scalar_ir,emit_rational_scalar_ir,partition_column_scales
from merlin.runtime.linalg_numpy import evaluate as evaluate_ir


def compare_consumed_ir(ir,proof,relu):
    graph=DenseIntegerReadout(ir,"i8","source_readout",()).graph()
    assert graph.input_bits==(32,)and graph.output_bits==8
    source=proof["source"]
    points={-(1<<31),(1<<31)-1,0,-1,1}
    for threshold in source["thresholds"]:
        points.update(x for x in (threshold-1,threshold,threshold+1)if -(1<<31)<=x<(1<<31))
    points.update(np.random.default_rng(917).integers(-(1<<31),1<<31,128).tolist())
    inputs=np.array(sorted(points),dtype=np.int32)
    actual,=evaluate_ir(parse_mlir_text(ir),[inputs])
    # Independent ordered binary32 arithmetic. It neither interprets the proof
    # thresholds nor reproduces the emitted integer comparison/select chain.
    expected=inputs.astype(np.float32)
    with np.errstate(over="ignore"):
        for scale in source["source_scales"]:expected=(expected*np.float32(scale)).astype(np.float32)
    expected=np.clip(np.rint(expected),0 if relu else -128,127).astype(np.int8)
    # A constant unary function has one scalar interpreter result; each input
    # invocation returns that same scalar.
    assert np.array_equal(np.broadcast_to(actual,inputs.shape),expected)
    return graph


@pytest.mark.parametrize("scales,relu",[
    ([1/48],False),([1/48],True),([.1,.3],False),
    ([2.**-80],False),([2.**40],False),([2.**-149],True),
])
def test_fallback_consumes_all_transitions_including_folded_and_coalesced(scales,relu):
    ir,proof=emit_exact_scalar_ir(scales,relu=relu)
    assert proof["schema"]=="exact_threshold_readout_v1"
    graph=compare_consumed_ir(ir,proof,relu)
    assert len(graph.operations)==proof["scalar_operations"]==1+4*proof["transition_count"]
    assert len(graph.operations)<=1021 and len(ir.encode())<=65536


@pytest.mark.parametrize("scales,relu",[([.5],False),([.25,.5],True),([1/7],False)])
def test_compact_proven_path_remains_byte_identical(scales,relu):
    assert emit_exact_scalar_ir(scales,relu=relu)==emit_rational_scalar_ir(scales,relu=relu)


def test_threshold_graph_limits_and_effect_restrictions_remain_explicit():
    ir,proof=emit_exact_scalar_ir([1/48])
    assert proof["scalar_operations"]==1021
    with pytest.raises(ScalarBudgetExceeded):scalar_graph(ir.encode(),1020,65536)
    with pytest.raises(ScalarBudgetExceeded):DenseIntegerReadout("//"+"x"*65536+"\n"+ir,"i8","too_large",()).graph()
    # A larger operation budget does not grant memory or looping operations.
    bad=ir.replace(" %initial=", " %unknown=func.call @readout(%x):(i32)->i8\n %initial=")
    with pytest.raises(ValueError):DenseIntegerReadout(bad,"i8","effects",()).graph()


def test_column_partition_accepts_exact_fallback_without_changing_cover():
    result=partition_column_scales([.01,.01,.5,.01],4)
    assert [(r["first_column"],r["columns"])for r in result["regions"]]==[(0,2),(2,1),(3,1)]
    for region in result["regions"]:compare_consumed_ir(region["scalar_ir"],region["readout_proof"],False)


@pytest.mark.parametrize("scales,relu",[([0],False),([-1],False),([float("nan")],False),([float("inf")],False),([.5],1)])
def test_nonmonotone_or_ambiguous_source_refuses(scales,relu):
    with pytest.raises((ValueError,OverflowError)):emit_exact_scalar_ir(scales,relu=relu)
