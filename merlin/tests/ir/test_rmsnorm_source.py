"""Private RMS intermediates are checked without absorbing residual users."""

from dataclasses import replace
from math import prod
from pathlib import Path
import struct

import pytest

from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.rmsnorm_source import recognize_rmsnorm


def source_graph(shape=(1, 2048), *, reciprocal_mean=False, residual=True):
    rank = len(shape)
    full, rows, column, gamma_type = [TensorType(s, "f32", "raw-bit-pattern")
        for s in (shape, shape[:-1], (*shape[:-1], 1), (shape[-1],))]
    identity = pointwise_identity_maps(rank, 1)[0]
    prefix = IndexMap(rank, identity.coefficients[:-1], (0,)*(rank-1))
    scalar = IndexMap(rank, (), ())
    broadcast = IndexMap(rank, (*identity.coefficients[:-1], (0,)*rank), (0,)*rank)
    gamma_map = IndexMap(rank, (identity.coefficients[-1],), (0,))
    constant_type = TensorType((), "f32", "raw-bit-pattern")
    attrs = lambda kind: (("numeric_semantics", f"ieee754-f32-{kind}-rne-v1"),)
    mean_kind = "mul" if reciprocal_mean else "div"
    nodes = [
        SemanticNode("H", "input", (), full, effect="input"),
        SemanticNode("gamma", "input", (), gamma_type, effect="input"),
        SemanticNode("seed", "constant", (), rows, effect="constant"),
        SemanticNode("factor", "constant", (), constant_type, effect="constant"),
        SemanticNode("eps", "constant", (), constant_type, effect="constant"),
        SemanticNode("square", "f32_mul", ("H", "H"), full, attrs("mul"), index_maps=pointwise_identity_maps(rank,2)),
        SemanticNode("sum", "source_f32_reduce_add_unqualified", ("square", "seed"), rows,
            (("axis", rank-1), ("reduction_order", "unresolved"),
             ("numeric_semantics", "source-f32-add-unqualified-v1"), ("scalar_add_order", "input-accumulator")),
            index_maps=(identity,prefix,prefix)),
        SemanticNode("sum_view", "reshape", ("sum",), column),
        SemanticNode("mean", "f32_"+mean_kind, ("sum_view", "factor"), column, attrs(mean_kind),
            index_maps=(identity,scalar,identity)),
        SemanticNode("shifted", "f32_add", ("mean", "eps"), column, attrs("add"), index_maps=(identity,scalar,identity)),
        SemanticNode("inverse", "source_f32_rsqrt_unqualified", ("shifted",), column,
            (("numeric_semantics", "source-f32-rsqrt-unqualified-v1"),), index_maps=pointwise_identity_maps(rank,1)),
        SemanticNode("normalized", "f32_mul", ("H", "inverse"), full, attrs("mul"), index_maps=(identity,broadcast,identity)),
        SemanticNode("result", "f32_mul", ("gamma", "normalized"), full, attrs("mul"), index_maps=(gamma_map,identity,identity)),
    ]
    if residual:
        nodes.append(SemanticNode("residual", "f32_add", ("H", "H"), full, attrs("add"), index_maps=pointwise_identity_maps(rank,2)))
    return KernelRequest(tuple(nodes), ("result",), ("external",), (("H","external"),("gamma","external")),
        "synthetic", "diagnostic", constants=(
            ConstantBinding("seed","f32-le-raw","00000000"*prod(shape[:-1])),
            ConstantBinding("factor","f32-le-raw",struct.pack("<f",1/shape[-1] if reciprocal_mean else shape[-1]).hex()),
            ConstantBinding("eps","f32-le-raw","acc52737")))


@pytest.mark.parametrize("shape", [(1,2048),(128,2048),(2,3,1024)])
@pytest.mark.parametrize("reciprocal", [False,True])
def test_source_mean_gamma_broadcast_and_preserved_H(shape, reciprocal):
    source=source_graph(shape, reciprocal_mean=reciprocal)
    region=recognize_rmsnorm(source,"result")
    assert region.matrix_shape==(prod(shape[:-1]),shape[-1])
    assert region.hidden=="H" and region.gamma=="gamma"
    assert region.hidden_consumers==("residual",)
    assert set(region.computations)=={"square","sum","sum_view","mean","shifted","inverse","normalized","result"}
    assert region.record()["epsilon_bits"]=="0x3727c5ac"


@pytest.mark.parametrize("bits", [0x3a83126f, 0x358637bd, 0x3f000000, 0x3f800000, 1, 0x7f7fffff])
def test_finite_positive_source_epsilon_is_preserved_without_target_policy(bits):
    source = source_graph()
    source = replace(source, constants=tuple(
        replace(c, data_hex=bits.to_bytes(4, "little").hex()) if c.node_id == "eps" else c
        for c in source.constants))
    assert recognize_rmsnorm(source, "result").epsilon_bits == f"0x{bits:08x}"
    assert recognize_rmsnorm(source, "result", epsilon_bits=f"0x{bits:08x}").epsilon_bits == f"0x{bits:08x}"
    if bits != 0x3727c5ac:
        with pytest.raises(ValueError, match="source epsilon differs"):
            recognize_rmsnorm(source, "result", epsilon_bits="0x3727c5ac")


@pytest.mark.parametrize("bits", [0, 0x80000000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc00000])
def test_nonpositive_and_nonfinite_source_epsilon_refuse(bits):
    source = source_graph()
    source = replace(source, constants=tuple(
        replace(c, data_hex=bits.to_bytes(4, "little").hex()) if c.node_id == "eps" else c
        for c in source.constants))
    with pytest.raises(ValueError, match="epsilon must be finite and positive"):
        recognize_rmsnorm(source, "result")


@pytest.mark.parametrize("fault", ["factor","seed","square","reduction","gamma_map","inverse_map","escape","output"])
def test_changed_literals_axes_and_escaping_intermediates_refuse(fault):
    source=source_graph()
    if fault in {"factor","seed"}:
        key=fault
        source=replace(source, constants=tuple(replace(c,data_hex="0000803f"*(len(c.data_hex)//8)) if c.node_id==key else c for c in source.constants))
    elif fault=="escape":
        source=replace(source,nodes=(*source.nodes,SemanticNode("escape","copy",("inverse",),source.node("inverse").type)))
    elif fault=="output":
        source=replace(source,outputs=("result","square"),output_storages=("external","external"))
    else:
        key={"square":"square","reduction":"sum","gamma_map":"result","inverse_map":"normalized"}[fault]
        node=source.node(key)
        if fault=="square":
            node=replace(node,inputs=("H","normalized"))
            # Avoid a non-topological synthetic graph; changing one input to a
            # distinct, otherwise identical boundary is enough to break H*H.
            extra=replace(source.node("H"),id="other")
            source=replace(source,nodes=(extra,*source.nodes),input_storages=(("other","external"),*source.input_storages))
            node=replace(node,inputs=("H","other"))
        elif fault=="reduction":
            node=replace(node,attrs=tuple((k,"axis-ascending") if k=="reduction_order" else(k,v)for k,v in node.attrs))
        else:
            maps=list(node.index_maps)
            index=0 if fault=="gamma_map" else 1
            maps[index]=replace(maps[index],offsets=tuple(1 for _ in maps[index].offsets))
            node=replace(node,index_maps=tuple(maps))
        source=replace(source,nodes=tuple(node if n.id==key else n for n in source.nodes))
    with pytest.raises(ValueError):
        recognize_rmsnorm(source,"result")


@pytest.mark.parametrize("case", ["decode1-input","decode1-post","prefill0-input","prefill0-post"])
def test_genuine_standalone_norm_linalg_source(case):
    from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
    path=Path(__file__).resolve().parents[4]/"out"/"norm-arithmetic-source-v2"/case/"model.mlir"
    if not path.is_file():
        pytest.skip("optional genuine standalone normalization source fixture")
    request=translate_linalg_text(path.read_text(),entry="forward",target_identity="synthetic",lowering_policy="diagnostic").request
    region=recognize_rmsnorm(request,request.outputs[0])
    assert region.matrix_shape==((128 if case.startswith("prefill") else 1),2048)
    assert len(region.computations)==8 and len(region.initializers)==3
    assert region.hidden not in region.computations and region.gamma not in region.computations
