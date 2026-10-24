"""Exact quantization may commute only through proved pure movement."""
import pytest


def test_readout_quantizer_fusion_retains_sub_f32_tie_information():
    torch=pytest.importorskip('torch')
    pytest.importorskip('m2m.capture.integer_readout')
    import torch.ao.quantization.fx._decomposed
    from merlin.targetgen.readout_quantizer_fusion import fuse_readout_quantizers
    graph=torch.fx.Graph();acc=graph.placeholder('acc');bias=graph.placeholder('bias')
    real=graph.call_function(torch.ops.m2m_q.exact_readout.default,(acc,.015625,[.015625],1.,bias))
    moved=graph.call_function(torch.ops.aten.reshape.default,(real,[1]))
    q=graph.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default,(moved,.015625,0,-128,127,torch.int8))
    graph.output(q);model=torch.fx.GraphModule(torch.nn.Module(),graph)
    values=torch.tensor([[32]],dtype=torch.int32);tiny=torch.tensor([2.**-40])
    assert model(values,tiny).tolist()==[0]  # The unnormalized source double rounds.
    proof=fuse_readout_quantizers(model);assert len(proof)==1
    assert model(values,tiny).tolist()==[1]
    assert model(values,-tiny).tolist()==[0]


@pytest.mark.parametrize('escape',['float_output','arithmetic','different_quantizer'])
def test_readout_fusion_refuses_unproved_live_paths(escape):
    torch=pytest.importorskip('torch')
    pytest.importorskip('m2m.capture.integer_readout')
    import torch.ao.quantization.fx._decomposed
    from merlin.targetgen.readout_quantizer_fusion import fuse_readout_quantizers
    graph=torch.fx.Graph();acc=graph.placeholder('acc')
    real=graph.call_function(torch.ops.m2m_q.exact_readout.default,(acc,.015625,[.015625]))
    source=graph.call_function(torch.ops.aten.neg.default,(real,)) if escape=='arithmetic' else real
    q=graph.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default,(source,.015625,0,-128,127,torch.int8))
    if escape=='float_output':graph.output((q,real))
    elif escape=='different_quantizer':
        other=graph.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default,(real,.03125,0,-128,127,torch.int8));graph.output((q,other))
    else:graph.output(q)
    model=torch.fx.GraphModule(torch.nn.Module(),graph);before=model.code
    assert not fuse_readout_quantizers(model)
    assert model.code==before

