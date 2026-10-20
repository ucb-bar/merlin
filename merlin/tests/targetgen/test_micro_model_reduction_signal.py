"""A normalization/reduction seam must not erase the composed test's signal."""

import pytest

from merlin.targetgen import micro_model as MM


@pytest.mark.parametrize("extent", [7, 16, 32])
def test_normalized_reduction_composition_rejects_constant_answers(extent):
    torch = pytest.importorskip("torch")
    spec = MM.MicroModelSpec(
        extent=extent,
        layers=[
            MM.LayerRequirement(family, "f32", side, emitted_op=op)
            for family, side, op in (
                ("contraction", MM.ACCELERATOR, "matmul"),
                ("normalization", MM.HOST, "layer_norm"),
                ("reduction", MM.HOST, "reduce_mean"),
                ("contraction", MM.ACCELERATOR, "matmul"),
            )
        ],
    )
    scope = {}
    exec(MM.emit_pytorch(spec), scope)
    model, inputs = scope["get_model_and_inputs"]()
    with torch.no_grad():
        result = model(*inputs)
    assert result.shape == inputs[0].shape
    assert bool(torch.isfinite(result).all())
    assert not torch.allclose(result, torch.zeros_like(result), atol=0.03125, rtol=0.02)
    assert not torch.allclose(result, result.mean().expand_as(result), atol=0.03125, rtol=0.02)
