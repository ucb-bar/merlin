"""Source-only storage planning reuses the checked bounded-window transform."""

import pytest
from test_im2col_panel_pack import _module
from test_integer_matmul_calls import EFFECTS

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.convolution_routes import prepare_integer_convolution_panels
from merlin.llvmlower.integer_matmul_calls import prepare_integer_matmul_calls
from merlin.transforms.certificates import ir_text


def source(*, splat=False, weight_views=False):
    text = _module(c=2, kh=2, kw=3, n=2, oh=3, ow=7, f=3, sh=2, sw=1, dh=1, dw=2)
    if splat:
        text = text.replace(
            "%acc = linalg.fill ins(%c0 : i32) outs(%ae : tensor<3x42xi32>) -> tensor<3x42xi32>",
            "%acc = tensor.splat %c0 : tensor<3x42xi32>",
        )
    if weight_views:
        text = text.replace("    %ae =", """
    %weight_flat = tensor.collapse_shape %w [[0, 1]] : tensor<3x12xi8> into tensor<36xi8>
    %weight_view = tensor.expand_shape %weight_flat [[0, 1]] output_shape [3, 12]
      : tensor<36xi8> into tensor<3x12xi8>
    %ae =""").replace("ins(%w, %col", "ins(%weight_view, %col")
    module = parse_mlir_text(text)
    from xdsl.dialects.builtin import StringAttr

    for op in module.walk():
        if op.name != "builtin.module":
            op.attributes["prov.trace_role"] = StringAttr("diagnostic")
    return ir_text(module)


@pytest.mark.parametrize("splat,weight_views", [(False, False), (True, False), (True, True)])
def test_payload_bound_tail_coverage_and_nested_products(splat, weight_views):
    text = source(splat=splat, weight_views=weight_views)
    result, report = prepare_integer_convolution_panels(text, max_packing_bytes=60)
    assert result != text and report["source_sha256"] != report["output_sha256"]
    route, = report["routes"]
    assert route["selected"] == "panel" and route["width"] == 3
    assert route["full_packing_bytes"] == 504
    assert route["realization"]["resident_packing_bytes"] == 48
    parse_mlir_text(result).verify()
    prepared = prepare_integer_matmul_calls(result, select=lambda call: True, borrowed_effects=EFFECTS)
    assert sorted(call.n for call in prepared.calls) == [1, 3]
    assert all((call.m, call.k) == (3, 12) for call in prepared.calls)
    assert '"scf.for"' in prepared.source


@pytest.mark.parametrize("budget,candidates,reason", [
    (504, 128, "full_payload_within_budget"),
    (11, 128, "no_panel_within_payload_budget"),
    (60, 1, "no_panel_within_payload_budget"),
])
def test_no_legal_or_needed_panel_returns_original_bytes(budget, candidates, reason):
    text = source(splat=True)
    result, report = prepare_integer_convolution_panels(text, max_packing_bytes=budget, max_candidates=candidates)
    assert result == text and report["output_sha256"] == report["source_sha256"]
    assert report["routes"][0]["reason"] == reason


@pytest.mark.parametrize("mutation", ["unsigned", "nonzero", "extra_observer", "unknown_attribute"])
def test_source_refusals_preserve_original_bytes(mutation):
    from xdsl.dialects.builtin import UnitAttr

    text = source()
    if mutation == "unsigned":
        text = text.replace('"arith.extsi"', '"arith.extui"')
    elif mutation == "nonzero":
        text = text.replace("value = 0 : i32", "value = 1 : i32")
    else:
        module = parse_mlir_text(text)
        gather = next(o for o in module.walk() if o.name == "linalg.generic")
        if mutation == "unknown_attribute":
            gather.attributes["unknown"] = UnitAttr()
        else:
            from xdsl.dialects import builtin, func

            function = next(o for o in module.walk() if isinstance(o, func.FuncOp))
            function.properties["function_type"] = builtin.FunctionType.from_lists(
                list(function.function_type.inputs), [*function.function_type.outputs, gather.results[0].type],
            )
            returned = function.body.block.last_op
            returned.operands = (*returned.operands, gather.results[0])
        text = ir_text(module)
    result, report = prepare_integer_convolution_panels(text, max_packing_bytes=60)
    assert result == text and not report["routes"]


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_payload_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="positive"):
        prepare_integer_convolution_panels(source(), max_packing_bytes=budget)
