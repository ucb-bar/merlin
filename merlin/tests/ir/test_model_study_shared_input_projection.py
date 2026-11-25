"""Source-only projection groups keep independent host arithmetic and observers."""

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
from merlin.llvmlower.sibling_components import prepare_sibling_components
from merlin.llvmlower.sibling_source_route import _command_buffer
from merlin.runtime.linalg_numpy import evaluate
from merlin.xdsl_dialects._common import text


def projection_source(widths, *, aliased_rhs=False, distinct_lhs=False):
    """Independent signed products with unequal column scales and a raw observer."""
    m, k = 3, 5
    arguments = [f"%a:tensor<{m}x{k}xi8>", f"%other:tensor<{m}x{k}xi8>"]
    arguments += [f"%w{i}:tensor<{k}x{n}xi8>" for i, n in enumerate(widths)]
    arguments += [f"%scale{i}:tensor<{n}xf32>" for i, n in enumerate(widths)]
    results = [f"tensor<{m}x{n}xf32>" for n in widths] + [f"tensor<{m}x{widths[0]}xi32>"]
    lines = [
        "module {func.func @forward(" + ",".join(arguments) + ")->(" + ",".join(results) + "){%z=arith.constant 0:i32"
    ]
    for i, n in enumerate(widths):
        lhs = "other" if distinct_lhs and i else "a"
        rhs = "w0" if aliased_rhs else f"w{i}"
        lines.append(f"""
          %seed{i}=tensor.splat %z:tensor<{m}x{n}xi32>
          %product{i}=linalg.generic {{
            indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,
                           affine_map<(m,n,k)->(m,n)>],
            iterator_types=["parallel","parallel","reduction"]}}
          ins(%{lhs},%{rhs}:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>)
          outs(%seed{i}:tensor<{m}x{n}xi32>) {{
          ^bb0(%x:i8,%y:i8,%acc:i32):
            %xx=arith.extsi %x:i8 to i32
            %yy=arith.extsi %y:i8 to i32
            %p=arith.muli %xx,%yy:i32
            %sum=arith.addi %acc,%p:i32
            linalg.yield %sum:i32
          }} ->tensor<{m}x{n}xi32>
          %empty{i}=tensor.empty():tensor<{m}x{n}xf32>
          %readout{i}=linalg.generic {{
            indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(n)>,
                           affine_map<(m,n)->(m,n)>],
            iterator_types=["parallel","parallel"]}}
          ins(%product{i},%scale{i}:tensor<{m}x{n}xi32>,tensor<{n}xf32>)
          outs(%empty{i}:tensor<{m}x{n}xf32>) {{
          ^bb0(%v:i32,%scale:f32,%unused:f32):
            %fp=arith.sitofp %v:i32 to f32
            %scaled=arith.mulf %fp,%scale:f32
            linalg.yield %scaled:f32
          }} ->tensor<{m}x{n}xf32>""")
    lines.append(
        "return "
        + ",".join([f"%readout{i}" for i in range(len(widths))] + ["%product0"])
        + ":"
        + ",".join(results)
        + "}}"
    )
    return parse_mlir_text("\n".join(lines))


def source_effect_fixture():
    # Only the local NumPy callback below satisfies these premises. This never
    # qualifies an accelerator, foreign kernel or serialized effect receipt.
    return BorrowedBufferEffects("local-test-synchronous-array-products", True, True, True)


@pytest.mark.parametrize("widths", [(7, 2, 3), (11, 11)])
def test_projection_tails_keep_all_scales_raw_observers_and_result_order(widths):
    source = projection_source(widths)
    before = text(source, generic=True)
    prepared = prepare_sibling_components(source, effects=source_effect_fixture())
    assert len(prepared.components) == 1
    component = prepared.components[0]
    commands, group = _command_buffer(component.core, "test-target")
    assert group.widths == widths
    products = [c for c in commands["commands"] if c["opcode"] == "MATMUL_RESIDENT"]
    assert len(products) == len(widths)
    assert {p["operands"]["lhs"] for p in products} == {"input_0"}
    assert [r["attributes"] for r in commands["commands"] if r["opcode"] == "COMMIT"] == [
        {"epilogue": [], "output_dtype": "i32"} for _ in widths
    ]
    lhs = np.arange(15, dtype=np.int8).reshape(3, 5) - 7
    weights = [(np.arange(5 * n, dtype=np.int16).reshape(5, n) % 17 - 8).astype(np.int8) for n in widths]
    scales = [np.arange(1, n + 1, dtype=np.float32) * np.float32((i + 1) / 32) for i, n in enumerate(widths)]
    arguments = [lhs, -lhs, *weights, *scales]

    def original_products(values):
        return tuple(
            (values[0].astype(np.int64) @ rhs.astype(np.int64)).astype(np.int32) for rhs in values[1 : 1 + len(widths)]
        )

    expected = evaluate(source, arguments)
    actual = evaluate(prepared.module, arguments, calls={component.writer.symbol: original_products})
    assert len(actual) == len(widths) + 1
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert text(source, generic=True) == before
    assert prepared.receipt["effect_or_placement_grant"] is False
    assert component.receipt["physical_effects_and_target_execution_established"] is False


@pytest.mark.parametrize("options", [{"aliased_rhs": True}, {"distinct_lhs": True}])
def test_equal_shapes_do_not_replace_distinct_input_and_weight_identity(options):
    source = projection_source((7, 7), **options)
    before = text(source, generic=True)
    prepared = prepare_sibling_components(source, effects=source_effect_fixture())
    assert not prepared.components
    assert text(source, generic=True) == before
    assert prepared.receipt["unmatched_source_retained"] is True


def test_source_group_cannot_be_selected_with_an_unqualified_effect_record():
    source = projection_source((7, 2, 3))
    with pytest.raises(ValueError, match="typed effects"):
        prepare_sibling_components(source, effects={"complete_before_return": True})
