"""Constant index specialization is checked over the whole static domain."""

import pytest

from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def source(dtype="f32", extra="", column="%cols", body_attrs=""):
    return f"""module {{
      func.func @forward(%table: tensor<4x5x{dtype}>, %rows: tensor<2x1xi64>,
                         %cols: tensor<2x3xi64>) -> tensor<2x3x{dtype}> {{
        {extra}
        %empty = tensor.empty() : tensor<2x3x{dtype}>
        %read = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,0)>,
          affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
          iterator_types = ["parallel", "parallel"]}}
          ins(%rows, {column} : tensor<2x1xi64>, tensor<2x3xi64>)
          outs(%empty : tensor<2x3x{dtype}>) {{
        ^bb0(%r: i64, %c: i64, %unused: {dtype}):
          %i = arith.index_cast %r {body_attrs} : i64 to index
          %j = arith.index_cast %c : i64 to index
          %v = tensor.extract %table[%i, %j] : tensor<4x5x{dtype}>
          linalg.yield %v : {dtype}
        }} -> tensor<2x3x{dtype}>
        return %read : tensor<2x3x{dtype}>
      }}
    }}"""


def compile_source(text=None, rows=(1, 3), columns=(0, 2, 4, 0, 2, 4), runtime=False):
    bindings = {1: b"".join(x.to_bytes(8, "little", signed=True) for x in rows)}
    if not runtime:
        bindings[2] = b"".join(x.to_bytes(8, "little", signed=True) for x in columns)
    return translate_linalg_text(
        text or source(), entry="forward", target_identity="synthetic", argument_constants=bindings
    )


@pytest.mark.parametrize("dtype", ("f32", "i1", "bf16"))
def test_affine_read_preserves_table_bits_and_projections(dtype):
    translated = compile_source(source(dtype))
    graph = translated.request
    value = (0, 1) * 10 if dtype == "i1" else tuple(range(20))
    if dtype == "f32":
        value = (0x80000000, 0x7FC01234, 1, 0x7F800000, 0xFF800000) * 4
    outputs = evaluate_graph(
        graph, {"arg0": TensorValue(graph.node("arg0").type, value)}, constants=translated.constants
    )
    assert outputs[0].elements == tuple(value[r * 5 + c] for r in (1, 3) for c in (0, 2, 4))
    node = graph.node(graph.outputs[0])
    assert node.op == "gather" and node.inputs == ("arg0",)
    assert node.index_maps[0].coefficients == ((2, 0), (0, 2))
    assert node.index_maps[0].offsets == (1, 0)
    assert type(graph).from_record(graph.record()).digest() == graph.digest()


@pytest.mark.parametrize(
    "columns,reason",
    (
        ((0, 2, 3, 0, 2, 3), "affine"),
        ((0, 2, 5, 0, 2, 5), "bounds"),
        ((-1, 1, 3, -1, 1, 3), "bounds"),
    ),
)
def test_all_indices_checked_including_non_affine_and_out_of_bounds(columns, reason):
    with pytest.raises(ValueError, match=reason):
        compile_source(columns=columns)


def test_runtime_indices_cannot_be_specialized_from_sample_values():
    with pytest.raises(ValueError, match="declared compile-time constant"):
        compile_source(runtime=True)


def test_constant_arithmetic_and_views_produce_a_checked_map():
    extra = """%one = arith.constant dense<1> : tensor<2x3xi64>
      %empty2 = tensor.empty() : tensor<2x3xi64>
      %sum = linalg.generic {indexing_maps = [affine_map<(a,b)->(a,b)>,
        affine_map<(a,b)->(a,b)>, affine_map<(a,b)->(a,b)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%cols, %one : tensor<2x3xi64>, tensor<2x3xi64>) outs(%empty2 : tensor<2x3xi64>) {
      ^bb0(%x: i64, %y: i64, %unused2: i64):
        %added = arith.addi %x, %y : i64
        linalg.yield %added : i64
      } -> tensor<2x3xi64>
      %flat = tensor.collapse_shape %sum [[0, 1]] : tensor<2x3xi64> into tensor<6xi64>
      %expanded = tensor.expand_shape %flat [[0, 1]] output_shape [2, 3] : tensor<6xi64> into tensor<2x3xi64>"""
    translated = compile_source(source(extra=extra, column="%expanded"), columns=(-1, 1, 3, -1, 1, 3))
    assert translated.request.nodes[-1].index_maps[0].coefficients == ((2, 0), (0, 2))


def test_unknown_scalar_attributes_and_changed_operand_connections_refuse():
    with pytest.raises(ValueError, match="only i64 casts"):
        compile_source(source(body_attrs='{unknown = "effect"}'))
    with pytest.raises(ValueError, match="index, or yielded"):
        compile_source(source().replace("%table[%i, %j]", "%table[%j, %i]"))


def test_work_budget_is_an_explicit_resource_failure():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.semantic_compiler.constant_indexing import translate_constant_index_read
    from merlin.semantic_compiler.target_binding import NativeCompilationError

    op = next(op for op in parse_mlir_text(source()).walk() if op.name == "linalg.generic")
    with pytest.raises(NativeCompilationError, match="finite-domain work budget") as error:
        translate_constant_index_read(op, {}, [], {}, node_id="test", max_work=1)
    assert error.value.status == "resource_limit"
