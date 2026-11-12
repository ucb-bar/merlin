"""Target-independent graph geometry, source semantics and closed-body refusals."""

import pytest

from merlin.frontends.linalg_tensor_graph import read_tensor_graph
from merlin.semantic_compiler.model import KernelRequest


def read(source):
    return read_tensor_graph(source, entry="compute", target_identity="arbitrary-device")


def pointwise(shape=(3, 7), broadcast=(3, 1), dtype="f32", operation="sub"):
    rank = len(shape)
    dims = ",".join("d" + str(i) for i in range(rank))
    projection = ",".join("0" if a == 1 and b != 1 else "d" + str(i) for i, (a, b) in enumerate(zip(broadcast, shape)))

    def typ(s):
        return "tensor<" + "x".join(map(str, s)) + "x" + dtype + ">"

    out, rhs = typ(shape), typ(broadcast)
    return f"""module {{
      func.func @compute(%a: {out}, %b: {rhs}) -> {out} {{
        %empty = tensor.empty() : {out}
        %result = linalg.generic {{indexing_maps = [affine_map<({dims})->({dims})>,
          affine_map<({dims})->({projection})>, affine_map<({dims})->({dims})>],
          iterator_types = [{",".join('"parallel"' for _ in shape)}]}}
          ins(%a, %b : {out}, {rhs}) outs(%empty : {out}) {{
            ^bb0(%x: {dtype}, %y: {dtype}, %old: {dtype}):
              %z = arith.{operation}f %x, %y : {dtype}
              linalg.yield %z : {dtype}
          }} -> {out}
        func.return %result : {out}
      }}
    }}"""


def reduction(shape=(3, 7), axis=1, dtype="f32", kind="sum"):
    src = "tensor<" + "x".join(map(str, shape)) + "x" + dtype + ">"
    dst = "tensor<" + "x".join(str(d) for i, d in enumerate(shape) if i != axis) + "x" + dtype + ">"
    literal = (
        "0.000000e+00"
        if kind == "sum"
        else {"f16": "0xfc00", "bf16": "0xff80", "f32": "0xff800000", "f64": "0xfff0000000000000"}[dtype]
    )
    operation = "addf" if kind == "sum" else "maximumf"
    return f"""module {{
      func.func @compute(%a: {src}) -> {dst} {{
        %literal = arith.constant {literal} : {dtype}
        %init = tensor.splat %literal : {dst}
        %r = linalg.reduce ins(%a:{src}) outs(%init:{dst}) dimensions = [{axis}]
          (%x: {dtype}, %y: {dtype}) {{
            %z = arith.{operation} %x, %y : {dtype}
            linalg.yield %z : {dtype}
          }}
        func.return %r : {dst}
      }}
    }}"""


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
@pytest.mark.parametrize("shape,broadcast", [((3, 7), (3, 1)), ((2, 5, 7), (1, 5, 1))])
def test_pointwise_types_and_broadcasts(dtype, operation, shape, broadcast):
    graph = read(pointwise(shape, broadcast, dtype, operation))
    assert graph.nodes[-1].op == operation
    assert graph.nodes[-1].type.shape == shape
    assert graph.nodes[-1].type.dtype == dtype
    assert graph.nodes[1].type.shape == broadcast
    assert KernelRequest.from_record(graph.record()) == graph


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("kind", ["sum", "max"])
def test_reductions_preserve_axis_seed_and_order(dtype, axis, kind):
    graph = read(reduction(axis=axis, dtype=dtype, kind=kind))
    assert graph.nodes[-1].op == "reduce_" + kind
    assert dict(graph.nodes[-1].attrs) == dict(axis=axis, reduction_order="source-unspecified")
    assert graph.nodes[-1].inputs == ("input0", "node1")
    assert graph.nodes[-1].type.shape == ((7,) if axis == 0 else (3,))


@pytest.mark.parametrize(
    "old,new",
    [
        ("arith.subf %x, %y", "arith.subf %x, %old"),
        ("linalg.yield %z", "linalg.yield %old"),
        ("arith.subf %x, %y : f32", "arith.subf %x, %y fastmath<fast> : f32"),
        ("->(d0,0)", "->(0,d1)"),
        ("%empty =", "%dead = tensor.empty() : tensor<3x7xf32>\n %empty ="),
        ("module {", 'module attributes {unknown = "semantic"} {'),
    ],
)
def test_semantic_mutations_refuse(old, new):
    source = pointwise()
    assert old in source
    with pytest.raises(Exception):
        read(source.replace(old, new))


def test_reduction_seed_and_scalar_operands_refuse():
    for source in (
        reduction().replace("0.000000e+00", "-0.000000e+00"),
        reduction().replace("arith.addf %x, %y", "arith.addf %x, %x"),
        reduction(kind="max").replace("0xff800000", "0x7f800000"),
    ):
        with pytest.raises(ValueError):
            read(source)


def test_provenance_does_not_select_semantics_and_outputs_keep_order():
    source = pointwise()
    a = read(source)
    b = read(source.replace("module {", 'module attributes {prov.family = "unrelated"} {'))
    assert a.nodes == b.nodes and a.source_identity != b.source_identity
    source = source.replace(") -> tensor<3x7xf32> {", ") -> (tensor<3x7xf32>, tensor<3x7xf32>) {", 1)
    source = source.replace(
        "func.return %result : tensor<3x7xf32>", "func.return %result, %a : tensor<3x7xf32>, tensor<3x7xf32>"
    )
    assert read(source).outputs == ("node2", "input0")


def dense(dtype="f32"):
    return f"""module {{
      func.func @compute(%a: tensor<3x5x{dtype}>, %w: tensor<7x5x{dtype}>) -> tensor<3x7x{dtype}> {{
        %te = tensor.empty() : tensor<5x7x{dtype}>
        %t = linalg.transpose ins(%w:tensor<7x5x{dtype}>) outs(%te:tensor<5x7x{dtype}>) permutation = [1, 0]
        %e = tensor.empty() : tensor<3x7x{dtype}>
        %z = arith.constant 0.000000e+00 : {dtype}
        %f = linalg.fill ins(%z:{dtype}) outs(%e:tensor<3x7x{dtype}>) -> tensor<3x7x{dtype}>
        %r = linalg.matmul ins(%a, %t : tensor<3x5x{dtype}>, tensor<5x7x{dtype}>)
          outs(%f:tensor<3x7x{dtype}>) -> tensor<3x7x{dtype}>
        func.return %r : tensor<3x7x{dtype}>
      }}
    }}"""


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
def test_rectangular_transpose_and_dense_keep_source_precision(dtype):
    graph = read(dense(dtype))
    assert [n.op for n in graph.nodes] == ["input", "input", "transpose", "matmul"]
    assert graph.nodes[2].type.shape == (5, 7)
    assert graph.nodes[-1].inputs == ("input0", "node2")
    assert graph.nodes[-1].type.dtype == dtype
    assert dict(graph.nodes[-1].attrs) == {"reduction_order": "source-unspecified"}


def test_nonzero_matmul_seed_refuses():
    with pytest.raises(ValueError, match="positive zero"):
        read(dense().replace("0.000000e+00", "1.000000e+00"))


@pytest.mark.parametrize("kind", ["exp", "relu"])
def test_unary_sources(kind):
    body = (
        """%z = math.exp %x : f32"""
        if kind == "exp"
        else """
       %zero = arith.constant 0.000000e+00 : f32
       %z = arith.maximumf %x, %zero : f32"""
    )
    source = """module {
      func.func @compute(%a: tensor<3x7xf32>) -> tensor<3x7xf32> {
        %e = tensor.empty() : tensor<3x7xf32>
        %r = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
          iterator_types = ["parallel", "parallel"]} ins(%a:tensor<3x7xf32>) outs(%e:tensor<3x7xf32>) {
          ^bb0(%x: f32, %old: f32):
          BODY
          linalg.yield %z : f32
        } -> tensor<3x7xf32>
        func.return %r : tensor<3x7xf32>
      }
    }""".replace("BODY", body)
    assert read(source).nodes[-1].op == kind
    with pytest.raises(ValueError):
        read(source.replace("linalg.yield %z", "linalg.yield %old"))


def test_reduce_expand_dims_keeps_ssa_edge():
    source = reduction().replace(") -> tensor<3xf32> {", ") -> tensor<3x1xf32> {", 1)
    source = source.replace(
        "func.return %r : tensor<3xf32>",
        """
      %expanded = tensor.expand_shape %r [[0 : i64, 1 : i64]] output_shape [3, 1]
        : tensor<3xf32> into tensor<3x1xf32>
      func.return %expanded : tensor<3x1xf32>""",
    )
    graph = read(source)
    assert graph.nodes[-1].op == "expand_dims"
    assert graph.nodes[-1].inputs == (graph.nodes[-2].id,)
    assert graph.nodes[-1].type.shape == (3, 1)


@pytest.mark.parametrize("mutation", ["operands", "seed_type", "body_types", "map", "flags"])
def test_named_matmul_does_not_hide_a_changed_scalar_body(mutation):
    from io import StringIO

    from xdsl.dialects import arith, builtin
    from xdsl.dialects.linalg.ops import MatmulOp
    from xdsl.printer import Printer

    from merlin.frontends.linalg_text import parse_mlir_text

    module = parse_mlir_text(dense())
    op = next(op for op in module.walk() if type(op) is MatmulOp)
    block = op.body.block
    if mutation == "operands":
        block.first_op.operands = (block.args[0], block.args[0])
    elif mutation == "seed_type":
        op.outputs[0]._type = builtin.TensorType(builtin.f64, (3, 7))
    elif mutation == "body_types":
        block.args[0]._type = builtin.f64
    elif mutation == "map":
        op.properties["indexing_maps"] = builtin.ArrayAttr(tuple(op.indexing_maps)[::-1])
    else:
        block.first_op.properties["fastmath"] = arith.FastMathFlagsAttr("fast")
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    with pytest.raises(Exception):
        read(stream.getvalue())


def test_nan_literal_is_not_silently_canonicalized():
    with pytest.raises(ValueError, match="NaN"):
        read(reduction().replace("0.000000e+00", "0x7fc00123"))


def test_higher_rank_reduction_keeps_other_extents():
    graph = read(reduction(shape=(2, 5, 7), axis=1))
    assert graph.nodes[-1].type.shape == (2, 7)
