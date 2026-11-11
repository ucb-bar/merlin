"""Source graph edges, retained outputs and fail-closed scalar correspondence."""

import pytest

from merlin.frontends.linalg_float_graph import read_float_graph
from merlin.semantic_compiler.model import KernelRequest
from test_ordered_float_dense import POLICY, source


def graph_source():
    original = source(m=32, k=32, n=32)
    relu = '''
    %re = tensor.empty() : tensor<32x32xbf16>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%out : tensor<32x32xbf16>) outs(%re : tensor<32x32xbf16>)
      attrs = {merlin.numeric_semantics = "bf16-sign-clamp-v1",
               merlin.input_domain = "finite-normal-or-zero-bf16"} {
      ^bb0(%x: bf16, %old: bf16):
        %z = arith.constant 0.000000e+00 : bf16
        %r = arith.maximumf %x, %z : bf16
        linalg.yield %r : bf16
    } -> tensor<32x32xbf16>
    %pe = tensor.empty() : tensor<32x32xf8E4M3FN>
    %packed = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%relu : tensor<32x32xbf16>) outs(%pe : tensor<32x32xf8E4M3FN>)
      attrs = {merlin.numeric_semantics = "declared-cast-v1",
               merlin.input_domain = "declared-subset", merlin.scale_code = 127 : i32} {
      ^bb0(%x: bf16, %old: f8E4M3FN):
        %p = arith.truncf %x : bf16 to f8E4M3FN
        linalg.yield %p : f8E4M3FN
    } -> tensor<32x32xf8E4M3FN>
    func.return %out, %relu, %packed : tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xf8E4M3FN>'''
    return original.replace(') -> tensor<32x32xbf16> {',
                            ') -> (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xf8E4M3FN>) {', 1).replace(
                                'func.return %out : tensor<32x32xbf16>', relu)


def read(value):
    return read_float_graph(value, entry="dense", target_identity="test-target", ordered_policies=(POLICY,))


def test_producer_edges_retained_outputs_and_unresolved_cast_domain():
    graph = read(graph_source())
    assert tuple(node.op for node in graph.nodes) == ("input", "input", "ordered_dot", "relu", "cast")
    assert graph.outputs == ("node2", "node3", "node4")
    assert graph.nodes[2].inputs == ("input0", "input1")
    assert graph.nodes[3].inputs == ("node2",)
    assert graph.nodes[4].inputs == ("node3",)
    assert dict(graph.nodes[4].attrs) == {"input_domain": "declared-subset", "scale_code": 127}
    assert KernelRequest.from_record(graph.record()).digest() == graph.digest()
    assert len(graph.input_storages) == 2


@pytest.mark.parametrize("old,new", [
    ("arith.maximumf %x, %z", "arith.maximumf %old, %z"),
    ("%z = arith.constant 0.000000e+00", "%z = arith.constant -0.000000e+00"),
    ("arith.truncf %x", "arith.truncf %old"),
    ("merlin.scale_code = 127", "merlin.scale_code = 126"),
    ("linalg.yield %p", "linalg.yield %old"),
    ("merlin.numeric_semantics = \"bf16-sign-clamp-v1\"", "merlin.numeric_semantics = \"unknown\""),
    ("math.fma %ae, %be, %c", "math.fma %ae, %ae, %c"),
    ("%zero = arith.constant 0.000000e+00", "%zero = arith.constant 1.000000e+00"),
    ("    %re =", "    %dead = tensor.empty() : tensor<32x32xbf16>\n    %re ="),
])
def test_changed_body_initializer_policy_or_unused_source_refuses(old, new):
    original = graph_source()
    assert old in original
    with pytest.raises(Exception):
        read(original.replace(old, new))


def test_return_order_changes_graph_and_preserves_repeated_outputs():
    original = graph_source()
    changed = original.replace("func.return %out, %relu, %packed", "func.return %relu, %relu, %packed")
    graph = read(changed)
    assert graph.outputs == ("node3", "node3", "node4")
    assert graph.digest() != read(original).digest()


@pytest.mark.parametrize("kind,policy,domain", [
    ("add", "selected-f32-rne-bf16-chop-v1", "finite-normal-exact-fp32-sum-v1"),
    ("sub", "selected-f32-rne-bf16-chop-v1", "finite-normal-exact-fp32-difference-or-zero-v1"),
    ("mul", "bf16-product-rne-v1", "finite-normal-bf16-product-normal-v1"),
])
def test_explicit_chop_and_rne_pointwise_bodies(kind, policy, domain):
    extra = '''
      %half = arith.constant 32767 : i32
      %one = arith.constant 1 : i32
      %high = arith.shrui %bits, %shift : i32
      %lsb = arith.andi %high, %one : i32
      %bias = arith.addi %half, %lsb : i32
      %biased = arith.addi %bits, %bias : i32
      %rounded = arith.shrui %biased, %shift : i32
''' if kind == "mul" else '      %rounded = arith.shrui %bits, %shift : i32\n'
    value = f'''module {{
      func.func @dense(%a: tensor<8x12xbf16>, %b: tensor<8x12xbf16>) -> tensor<8x12xbf16> {{
        %empty = tensor.empty() : tensor<8x12xbf16>
        %out = linalg.generic {{indexing_maps = [affine_map<(i,j)->(i,j)>,
           affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]}}
          ins(%a, %b : tensor<8x12xbf16>, tensor<8x12xbf16>) outs(%empty : tensor<8x12xbf16>)
          attrs = {{merlin.numeric_semantics = "{policy}", merlin.input_domain = "{domain}"}} {{
          ^bb0(%x: bf16, %y: bf16, %old: bf16):
            %xf = arith.extf %x : bf16 to f32
            %yf = arith.extf %y : bf16 to f32
            %value = arith.{kind}f %xf, %yf : f32
            %bits = arith.bitcast %value : f32 to i32
            %shift = arith.constant 16 : i32
            {extra}
            %narrow = arith.trunci %rounded : i32 to i16
            %result = arith.bitcast %narrow : i16 to bf16
            linalg.yield %result : bf16
        }} -> tensor<8x12xbf16>
        func.return %out : tensor<8x12xbf16>
      }}
    }}'''
    graph = read(value)
    assert graph.nodes[-1].op == kind
    assert graph.nodes[-1].type.numerical_policy == policy
    assert graph.nodes[-1].type.shape == (8, 12)
    for bad in (value.replace('constant 16', 'constant 15'),
                value.replace('%xf, %yf : f32', '%xf, %xf : f32'),
                value.replace('linalg.yield %result', 'linalg.yield %old')):
        with pytest.raises(ValueError):
            read(bad)
