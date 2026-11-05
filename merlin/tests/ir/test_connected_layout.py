"""Connected exact scalar/reduction layout choices reach real host execution."""
import ctypes
import hashlib
import subprocess

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.graph_optimizations import (
    GraphOptimizationContract, optimize_host_graph,
)
from merlin.llvmlower.host_layout_planning import plan_islands
from merlin.llvmlower.layout_conformance import ordered_reduction_bounds


def source(m=3, k=4, n=5, width=2, *, public_quant=False):
    result = f"tensor<{n}x{width}xi32>,tensor<{m}x{n}xi32>"
    if public_quant:
        result += f",tensor<{n}x{m}xi8>"
    return f'''module {{
 func.func @kernel(%a:tensor<{m}x{k}xi8>,%b:tensor<{k}x{n}xi8>,
  %seed:tensor<{m}x{n}xi32>,%c:tensor<{m}x{width}xi8>,%last:tensor<{n}x{width}xi32>) -> ({result}) {{
  %p=linalg.generic {{indexing_maps=[affine_map<(i,j,r)->(i,r)>,
   affine_map<(i,j,r)->(r,j)>,affine_map<(i,j,r)->(i,j)>],
   iterator_types=["parallel","parallel","reduction"]}}
   ins(%a,%b:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%seed:tensor<{m}x{n}xi32>) {{
   ^bb0(%av:i8,%bv:i8,%acc:i32):
    %ai=arith.extsi %av:i8 to i32
    %bi=arith.extsi %bv:i8 to i32
    %mul=arith.muli %ai,%bi:i32
    %sum=arith.addi %acc,%mul:i32
    linalg.yield %sum:i32
  }} -> tensor<{m}x{n}xi32>
  %et=tensor.empty():tensor<{n}x{m}xi32>
  %t=linalg.transpose ins(%p:tensor<{m}x{n}xi32>) outs(%et:tensor<{n}x{m}xi32>) permutation=[1,0]
  %eq=tensor.empty():tensor<{n}x{m}xi8>
  %q=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
   iterator_types=["parallel","parallel"]}}
   ins(%t:tensor<{n}x{m}xi32>) outs(%eq:tensor<{n}x{m}xi8>) {{
   ^bb0(%v:i32,%unused:i8):
    %lo=arith.constant -128:i32
    %hi=arith.constant 127:i32
    %lower=arith.maxsi %v,%lo:i32
    %upper=arith.minsi %lower,%hi:i32
    %narrow=arith.trunci %upper:i32 to i8
    linalg.yield %narrow:i8
  }} -> tensor<{n}x{m}xi8>
  %answer=linalg.generic {{indexing_maps=[affine_map<(i,j,r)->(i,r)>,
   affine_map<(i,j,r)->(r,j)>,affine_map<(i,j,r)->(i,j)>],
   iterator_types=["parallel","parallel","reduction"]}}
   ins(%q,%c:tensor<{n}x{m}xi8>,tensor<{m}x{width}xi8>) outs(%last:tensor<{n}x{width}xi32>) {{
   ^bb0(%qv:i8,%cv:i8,%acc:i32):
    %qi=arith.extsi %qv:i8 to i32
    %ci=arith.extsi %cv:i8 to i32
    %mul=arith.muli %qi,%ci:i32
    %sum=arith.addi %acc,%mul:i32
    linalg.yield %sum:i32
  }} -> tensor<{n}x{width}xi32>
  func.return %answer,%p{',%q' if public_quant else ''}:{result}
 }}
}}'''


def select(program, **choices):
    contract = GraphOptimizationContract(
        hashlib.sha256(program.encode()).hexdigest(), "kernel", propagate_layout=True,
        independent_element_order_unobserved=True, physical_layout=True,
        connected_layout=True, **choices,
    )
    return optimize_host_graph(program, contract)


@pytest.mark.parametrize("shape", [(3, 4, 5, 2), (4, 3, 7, 3)])
def test_connected_contraction_quantizer_consumer_removes_real_conversion(shape):
    program = source(*shape)
    selected, report = select(program)
    before, after = parse_mlir_text(program), parse_mlir_text(selected)
    original = [op for op in before.walk() if ordered_reduction_bounds(op) is not None]
    actual = [op for op in after.walk() if ordered_reduction_bounds(op) is not None]
    assert len(actual) == len(original) == 2
    assert report["layout"]["before_transpose_bytes"] == shape[0] * shape[2] * 4
    assert report["layout"]["after_transpose_bytes"] == 0
    assert not any(op.name == "linalg.transpose" for op in after.walk())
    assert tuple(actual[1].inputs[0].type.get_shape()) == (shape[0], shape[2])
    assert actual[1].indexing_maps.data[0].data.results == tuple(reversed(original[1].indexing_maps.data[0].data.results))
    for op in actual:
        assert op.outputs[0] is op.parent.args[2 if op is actual[0] else 4]
        assert op.body.block.args[-1].uses
        assert tuple(x.data.value for x in op.iterator_types) == ("parallel", "parallel", "reduction")
    assert report["layout"]["pointwise_work"]["before_scalar_evaluations"] == report["layout"]["pointwise_work"]["after_scalar_evaluations"]


def test_default_pointwise_scope_and_public_quantized_observer_keep_conversion():
    program = source()
    assert all(ordered_reduction_bounds(op) is None for island in plan_islands(parse_mlir_text(program))
               for op in island.operations)
    default = GraphOptimizationContract(hashlib.sha256(program.encode()).hexdigest(), "kernel",
        propagate_layout=True, independent_element_order_unobserved=True, physical_layout=True)
    _, pointwise_report = optimize_host_graph(program, default)
    assert pointwise_report["layout"]["after_transpose_bytes"] == 3 * 5
    observed = source(public_quant=True)
    selected, report = select(observed)
    assert report["layout"]["after_transpose_bytes"] == 3 * 5  # The public INT8 boundary still materializes.
    assert any(op.name == "linalg.transpose" for op in parse_mlir_text(selected).walk())
    assert parse_mlir_text(selected).body.block.first_op.function_type == parse_mlir_text(observed).body.block.first_op.function_type


def test_unknown_observer_still_demands_canonical_materialization():
    program = source().replace("module {", "module {func.func private @observe(tensor<5x3xi8>)", 1)
    program = program.replace("  %answer=linalg.generic", "  func.call @observe(%q):(tensor<5x3xi8>)->()\n  %answer=linalg.generic")
    selected, report = select(program)
    assert report["layout"]["after_transpose_bytes"] == 15
    module = parse_mlir_text(selected)
    observer = next(op for op in module.walk() if op.name == "func.call")
    assert tuple(observer.operands[0].type.get_shape()) == (5, 3)
    assert observer.operands[0].owner.name == "linalg.transpose"


def test_initializer_replacement_is_rejected_by_independent_coordinate_check():
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import i32
    from merlin.llvmlower.graph_optimizations import _check_layout_output, _layout_bodies

    program = source()
    selected, _ = select(program)
    original, changed = parse_mlir_text(program), parse_mlir_text(selected)
    reduction = next(op for op in changed.walk() if ordered_reduction_bounds(op) is not None)
    zero = arith.ConstantOp.from_int_and_width(0, i32)
    initialized = tensor.SplatOp.create(operands=[zero.result], result_types=[reduction.outputs[0].type])
    reduction.parent.insert_ops_before([zero, initialized], reduction)
    reduction.operands = (*reduction.inputs, initialized.results[0])
    changed.verify()
    with pytest.raises(ValueError, match="logical output coordinates"):
        _check_layout_output(original, changed, _layout_bodies(original), connected=True)


def test_ordinary_pre_group_policy_activates_and_reopens_exact_source(tmp_path):
    import json
    import yaml
    from merlin.llvmlower.host_graph_policy import prepare_group_layout, read_host_graph_policy
    from merlin.llvmlower.graph_optimizations import recheck_host_graph_artifacts
    from merlin.llvmlower.host_optimization_request import POLICY_SCHEMA

    software = tmp_path / "software.yaml"
    software.write_text(yaml.safe_dump({"schema": "merlin.software_spec.v1", "status": "reviewed", "target": "neutral",
        "host_optimization": {"schema": POLICY_SCHEMA, "permitted_stages": ["logical_host"],
            "permitted_optimizations": ["propagate_layout"], "independent_element_order_unobserved": True}}))
    policy = read_host_graph_policy(software, target="neutral", entry="kernel", connected_layout=True)
    directory = tmp_path / "pre-group"
    selected = prepare_group_layout(parse_mlir_text(source()), policy, directory)
    assert not any(op.name == "linalg.transpose" for op in selected.walk())
    assert json.loads((directory / "host-graph-optimization.json").read_text())["contract"]["connected_layout"] is True
    assert recheck_host_graph_artifacts(directory, policy)["selected"]["sha256"]


def test_flexible_ports_cannot_relax_output_or_coupled_layout_constraints():
    from merlin.perf.physical_layout import LayoutOp, OperatorEncoding, PhysicalLayoutError

    option = (OperatorEncoding("canonical", "perm_0_1"),)
    for ports, coupled in [(("out",), False), (("input",), True), (("input", "input"), False)]:
        with pytest.raises(PhysicalLayoutError, match="distinct uncoupled inputs"):
            LayoutOp("op", "checked", ("input",), ("out",), coupled, option, flexible_inputs=ports)


@pytest.mark.parametrize("shape,public_quant", [((3, 4, 5, 2), False), ((4, 3, 7, 3), False), ((3, 4, 5, 2), True)])
def test_selected_source_emits_and_executes_nonzero_initialized_reductions(tmp_path, shape, public_quant):
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp
    from merlin.llvmlower.host_builder import PTR, FnBuilder
    from merlin.llvmlower.host_scalar_builder import FpBuilder
    from merlin.llvmlower.host_tensor_codegen import HostLinalg
    from merlin.llvmlower.toolchain import clang
    from merlin.xdsl_dialects._common import text

    tool = clang()
    if not tool.is_file() or not (tool.parent / "mlir-translate").is_file():
        pytest.skip("selected upstream LLVM tools unavailable")
    m, k, n, width = shape
    program = source(*shape, public_quant=public_quant)
    selected, _ = select(program)
    a = ((np.arange(m * k) * 7) % 29 - 14).astype(np.int8).reshape(m, k)
    b = ((np.arange(k * n) * 11) % 31 - 15).astype(np.int8).reshape(k, n)
    c = ((np.arange(m * width) * 5) % 19 - 9).astype(np.int8).reshape(m, width)
    seed = (np.arange(m * n) * 17 - 139).astype(np.int32).reshape(m, n)
    last = (np.arange(n * width) * 13 + 73).astype(np.int32).reshape(n, width)
    p = a.astype(np.int32) @ b.astype(np.int32) + seed
    q = np.clip(p.T, -128, 127).astype(np.int8)
    expected = q.astype(np.int32) @ c.astype(np.int32) + last
    for label, materialized in (("source", program), ("selected", selected)):
        function = parse_mlir_text(materialized).body.block.first_op
        count = 8 if public_quant else 7
        builder = FnBuilder([PTR] * count)
        HostLinalg(FpBuilder(builder), list(builder.entry.args[:5]), list(builder.entry.args[5:])).run(function)
        module = ModuleOp([llvm.FuncOp("execute_region", llvm.LLVMFunctionType([PTR] * count),
            linkage=llvm.LinkageAttr("external"), body=builder.finish())])
        module.verify()
        stem = label + "_" + "_".join(map(str, (*shape, int(public_quant))))
        mlir, ll, so = [tmp_path / (stem + suffix) for suffix in (".mlir", ".ll", ".so")]
        mlir.write_text(text(module, generic=True))
        subprocess.run([str(tool.parent / "mlir-translate"), "--mlir-to-llvmir", str(mlir), "-o", str(ll)],
                       check=True, capture_output=True, timeout=30)
        subprocess.run([str(tool), "-O2", "-shared", "-fPIC", str(ll), "-o", str(so)],
                       check=True, capture_output=True, timeout=30)
        execute = ctypes.CDLL(str(so)).execute_region
        execute.argtypes = [ctypes.c_void_p] * count
        guarded_output = np.full(expected.size + 8, 123456789, dtype=np.int32)
        output = guarded_output[4:-4].reshape(expected.shape)
        observed_producer = np.full_like(p, -987654321)
        observed_quant = np.full(q.shape, 17, dtype=np.int8)
        arguments = (a, b, seed, c, last, output, observed_producer)
        if public_quant:
            arguments += (observed_quant,)
        execute(*(value.ctypes.data for value in arguments))
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(observed_producer, p)
        if public_quant:
            np.testing.assert_array_equal(observed_quant, q)
        assert np.all(guarded_output[:4] == 123456789) and np.all(guarded_output[-4:] == 123456789)
        np.testing.assert_array_equal(seed, (np.arange(m * n) * 17 - 139).astype(np.int32).reshape(m, n))
        np.testing.assert_array_equal(last, (np.arange(n * width) * 13 + 73).astype(np.int32).reshape(n, width))
        assert np.count_nonzero(seed) == seed.size and np.count_nonzero(last) == last.size
