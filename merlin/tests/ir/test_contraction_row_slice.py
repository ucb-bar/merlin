"""Only unobserved parallel rows disappear; ordered dots and readouts survive."""

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.linalg.ops import GenericOp

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.contraction_row_slice import pushback_contraction_row_slices
from merlin.llvmlower.passes_xdsl import lower_quant_ext
from merlin.runtime.linalg_numpy import evaluate
from merlin.xdsl_dialects._common import text


def source(m=4, k=8, n=16, offset=3, rows=1, *, runtime_scale=False):
    scale_arg = f",%scale:tensor<{n}xf32>" if runtime_scale else ""
    scale = "" if runtime_scale else f"%scale=arith.constant dense<0.03125>:tensor<{n}xf32>"
    return f"""module {{
      func.func @forward(%lhs:tensor<{m}x{k}xi8>,%rhs:tensor<{k}x{n}xi8>{scale_arg}) -> tensor<1x{rows}x{n}xf32> {{
        %z=arith.constant 0:i32
        %init=tensor.splat %z:tensor<{m}x{n}xi32>
        %dot=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
          affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
          iterator_types=["parallel","parallel","reduction"]}}
          ins(%lhs,%rhs:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%init:tensor<{m}x{n}xi32>) {{
          ^bb0(%a:i8,%b:i8,%c:i32):
            %aa=arith.extsi %a:i8 to i32
            %bb=arith.extsi %b:i8 to i32
            %p=arith.muli %aa,%bb:i32
            %s=arith.addi %c,%p:i32
            linalg.yield %s:i32
        }} -> tensor<{m}x{n}xi32>
        %e=tensor.empty():tensor<{m}x{n}xf32>
        %cast=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
          iterator_types=["parallel","parallel"]}} ins(%dot:tensor<{m}x{n}xi32>) outs(%e:tensor<{m}x{n}xf32>) {{
          ^bb0(%cx:i32,%cu:f32):
            %cy=arith.sitofp %cx:i32 to f32
            linalg.yield %cy:f32
        }} -> tensor<{m}x{n}xf32>
        %flat=tensor.collapse_shape %cast [[0,1]]:tensor<{m}x{n}xf32> into tensor<{m * n}xf32>
        %view=tensor.expand_shape %flat [[0,1,2]] output_shape [1,{m},{n}]
          :tensor<{m * n}xf32> into tensor<1x{m}x{n}xf32>
        {scale}
        %se=tensor.empty():tensor<1x{m}x{n}xf32>
        %scaled=linalg.generic {{indexing_maps=[affine_map<(i,j,k)->(i,j,k)>,
          affine_map<(i,j,k)->(k)>,affine_map<(i,j,k)->(i,j,k)>],
          iterator_types=["parallel","parallel","parallel"]}}
          ins(%view,%scale:tensor<1x{m}x{n}xf32>,tensor<{n}xf32>) outs(%se:tensor<1x{m}x{n}xf32>) {{
          ^bb0(%sx:f32,%ss:f32,%su:f32):
            %sy=arith.mulf %sx,%ss:f32
            linalg.yield %sy:f32
        }} -> tensor<1x{m}x{n}xf32>
        %qs=arith.constant dense<0.5>:tensor<f32>
        %qz=arith.constant dense<0>:tensor<i64>
        %quant="quant_ext.quantize_per_tensor"(%scaled,%qs,%qz)
          <{{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}}>
          :(tensor<1x{m}x{n}xf32>,tensor<f32>,tensor<i64>)->tensor<1x{m}x{n}xi8>
        %out="quant_ext.dequantize_per_tensor"(%quant,%qs,%qz)<{{quant_min=-128:i64,quant_max=127:i64}}>
          :(tensor<1x{m}x{n}xi8>,tensor<f32>,tensor<i64>)->tensor<1x{m}x{n}xf32>
        %slice=tensor.extract_slice %out[0,{offset},0][1,{rows},{n}][1,1,1]
          :tensor<1x{m}x{n}xf32> to tensor<1x{rows}x{n}xf32>
        return %slice:tensor<1x{rows}x{n}xf32>
      }}
    }}"""


def run(module, arguments):
    canonical = module.clone()
    lower_quant_ext(canonical)
    return evaluate(canonical.body.block.first_op, arguments)


@pytest.mark.parametrize("shape", [(4, 8, 16, 3, 1), (7, 5, 13, 2, 3), (3, 1, 1, 0, 2)])
def test_aligned_ragged_and_multirow_are_bit_exact(shape):
    m, k, n, offset, rows = shape
    original = parse_mlir_text(source(*shape))
    before = text(original, generic=True)
    result, report = pushback_contraction_row_slices(original)
    assert report["rewrites"] == 1
    assert report["regions"][0]["selected_m_k_n"] == [rows, k, n]
    assert text(original, generic=True) == before
    roots = [
        op
        for op in result.walk()
        if isinstance(op, GenericOp) and any(it.data.value == "reduction" for it in op.iterator_types)
    ]
    (root,) = roots
    old = next(op for op in original.walk() if isinstance(op, GenericOp))
    assert root.body.is_structurally_equivalent(old.body)
    assert root.inputs[1].owner is result.body.block.first_op.body.block
    assert tuple(root.inputs[0].type.get_shape()) == (rows, k)
    assert tuple(root.outputs[0].type.get_shape()) == (rows, n)
    assert isinstance(root.outputs[0].owner, tensor.SplatOp)
    rng = np.random.default_rng(83)
    args = [rng.integers(-128, 128, (m, k), dtype=np.int8), rng.integers(-128, 128, (k, n), dtype=np.int8)]
    actual, expected = run(result, args), run(original, args)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    assert pushback_contraction_row_slices(result)[1]["rewrites"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "raw_observer",
        "epilogue_observer",
        "runtime_scale",
        "strict",
        "dynamic_slice",
        "stride",
        "column_slice",
        "seed",
        "index",
        "effect",
    ],
)
def test_unproved_observers_or_domains_leave_exact_original_ir(fault):
    program = source(runtime_scale=fault == "runtime_scale")
    if fault == "strict":
        program = program.replace("-> tensor<1x1x16xf32> {", "-> tensor<1x1x16xf32> attributes {strictfp} {")
    elif fault == "stride":
        program = program.replace("[0,3,0][1,1,16][1,1,1]", "[0,3,0][1,1,8][1,1,2]")
        program = program.replace("tensor<1x1x16xf32>", "tensor<1x1x8xf32>")
    elif fault == "column_slice":
        program = program.replace("[0,3,0][1,1,16]", "[0,3,0][1,1,8]")
        program = program.replace("tensor<1x1x16xf32>", "tensor<1x1x8xf32>")
    elif fault == "seed":
        program = program.replace("%z=arith.constant 0:i32", "%z=arith.constant 1:i32")
    elif fault == "index":
        program = program.replace(
            "%sy=arith.mulf %sx,%ss:f32", "%idx=linalg.index 1:index\n %sy=arith.mulf %sx,%ss:f32"
        )
    elif fault == "effect":
        program = program.replace(
            "%sy=arith.mulf %sx,%ss:f32", '"opaque.effect"() : () -> ()\n %sy=arith.mulf %sx,%ss:f32'
        )
    module = parse_mlir_text(program)
    forward = module.body.block.first_op
    if fault in {"raw_observer", "epilogue_observer"}:
        index = 0 if fault == "raw_observer" else -1
        producers = [op for op in module.walk() if isinstance(op, GenericOp)]
        observed = producers[index].results[0]
        returned = forward.body.block.last_op
        returned.operands = (*returned.operands, observed)
        forward.update_function_type()
    elif fault == "dynamic_slice":
        observer = next(op for op in module.walk() if isinstance(op, tensor.ExtractSliceOp))
        idx = arith.ConstantOp.from_int_and_width(3, builtin.IndexType())
        forward.body.block.insert_op_before(idx, observer)
        observer.operands = [observer.source, idx.result]
        observer.properties["static_offsets"] = builtin.DenseArrayBase.from_list(
            builtin.i64, [0, builtin.DYNAMIC_INDEX, 0]
        )
        observer.properties["operandSegmentSizes"] = builtin.DenseArrayBase.from_list(builtin.i32, [1, 1, 0, 0])
    module.verify()
    before = text(module, generic=True)
    result, report = pushback_contraction_row_slices(module)
    assert report["rewrites"] == 0
    assert text(result, generic=True) == before


def test_immutable_coefficient_bytes_enable_same_values_without_specialization():
    module = parse_mlir_text(source(runtime_scale=True))
    scales = np.linspace(0.001, 0.031, 16, dtype=np.float32)
    result, report = pushback_contraction_row_slices(
        module, immutable_argument_loader=lambda a: scales.tobytes() if a.index == 2 else None
    )
    assert report["rewrites"] == 1
    assert not any(
        isinstance(op, arith.ConstantOp)
        and isinstance(op.value, builtin.DenseIntOrFPElementsAttr)
        and op.value.type.get_shape() == (16,)
        for op in result.walk()
    )
    args = [np.arange(32, dtype=np.int8).reshape(4, 8), np.arange(128, dtype=np.int8).reshape(8, 16), scales]
    for actual, expected in zip(run(result, args), run(module, args), strict=True):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_existing_registry_and_discovery_expose_explicit_source_api(monkeypatch):
    from merlin.compile.scheduling.api_catalog import scheduling_capabilities
    from merlin.compile.scheduling.library_sources import library_modules
    from merlin.llvmlower.contraction_row_slice import prepare_contraction_row_slices, selected
    from merlin.llvmlower.optional_passes import REGISTRY

    monkeypatch.delenv("MERLIN_CONTRACTION_ROW_SLICE", raising=False)
    monkeypatch.delenv("MERLIN_PASSES", raising=False)
    assert not selected()
    module = parse_mlir_text(source())
    assert prepare_contraction_row_slices(module)[0] is module
    monkeypatch.setenv("MERLIN_PASSES", "contraction-row-slice")
    assert selected() and prepare_contraction_row_slices(module)[1]["rewrites"] == 1
    assert any(p.name == "contraction-row-slice" and not p.on_by_default for p in REGISTRY)
    assert "merlin.llvmlower.contraction_row_slice" in library_modules(24)
    assert any(p["name"] == "sole_observer_contraction_rows" for p in scheduling_capabilities(24)["capabilities"])


def test_zero_budget_retains_exact_source():
    module = parse_mlir_text(source())
    result, report = pushback_contraction_row_slices(module, max_rewrites=0)
    assert report["rewrites"] == 0 and text(result, generic=True) == text(module, generic=True)


def test_fill_seed_and_independent_operand_observer_survive():
    program = source().replace(
        "%init=tensor.splat %z:tensor<4x16xi32>",
        """
      %seed_empty=tensor.empty():tensor<4x16xi32>
      %init=linalg.fill ins(%z:i32) outs(%seed_empty:tensor<4x16xi32>)->tensor<4x16xi32>""",
    )
    module = parse_mlir_text(program)
    forward = module.body.block.first_op
    ret = forward.body.block.last_op
    ret.operands = (*ret.operands, forward.body.block.args[0])
    forward.update_function_type()
    result, report = pushback_contraction_row_slices(module)
    assert report["rewrites"] == 1
    root = next(op for op in result.walk() if isinstance(op, GenericOp))
    assert root.outputs[0].owner.name == "linalg.fill"
    assert root.outputs[0].type.get_shape() == (1, 16)
    args = [np.arange(32, dtype=np.int8).reshape(4, 8), np.arange(128, dtype=np.int8).reshape(8, 16)]
    for actual, expected in zip(run(result, args), run(module, args), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_global_scale_reduction_keeps_all_original_rows():
    program = source().replace(
        "%scale=arith.constant dense<0.03125>:tensor<16xf32>",
        """
      %scale_zero=arith.constant 0.0:f32
      %scale_empty=tensor.empty():tensor<16xf32>
      %scale_init=linalg.fill ins(%scale_zero:f32) outs(%scale_empty:tensor<16xf32>)->tensor<16xf32>
      %scale=linalg.generic {indexing_maps=[affine_map<(n,m)->(m,n)>,affine_map<(n,m)->(n)>],
        iterator_types=["parallel","reduction"]}
        ins(%dot:tensor<4x16xi32>) outs(%scale_init:tensor<16xf32>) {
        ^bb0(%scale_x:i32,%scale_acc:f32):
          %scale_f=arith.sitofp %scale_x:i32 to f32
          %scale_sum=arith.addf %scale_acc,%scale_f:f32
          linalg.yield %scale_sum:f32
      } -> tensor<16xf32>""",
    )
    module = parse_mlir_text(program)
    result, report = pushback_contraction_row_slices(module)
    assert report["rewrites"] == 0 and text(result, generic=True) == text(module, generic=True)


def test_ordinary_preparation_consumes_explicit_registry_choice_before_device_binding(tmp_path, monkeypatch):
    from merlin.runtime.backends.zephyr_model import _prepare_model_mlir

    model = tmp_path / "model.mlir"
    model.write_text(source())
    monkeypatch.setenv("MERLIN_PASSES", "contraction-row-slice")
    selected = _prepare_model_mlir(model, tmp_path)
    module = parse_mlir_text(selected.read_text())
    root = next(op for op in module.walk() if isinstance(op, GenericOp))
    assert root.results[0].type.get_shape() == (1, 16)
    import json

    assert json.loads((tmp_path / "contraction-row-slice.json").read_text())["rewrites"] == 1
