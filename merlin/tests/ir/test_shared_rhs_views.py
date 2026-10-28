"""Actual repeated input coordinates feed the existing exact row-batching pass."""

import numpy as np
import pytest
from xdsl.dialects import builtin, func, tensor
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.repeated_pointwise import hoist_repeated_pointwise
from merlin.transforms.certificates import ir_digest, ir_text
from merlin.transforms.dense_row_batching import DenseRowBatchingBudgets, plan_source_dense_row_batching
from merlin.transforms.pipeline_state import PipelineContext
from merlin.transforms.shared_rhs_views import plan_shared_rhs_row_views


def source(g=2, r=3, m=2, k=5, n=7, *, transpose=False, scale="literal", differing=False, planes=1):
    h = g * r
    before_k, before_n = (n, k) if transpose else (k, n)
    scale_type = f"tensor<{h}x1x1xf32>" if scale != "scalar" else "tensor<f32>"
    arguments = [f"%a{p}:tensor<{h}x{m}x{k}xi8>" for p in range(planes)]
    arguments += [f"%input:tensor<{g}x1x{before_k}x{before_n}xf32>"]
    if scale != "literal":
        arguments += [f"%scale:{scale_type}"]
    returns = [f"tensor<{m}x{n}xi32>"] * (h * planes) + [f"tensor<{h}x{k}x{n}xi8>"]
    lines = [f"module {{ func.func @forward({','.join(arguments)})->({','.join(returns)}) {{"]
    if scale == "literal":
        values = [2 + i // r for i in range(h)]
        if differing:
            values[1] += 1
        data = ",".join(f"[[{float(v)}]]" for v in values)
        lines.append(f"%scale = arith.constant dense<[{data}]>:{scale_type}")
    lines += [
        f"""
 %empty=tensor.empty():tensor<{g}x{r}x{before_k}x{before_n}xf32>
 %repeat=linalg.generic {{indexing_maps=[affine_map<(g,r,k,n)->(g,0,k,n)>,
 affine_map<(g,r,k,n)->(g,r,k,n)>],iterator_types=["parallel","parallel","parallel","parallel"]}}
 ins(%input:tensor<{g}x1x{before_k}x{before_n}xf32>)
 outs(%empty:tensor<{g}x{r}x{before_k}x{before_n}xf32>) {{
 ^bb0(%x:f32,%unused:f32):linalg.yield %x:f32
 }} -> tensor<{g}x{r}x{before_k}x{before_n}xf32>
 %view=tensor.collapse_shape %repeat [[0,1],[2],[3]]
 :tensor<{g}x{r}x{before_k}x{before_n}xf32> into tensor<{h}x{before_k}x{before_n}xf32>
 """
    ]
    quant_input = "%view"
    if transpose:
        lines += [
            f"""
 %ti=tensor.empty():tensor<{h}x{k}x{n}xf32>
 %tv=linalg.transpose ins(%view:tensor<{h}x{n}x{k}xf32>) outs(%ti:tensor<{h}x{k}x{n}xf32>)
 permutation=[0,2,1]
 """
        ]
        quant_input = "%tv"
    scale_map = "(h,0,0)" if scale != "scalar" else "()"
    lines += [
        f"""
 %qi=tensor.empty():tensor<{h}x{k}x{n}xi8>
 %rhs_dense=linalg.generic {{indexing_maps=[affine_map<(h,k,n)->(h,k,n)>,affine_map<(h,k,n)->{scale_map}>,
 affine_map<(h,k,n)->(h,k,n)>],iterator_types=["parallel","parallel","parallel"]}}
 ins({quant_input},%scale:tensor<{h}x{k}x{n}xf32>,{scale_type}) outs(%qi:tensor<{h}x{k}x{n}xi8>) {{
 ^bb0(%x:f32,%s:f32,%unused:i8):
 %d=arith.divf %x,%s:f32
 %rounded=math.roundeven %d:f32
 %lo=arith.constant -127:i64
 %flo=arith.sitofp %lo:i64 to f32
 %below=arith.cmpf olt,%rounded,%flo:f32
 %lower=arith.select %below,%flo,%rounded:f32
 %hi=arith.constant 127:i64
 %fhi=arith.sitofp %hi:i64 to f32
 %above=arith.cmpf ogt,%lower,%fhi:f32
 %bounded=arith.select %above,%fhi,%lower:f32
 %y=arith.fptosi %bounded:f32 to i8
 linalg.yield %y:i8
 }} -> tensor<{h}x{k}x{n}xi8>
 """
    ]
    outputs = []
    for p in range(planes):
        for head in range(h):
            name = f"p{p}_{head}"
            for role, root, rows, cols in (("a", f"a{p}", m, k), ("b", "rhs_dense", k, n)):
                lines += [
                    f"""
 %{name}{role}s=tensor.extract_slice %{root}[{head},0,0][1,{rows},{cols}][1,1,1]
 :tensor<{h}x{rows}x{cols}xi8> to tensor<1x{rows}x{cols}xi8>
 %{name}{role}=tensor.collapse_shape %{name}{role}s [[0,1],[2]]
 :tensor<1x{rows}x{cols}xi8> into tensor<{rows}x{cols}xi8>
 """
                ]
            lines += [
                f"""
 %{name}zero=arith.constant 0:i32
 %{name}init=tensor.splat %{name}zero:tensor<{m}x{n}xi32>
 %{name}=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
 affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]}}
 ins(%{name}a,%{name}b:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%{name}init:tensor<{m}x{n}xi32>) {{
 ^bb0(%a:i8,%b:i8,%c:i32):
 %aa=arith.extsi %a:i8 to i32
 %bb=arith.extsi %b:i8 to i32
 %product=arith.muli %aa,%bb:i32
 %sum=arith.addi %product,%c:i32
 linalg.yield %sum:i32
 }} -> tensor<{m}x{n}xi32>
 """
            ]
            outputs.append(f"%{name}")
    lines += [f"return {','.join([*outputs, '%rhs_dense'])}:{','.join(returns)}", "} }"]
    return parse_mlir_text("\n".join(lines))


def prepare(module):
    hoisted, report = hoist_repeated_pointwise(module)
    selected = plan_shared_rhs_row_views(hoisted)
    assert selected is not None
    batched = plan_source_dense_row_batching(
        selected.candidate, PipelineContext(numerical_policy="ordered-source-integer"), DenseRowBatchingBudgets()
    )
    assert batched is not None
    return hoisted, report, selected, batched


@pytest.mark.parametrize("shape", [(2, 3, 2, 5, 7), (3, 2, 1, 7, 3), (1, 5, 3, 2, 9)])
@pytest.mark.parametrize("transpose", [False, True])
def test_hoisted_exact_quantizer_and_observed_dense_tensor_reach_existing_row_batching(shape, transpose):
    original = source(*shape, transpose=transpose)
    before = ir_digest(original)
    hoisted, report, selected, batched = prepare(original)
    g, r, m, k, n = shape
    assert report["rewrites"] == 1
    assert ir_digest(original) == before
    assert plan_source_dense_row_batching(hoisted, PipelineContext(), DenseRowBatchingBudgets()) is None
    assert selected.record()["groups"] == [
        {
            "original_products": g * r,
            "shared_rhs_groups": g,
            "members_per_group": r,
            "row_problem": [r * m, k, n],
            "outputs": g * r,
        }
    ]
    assert len(batched.record()["batches"]) == g
    assert {tuple(row["shape"]) for row in batched.record()["batches"]} == {(r * m, k, n)}
    # The fully materialized original RHS remains observable; only contraction
    # inputs use the checked shared views. Every result and ABI type survives.
    assert original.body.block.first_op.function_type == batched.candidate.body.block.first_op.function_type
    selected.revalidate()
    batched.revalidate()


def test_runtime_scalar_omitting_repeat_axis_and_all_independent_planes_survive():
    _, report, selected, batched = prepare(source(scale="scalar", planes=2))
    assert report["constant_bytes"] == 0
    assert len(selected.record()["groups"]) == 2
    assert len(batched.record()["batches"]) == 4
    assert sum(row["members"] for row in batched.record()["batches"]) == 12


@pytest.mark.parametrize("kwargs", [{"scale": "runtime"}, {"differing": True}])
def test_equal_shape_or_differing_literals_do_not_authorize_sharing(kwargs):
    original = source(**kwargs)
    hoisted, report = hoist_repeated_pointwise(original)
    assert report["rewrites"] == 0
    assert plan_shared_rhs_row_views(hoisted) is None


@pytest.mark.parametrize("kwargs", [{"max_heads": 2}, {"max_operations": 2}, {"max_views": 1}])
def test_bounded_refusal_does_not_mutate_source_or_ssa_uses(kwargs):
    hoisted, _ = hoist_repeated_pointwise(source())
    before = ir_digest(hoisted)
    values = tuple(value for op in hoisted.walk() for value in op.results)
    uses = tuple(frozenset(value.uses) for value in values)
    assert plan_shared_rhs_row_views(hoisted, **kwargs) is None
    assert ir_digest(hoisted) == before
    assert tuple(frozenset(value.uses) for value in values) == uses


def test_fresh_source_and_candidate_identity_checks_refuse_mutations():
    from dataclasses import replace

    _, _, selected, _ = prepare(source())
    with pytest.raises(ValueError, match="SSA changed"):
        replace(selected, candidate=selected.candidate.clone()).revalidate()
    actual = next(op for op in selected.candidate.walk() if type(op) is tensor.ExtractSliceOp)
    actual.attributes["new_observer"] = builtin.StringAttr("unreviewed")
    with pytest.raises(ValueError, match="SSA changed"):
        selected.revalidate()

    hoisted, _ = hoist_repeated_pointwise(source())
    selected = plan_shared_rhs_row_views(hoisted)
    hoisted.attributes["changed_input_contract"] = builtin.StringAttr("new")
    with pytest.raises(ValueError, match="SSA changed"):
        selected.revalidate()


def test_original_scalar_programs_and_fresh_result_records_are_not_rewritten():
    original = source()
    hoisted, _ = hoist_repeated_pointwise(original)
    selected = plan_shared_rhs_row_views(hoisted)
    before = [op for op in hoisted.walk() if type(op) is GenericOp]
    after = [op for op in selected.candidate.walk() if type(op) is GenericOp]
    assert all(
        a.body.is_structurally_equivalent(b.body) and a.attributes == b.attributes and a.properties == b.properties
        for a, b in zip(before, after, strict=True)
    )
    record = selected.record()
    record["groups"][0]["row_problem"][0] = 0
    assert selected.record()["groups"][0]["row_problem"][0] == 6


def test_identity_broadcast_followed_by_actual_integer_transpose_is_read_from_source():
    hoisted, _ = hoist_repeated_pointwise(source(k=5, n=5))
    function = hoisted.body.block.first_op
    rhs = function.body.block.last_op.arguments[-1]
    first_slice = next(op for op in function.body.block.ops if type(op) is tensor.ExtractSliceOp)
    empty = tensor.EmptyOp([], rhs.type)
    transpose = TransposeOp(rhs, empty.tensor, builtin.DenseArrayBase.from_list(builtin.i64, (0, 2, 1)))
    function.body.block.insert_ops_before((empty, transpose), first_slice)
    rhs.replace_uses_with_if(transpose.results[0], lambda use: use.operation is not transpose)
    selected = plan_shared_rhs_row_views(hoisted)
    assert selected is not None
    selected.revalidate()
    assert len([op for op in selected.candidate.walk() if type(op) is TransposeOp]) == 2


def test_unknown_intervening_effect_preserves_complete_original_family():
    hoisted, _ = hoist_repeated_pointwise(source())
    dots = [
        op
        for op in hoisted.walk()
        if type(op) is GenericOp
        and len(op.iterator_types) == 3
        and any(it.data.value == "reduction" for it in op.iterator_types)
    ]
    declaration = func.FuncOp.external("observe", [], [])
    hoisted.body.block.add_op(declaration)
    dots[1].parent.insert_op_before(func.CallOp("observe", [], []), dots[1])
    before = ir_digest(hoisted)
    assert plan_shared_rhs_row_views(hoisted) is None
    assert ir_digest(hoisted) == before


def test_actual_view_checker_refuses_wrong_emitted_coordinates_before_publishing(monkeypatch):
    from merlin.transforms import shared_rhs_views

    original = shared_rhs_views._emit

    def changed(*args):
        inserted, edits, records = original(*args)
        edit = edits[0][0].inputs[0].owner
        edit.properties["static_offsets"] = builtin.DenseArrayBase.from_list(builtin.i64, (1, 0))
        return inserted, edits, records

    monkeypatch.setattr(shared_rhs_views, "_emit", changed)
    hoisted, _ = hoist_repeated_pointwise(source())
    before = ir_digest(hoisted)
    with pytest.raises(ValueError, match="element coordinates"):
        plan_shared_rhs_row_views(hoisted)
    assert ir_digest(hoisted) == before


def test_actual_emitted_native_host_preserves_quantizer_ties_tails_and_all_outputs(tmp_path):
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    g, r, m, k, n = 2, 3, 1, 5, 7
    original = source(g, r, m, k, n, transpose=True)
    _, _, selected, batched = prepare(original)
    rng = np.random.default_rng(1901)
    lhs = rng.integers(-32, 32, (g * r, m, k), dtype=np.int8)
    rhs = (np.arange(g * n * k, dtype=np.float32).reshape(g, 1, n, k) - 37) * np.float32(1.5)
    repeated = np.repeat(rhs, r, axis=1).reshape(g * r, n, k).transpose(0, 2, 1)
    scales = np.repeat(np.arange(2, 2 + g, dtype=np.float32), r).reshape(g * r, 1, 1)
    quantized = np.clip(np.rint(repeated / scales), -127, 127).astype(np.int8)
    expected = [*(lhs.astype(np.int32) @ quantized.astype(np.int32)), quantized]
    for name, module in (("original", original), ("views", selected.candidate), ("batched", batched.candidate)):
        result = lower_model(ir_text(module), tmp_path / name)
        invoke = HostModel.load(
            str(result.host_so), image_policy=PrivateHostImagePolicy(result.host_so.parent.resolve())
        )
        buffers = [np.full(value.size + 32, 89, dtype=value.dtype) for value in expected]
        outputs = [buffer[16:-16].reshape(value.shape) for buffer, value in zip(buffers, expected, strict=True)]
        invoke([(array.ctypes.data, array.shape) for array in (lhs, rhs, *outputs)])
        for output, reference, buffer in zip(outputs, expected, buffers, strict=True):
            np.testing.assert_array_equal(output, reference)
            assert (buffer[:16] == 89).all() and (buffer[-16:] == 89).all()
