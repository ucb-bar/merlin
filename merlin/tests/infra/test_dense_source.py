"""Typed source admission for the shared dense compiler; neutral execution only."""

from dataclasses import replace

import pytest
from test_compact_dense import native_dense_pair
from test_selected_dense_contracts import inputs
from xdsl.dialects import arith
from xdsl.dialects.builtin import StringAttr

from merlin.compile.scheduling.dense_source import DenseSourceUnavailable, schedule_source_dense
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.dense_planning import DenseResources


def contraction(shape, dtypes, *, lhs="a", rhs="b", output="r", tag="matmul"):
    m, k, n = shape
    x, y, z = dtypes
    expressions, values = [], []
    for source, dtype, name in (("x", x, "u"), ("y", y, "v")):
        if dtype != z:
            expressions.append(f"%{output}_{name} = arith.extsi %{output}_{source} : {dtype} to {z}")
            values.append(output + "_" + name)
        else:
            values.append(output + "_" + source)
    scalar = "\n".join(expressions)
    return f'''
 %zero_{output} = arith.constant 0 : {z}
 %empty_{output} = tensor.empty() : tensor<{m}x{n}x{z}>
 %fill_{output} = linalg.fill ins(%zero_{output} : {z})
   outs(%empty_{output} : tensor<{m}x{n}x{z}>) -> tensor<{m}x{n}x{z}>
 %{output} = linalg.generic {{indexing_maps = [affine_map<(m,n,k)->(m,k)>,
   affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
   iterator_types = ["parallel", "parallel", "reduction"], prov.region_id = "{tag}"}}
   ins(%{lhs}, %{rhs} : tensor<{m}x{k}x{x}>, tensor<{k}x{n}x{y}>)
   outs(%fill_{output} : tensor<{m}x{n}x{z}>) {{
 ^bb0(%{output}_x: {x}, %{output}_y: {y}, %{output}_s: {z}):
 {scalar}
 %{output}_product = arith.muli %{values[0]}, %{values[1]} : {z}
 %{output}_sum = arith.addi %{output}_s, %{output}_product : {z}
 linalg.yield %{output}_sum : {z}
 }} -> tensor<{m}x{n}x{z}>
'''


def source(shape, dtypes):
    m, k, n = shape
    x, y, z = dtypes
    text = f"""module {{func.func @forward(%a:tensor<{m}x{k}x{x}>, %b:tensor<{k}x{n}x{y}>)
      -> tensor<{m}x{n}x{z}> {{
      {contraction(shape, dtypes)}
      return %r : tensor<{m}x{n}x{z}>
    }}}}"""
    module = parse_mlir_text(text)
    return module, next(op for op in module.walk() if op.name == "linalg.generic")


def compile_source(model, module, operation, **kwargs):
    selected, _, storage, adapter = model
    return schedule_source_dense(
        module,
        operation,
        selected_contracts=selected,
        storage=storage,
        adapter=adapter,
        name="source_dense",
        static_budget=80,
        external_buffers_disjoint=True,
        verification_timeout_ms=20_000,
        **kwargs,
    )


@pytest.mark.parametrize(
    "tile,widths,shape,shared",
    [
        ((2, 3, 4), (1, 1, 4), (5, 14, 13), False),
        ((3, 2, 5), (2, 1, 4), (7, 9, 16), True),
        ((2, 3, 4), (2, 2, 4), (1, 7, 9), False),
        ((2, 3, 4), (4, 4, 4), (3, 8, 5), False),
    ],
)
def test_source_dimensions_formats_and_recurrence_drive_real_loop_emission(
    tmp_path, tile, widths, shape, shared, record_property
):
    import json

    model = inputs(tmp_path, tile=tile, widths=widths, shape=shape, shared=shared)
    selected, problem, storage, adapter = model
    module, op = source(shape, selected.dense.dtypes)
    result = compile_source(model, module, op)
    assert result is not None
    assert [step.name for step in result.source_passes] == ["source-admission", "dense-lowering"]
    assert result.compiled.realization["pipeline"]["context"]["hardware"] == selected.hardware_identity
    controls = DenseResources(1 << 20, 1 << 20, 3, 3, 2, partial_output_tiles=True, partial_output_blocks=True)
    pair = native_dense_pair(
        tmp_path,
        problem,
        controls,
        storage,
        result.compiled,
        adapter,
        dtypes=selected.dense.dtypes,
        selected_contracts=selected,
        overflow=widths[0] >= 2 and widths[1] >= 2,
        whole_graph=True,
    )
    record_property("source_dense_pair", json.dumps(pair))
    result.revalidate()


@pytest.mark.parametrize(
    "mutation", ["unsigned", "multiply", "accumulator", "zero", "flags", "attributes", "empty_attributes", "body"]
)
def test_source_mutations_cannot_be_admitted_from_shape_alone(tmp_path, mutation):
    model = inputs(tmp_path)
    module, op = source((5, 14, 13), model[0].dense.dtypes)
    if mutation == "unsigned":
        # Use canonical generic serialization so the replacement is exact.
        from merlin.transforms.certificates import ir_text

        text = ir_text(module).replace('"arith.extsi"', '"arith.extui"')
        module = parse_mlir_text(text)
        op = next(o for o in module.walk() if o.name == "linalg.generic")
    elif mutation == "zero":
        constant = next(o for o in module.walk() if o.name == "arith.constant")
        from xdsl.dialects.builtin import IntegerAttr, i32

        constant.properties["value"] = IntegerAttr(1, i32)
    elif mutation == "flags":
        arithmetic = next(o for o in op.walk() if o.name == "arith.addi")
        arithmetic.properties["overflowFlags"] = arith.IntegerOverflowAttr([arith.IntegerOverflowFlag.NSW])
    elif mutation == "attributes":
        op.attributes["unknown_effect"] = StringAttr("opaque")
    elif mutation == "empty_attributes":
        next(o for o in module.walk() if o.name == "tensor.empty").attributes["unknown_layout"] = StringAttr("opaque")
    elif mutation == "multiply":
        arithmetic = next(o for o in op.walk() if o.name == "arith.muli")
        arithmetic.operands = (arithmetic.operands[0], arithmetic.operands[0])
    elif mutation == "accumulator":
        arithmetic = next(o for o in op.walk() if o.name == "arith.addi")
        arithmetic.operands = (arithmetic.operands[1], arithmetic.operands[1])
    else:
        fill = next(o for o in module.walk() if o.name == "linalg.fill")
        fill.regions[0].block.last_op.operands = (fill.regions[0].block.args[1],)
    with pytest.raises(DenseSourceUnavailable, match="body|zero|unknown|unsupported|fresh tensor"):
        compile_source(model, module, op)


@pytest.mark.parametrize("mutation", ["source", "facts", "software", "kernel"])
def test_source_and_realized_kernel_custody_stays_current_through_emission(tmp_path, mutation):
    from merlin.sched.ir import Kernel

    model = inputs(tmp_path)
    module, op = source((5, 14, 13), model[0].dense.dtypes)
    result = compile_source(model, module, op)
    if mutation == "source":
        next(o for o in module.walk() if o.name == "arith.constant").properties["value"] = (
            arith.ConstantOp.from_int_and_width(1, 32).value
        )
    elif mutation in ("facts", "software"):
        path = model[0].facts_path if mutation == "facts" else model[0].software_path
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        object.__setattr__(result.compiled, "kernel", Kernel("changed", (), ()))
    with pytest.raises(ValueError, match="changed|replan"):
        result.emit_c(None, symbol="refused_before_emitter")


def test_explicit_source_order_ownership_budget_and_unsupported_capability(tmp_path):
    model = inputs(tmp_path)
    module, op = source((5, 14, 13), model[0].dense.dtypes)
    with pytest.raises(ValueError, match="ordering|prerequisites"):
        compile_source(model, module, op, pass_order=("dense-lowering", "source-admission"))
    with pytest.raises(DenseSourceUnavailable, match="element budget"):
        compile_source(model, module, op, max_source_elements=10)
    selected, _, storage, adapter = model
    with pytest.raises(ValueError, match="possibly aliased"):
        schedule_source_dense(
            module,
            op,
            selected_contracts=selected,
            storage=storage,
            adapter=adapter,
            name="bad",
            static_budget=80,
            external_buffers_disjoint=False,
        )
    with pytest.raises(ValueError, match="qualification"):
        schedule_source_dense(
            module,
            op,
            selected_contracts=selected,
            storage=storage,
            adapter=replace(adapter, qualification_identity=None),
            name="bad",
            static_budget=80,
            external_buffers_disjoint=True,
        )


def test_loop_return_allocations_require_the_existing_owned_deallocation_pipeline(monkeypatch):
    from merlin.llvmlower import pipeline

    monkeypatch.delenv("MERLIN_NO_DEALLOC", raising=False)
    ordinary = pipeline._upstream_pipeline()
    assert "allow-return-allocs-from-loops" in ordinary
    assert ordinary.index("one-shot-bufferize") < ordinary.index("ownership-based-buffer-deallocation")
    monkeypatch.setenv("MERLIN_NO_DEALLOC", "1")
    assert "allow-return-allocs-from-loops" not in pipeline._upstream_pipeline()


@pytest.mark.parametrize(
    "token,expected",
    [("int16", "i16"), ("i16", "i16"), ("int64", "i64"), ("i0", None), ("int-16", None), ("unknown", None)],
)
def test_declared_machine_integer_spelling_does_not_depend_on_a_corpus(token, expected):
    from merlin.system.offload import _mlir

    assert _mlir(token) == expected


def test_dense_shim_and_runtime_preserve_standard_signed_integer_widths():
    import numpy as np

    from merlin.llvmlower import c_runtime, device_shim

    assert device_shim._elem_bytes("i16") == 2
    assert device_shim._elem_bytes("int64") == 8
    assert device_shim._elem_bytes("i24") is None
    assert c_runtime.C_OF["i16"] == "int16_t"
    assert c_runtime._embed_array(np.array([-32768, -1, 0, 32767], np.int16), "i16") == "-32768,-1,0,32767"


@pytest.mark.parametrize("mutation", ["operation", "compiled"])
def test_another_valid_operation_or_lowering_cannot_replace_the_original_binding(tmp_path, mutation):
    model = inputs(tmp_path)
    module, first = source((5, 14, 13), model[0].dense.dtypes)
    other, second = source((3, 14, 13), model[0].dense.dtypes)
    function = other.body.block.first_op
    function.properties["sym_name"] = StringAttr("other")
    function.detach()
    module.body.block.add_op(function)
    first_result = compile_source(model, module, first)
    second_result = compile_source(model, module, second)
    object.__setattr__(
        first_result,
        "operation" if mutation == "operation" else "compiled",
        second if mutation == "operation" else second_result.compiled,
    )
    with pytest.raises(ValueError, match="changed; replan"):
        first_result.revalidate()


def test_falsely_declared_continuation_cannot_qualify_from_correct_command_counts(tmp_path):
    import numpy as np
    from test_dense_source_catalog import Catalog, implementation, qualify_native

    model = inputs(tmp_path)
    selected, _, storage, _ = model
    provider = Catalog(model)
    for i, shape in enumerate(((5, 14, 13), (5, 13, 14))):
        module, op = source(shape, selected.dense.dtypes)
        result = compile_source(model, module, op)
        p = selected.dense.problem(*shape)
        path = tmp_path / f"faulty_{i}.c"
        symbol = f"faulty_{i}"
        path.write_text(implementation(result, p, storage, selected.dense.dtypes, symbol, lose_continuation=True))
        provider.results.append((result, p, path, symbol))
    rng = np.random.default_rng(2)
    w0 = rng.integers(-128, 128, (14, 13), dtype=np.int8)
    w1 = rng.integers(-128, 128, (13, 14), dtype=np.int8)
    data = rng.integers(-128, 128, (5, 14), dtype=np.int8)
    state = np.zeros_like(data)
    with pytest.raises(AssertionError):
        qualify_native(provider, w0, w1, data, state, 7, tmp_path)
