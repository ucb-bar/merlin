"""Shared construction keeps source quantization frontiers for normal routing."""

from test_sibling_components import effects
from test_source_region import sibling_source

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.sibling_components import prepare_sibling_components
from merlin.llvmlower.source_region import _sibling_tensor_operation_is_pure
from merlin.xdsl_dialects._common import text


def quantized_between_roots():
    module = sibling_source()
    forward = module.body.block.first_op
    convert = next(
        o for o in forward.body.block.ops if o.name == "linalg.generic" and str(o.results[0].type).endswith("xf32>")
    )
    template = parse_mlir_text("""module {func.func @q(%x:tensor<2x2xf32>)->tensor<2x2xi8>{
      %s=arith.constant dense<0.25>:tensor<f32>
      %z=arith.constant dense<0>:tensor<i64>
      %q="quant_ext.quantize_per_tensor"(%x,%s,%z)<{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}>:(tensor<2x2xf32>,tensor<f32>,tensor<i64>)->tensor<2x2xi8>
      return %q:tensor<2x2xi8>}}""")
    qbody = template.body.block.first_op.body.block
    value_map = {qbody.args[0]: convert.results[0]}
    anchor = convert
    for op in tuple(qbody.ops)[:-1]:
        cloned = op.clone(value_mapper=value_map)
        forward.body.block.insert_op_after(cloned, anchor)
        value_map.update(zip(op.results, cloned.results, strict=True))
        anchor = cloned
    ret = forward.body.block.last_op
    ret.operands = (anchor.results[0], ret.arguments[1])
    forward.update_function_type()
    module.verify()
    return module, anchor


def test_quantization_purity_shadow_keeps_actual_source_and_use_lists():
    module, op = quantized_between_roots()
    before = text(module, generic=True)
    uses = [tuple(value.uses) for value in op.operands]
    assert _sibling_tensor_operation_is_pure(op)
    assert text(module, generic=True) == before
    assert [tuple(value.uses) for value in op.operands] == uses
    from xdsl.dialects.builtin import StringAttr

    op.attributes["unmodeled_effect"] = StringAttr("unknown")
    assert not _sibling_tensor_operation_is_pure(op)


def test_component_outline_retains_quantized_marker_and_external_result_type():
    module, _ = quantized_between_roots()
    before = text(module, generic=True)
    plan = prepare_sibling_components(module, effects=effects())
    assert len(plan.components) == 1
    markers = [
        op
        for op in plan.module.walk()
        if getattr(getattr(op, "op_name", None), "data", None) == "quant_ext.quantize_per_tensor"
    ]
    assert len(markers) == 1
    assert plan.module.body.block.first_op.function_type == module.body.block.first_op.function_type
    assert text(module, generic=True) == before


def test_shared_source_readout_is_exactly_proved_and_keeps_raw_observer():
    import numpy as np
    from test_output_fusion import _path

    from merlin.llvmlower.passes_xdsl import lower_quant_ext
    from merlin.runtime.linalg_numpy import evaluate

    module, _ = quantized_between_roots()
    plan = prepare_sibling_components(module, effects=effects(), output_path=_path())
    assert len(plan.components) == 1
    descriptors = plan.components[0].receipt["readouts"]
    assert [row["output_dtype"] for row in descriptors] == ["i8", "i32"]
    assert descriptors[0]["output_fusion"]["cut"] == "requant"
    args = [
        np.asarray([[1, -2], [3, 4]], dtype=np.int8),
        np.asarray([[2, 3], [-1, 4]], dtype=np.int8),
        np.asarray([[-3, 2], [1, -1]], dtype=np.int8),
    ]
    source = module.clone()
    lower_quant_ext(source)

    def call(values):
        lhs, w1, w2 = values[:3]
        a = lhs.astype(np.int32) @ w1.astype(np.int32)
        b = lhs.astype(np.int32) @ w2.astype(np.int32)
        return np.clip(np.rint(a.astype(np.float32) * 4), -128, 127).astype(np.int8), b

    actual = evaluate(plan.module, args, calls={plan.components[0].writer.symbol: call})
    expected = evaluate(source, args)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True))


def shared_readout_source(*, raw_observer=False):
    m, k, n = 3, 32, 32
    lhs = f"tensor<{m}x{k}xi8>"
    rhs = f"tensor<{k}x{n}xi8>"
    raw = f"tensor<{m}x{n}xi32>"
    flt = f"tensor<{m}x{n}xf32>"
    lines = [
        f"module {{func.func @forward(%a:{lhs},%w0:{rhs},%w1:{rhs})->(tensor<1x{m}x{n}xi8>,tensor<1x{m}x{n}xi8>"
        + ("," + raw if raw_observer else "")
        + ") {"
    ]
    scales = [[0.125] * 16 + [0.25] * 16, [0.5] * 16 + [0.0625] * 16]
    offsets = [[i % 7 - 3 for i in range(n)], [2 - i % 5 for i in range(n)]]
    for i in range(2):
        lines.extend(
            [
                f"%z{i}=arith.constant 0:i32",
                f"%s{i}=tensor.splat %z{i}:{raw}",
                f"""%r{i}=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
          affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
          iterator_types=["parallel","parallel","reduction"]}}
          ins(%a,%w{i}:{lhs},{rhs}) outs(%s{i}:{raw}) {{
        ^bb0(%x:i8,%y:i8,%s:i32):%xx=arith.extsi %x:i8 to i32
        %yy=arith.extsi %y:i8 to i32
        %p=arith.muli %xx,%yy:i32
        %q=arith.addi %s,%p:i32
        linalg.yield %q:i32}}->{raw}""",
            ]
        )
        body = f"""%fempty{i}=tensor.empty():{flt}
        %f{i}=linalg.generic {{indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>],
          iterator_types=["parallel","parallel"]}} ins(%r{i}:{raw}) outs(%fempty{i}:{flt}) {{
          ^bb0(%x:i32,%o:f32):%v=arith.sitofp %x:i32 to f32
        linalg.yield %v:f32}}->{flt}"""
        lines.append(body)
        for kind, operation, vector, input_name in [
            ("scale", "mulf", scales[i], f"f{i}"),
            ("bias", "addf", [a * b for a, b in zip(scales[i], offsets[i], strict=True)], f"scale{i}"),
        ]:
            vs = ",".join(f"{float(v):.8e}" for v in vector)
            lines.append(f"""%{kind}_v{i}=arith.constant dense<[{vs}]>:tensor<{n}xf32>
            %{kind}_e{i}=tensor.empty():{flt}
            %{kind}{i}=linalg.generic {{indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(n)>,
              affine_map<(m,n)->(m,n)>],iterator_types=["parallel","parallel"]}}
              ins(%{input_name},%{kind}_v{i}:{flt},tensor<{n}xf32>) outs(%{kind}_e{i}:{flt}) {{
              ^bb0(%x:f32,%s:f32,%o:f32):%v=arith.{operation} %x,%s:f32
            linalg.yield %v:f32}}->{flt}""")
        lines.append(f"""%view{i}=tensor.expand_shape %bias{i} [[0,1],[2]] output_shape [1,{m},{n}]
          :{flt} into tensor<1x{m}x{n}xf32>
        %qs{i}=arith.constant dense<1.0>:tensor<f32>
        %qz{i}=arith.constant dense<0>:tensor<i64>
        %out{i}="quant_ext.quantize_per_tensor"(%view{i},%qs{i},%qz{i})<{{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}}>:(tensor<1x{m}x{n}xf32>,tensor<f32>,tensor<i64>)->tensor<1x{m}x{n}xi8>""")
    lines.append(
        "return %out0,%out1"
        + (",%r0" if raw_observer else "")
        + f":tensor<1x{m}x{n}xi8>,tensor<1x{m}x{n}xi8>"
        + ("," + raw if raw_observer else "")
        + "}}"
    )
    return parse_mlir_text("\n".join(lines))


def test_separate_original_bias_scales_and_ranked_views_reach_typed_writer_exactly():
    import numpy as np
    from test_output_fusion import _path

    from merlin.llvmlower.passes_xdsl import lower_quant_ext
    from merlin.runtime.linalg_numpy import evaluate

    source = shared_readout_source()
    plan = prepare_sibling_components(source, effects=effects(), output_path=_path())
    assert len(plan.components) == 1
    component = plan.components[0]
    rows = component.receipt["readouts"]
    assert [r["output_dtype"] for r in rows] == ["i8", "i8"]
    assert [r["bias_argument"] for r in rows] == [3, 4]
    assert [r["attributes"]["acc_scale_by_column"] for r in rows] == [
        [0.125] * 16 + [0.25] * 16,
        [0.5] * 16 + [0.0625] * 16,
    ]
    rng = np.random.default_rng(19)
    inputs = [
        rng.integers(-4, 5, (3, 32), dtype=np.int8),
        *(rng.integers(-3, 4, (32, 32), dtype=np.int8) for _ in range(2)),
    ]
    original = source.clone()
    lower_quant_ext(original)
    expected = evaluate(original, inputs)

    def call(args):
        a, w0, w1, b0, b1 = args[:5]
        result = []
        for w, b, row in zip((w0, w1), (b0, b1), rows, strict=True):
            assert b.dtype == np.int32 and b.shape == (32,)
            raw = a.astype(np.int32) @ w.astype(np.int32) + b
            result.append(
                np.clip(
                    np.rint(
                        raw.astype(np.float32) * np.asarray(row["attributes"]["acc_scale_by_column"], dtype=np.float32)
                    ),
                    -128,
                    127,
                ).astype(np.int8)
            )
        return tuple(result)

    actual = evaluate(plan.module, inputs, calls={component.writer.symbol: call})
    assert all(a.shape == (1, 3, 32) and np.array_equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert component.writer.result_arguments == (5, 6)


def test_raw_observer_prevents_absorption_without_blocking_other_member():
    import numpy as np
    from test_output_fusion import _path

    from merlin.llvmlower.passes_xdsl import lower_quant_ext
    from merlin.runtime.linalg_numpy import evaluate

    source = shared_readout_source(raw_observer=True)
    before = text(source, generic=True)
    plan = prepare_sibling_components(source, effects=effects(), output_path=_path())
    component = plan.components[0]
    rows = component.receipt["readouts"]
    assert [row["output_dtype"] for row in rows] == ["i32", "i8"]
    assert [row["bias_argument"] for row in rows] == [None, 3]
    assert not rows[0]["absorbed"] and rows[1]["absorbed"]
    rng = np.random.default_rng(31)
    inputs = [
        rng.integers(-4, 5, (3, 32), dtype=np.int8),
        *(rng.integers(-3, 4, (32, 32), dtype=np.int8) for _ in range(2)),
    ]
    original = source.clone()
    lower_quant_ext(original)
    expected = evaluate(original, inputs)

    def call(args):
        lhs, first_weight, second_weight, bias = args[:4]
        first = lhs.astype(np.int32) @ first_weight.astype(np.int32)
        second = lhs.astype(np.int32) @ second_weight.astype(np.int32) + bias
        scale = np.asarray(rows[1]["attributes"]["acc_scale_by_column"], dtype=np.float32)
        narrowed = np.clip(np.rint(second.astype(np.float32) * scale), -128, 127).astype(np.int8)
        return first, narrowed

    host = plan.module.clone()
    lower_quant_ext(host)
    actual = evaluate(host, inputs, calls={component.writer.symbol: call})
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert text(source, generic=True) == before


def test_unsupported_scale_granularity_preserves_original_host_chain():
    from test_output_fusion import _path

    source = shared_readout_source()
    before = text(source, generic=True)
    path = _path()
    for stage in path["readout"]:
        if stage["kind"] == "scale":
            stage["granularity"] = "per_tensor"
            stage.pop("columns_per_scale", None)
    plan = prepare_sibling_components(source, effects=effects(), output_path=path)
    assert len(plan.components) == 1
    rows = plan.components[0].receipt["readouts"]
    assert [row["output_dtype"] for row in rows] == ["i32", "i32"]
    assert not any(row["absorbed"] for row in rows)
    markers = [
        op
        for op in plan.module.walk()
        if getattr(getattr(op, "op_name", None), "data", None) == "quant_ext.quantize_per_tensor"
    ]
    assert len(markers) == 2
    assert text(source, generic=True) == before


def test_disjoint_source_components_share_one_current_shape_analysis(monkeypatch):
    from test_output_fusion import _path
    from test_sibling_components import multi_source

    from merlin.compile.analysis import contraction_shapes

    original = contraction_shapes.contraction_extents
    observed = []

    def count(module):
        observed.append(module)
        return original(module)

    monkeypatch.setattr(contraction_shapes, "contraction_extents", count)
    plan = prepare_sibling_components(multi_source(), effects=effects(), output_path=_path())
    assert len(plan.components) == 2
    assert len(observed) == 1
