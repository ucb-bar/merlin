"""Source boundaries preserve integer reductions, BF16 rounds and live uses."""

import pytest
from xdsl.dialects import func
from xdsl.dialects.builtin import UnitAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.observation_boundary import (
    analyze_observation_boundary,
    validate_observation_boundary,
)


def fixture(m=2, k=3, n=5, *, escape=False, opaque=False, uninitialized=False):
    a, b, c, f, s = (f"tensor<{m}x{k}xi8>", f"tensor<{k}x{n}xi8>",
                      f"tensor<{m}x{n}xi32>", f"tensor<{m}x{n}xbf16>", f"tensor<{m}xbf16>")
    seed = f"%seed = tensor.empty() : {c}" if uninitialized else f"%seed = tensor.splat %zero : {c}"
    tail = f"%last = func.call @opaque(%scaled) : ({f}) -> {f}" if opaque else ""
    output = "%last" if opaque else "%scaled"
    extra, extra_type = (", %scale", ", " + s) if escape else ("", "")
    module = parse_mlir_text(f'''module {{
      func.func private @opaque({f}) -> {f}
      func.func @test(%a: {a}, %b: {b}, %scale: {s}) -> ({f}{extra_type}) {{
        %zero = arith.constant 0 : i32
        {seed}
        %dot = linalg.generic {{indexing_maps = [affine_map<(i,j,z)->(i,z)>, affine_map<(i,j,z)->(z,j)>, affine_map<(i,j,z)->(i,j)>], iterator_types = ["parallel", "parallel", "reduction"]}} ins(%a,%b : {a},{b}) outs(%seed : {c}) {{
          ^bb0(%x: i8,%y: i8,%acc: i32):
            %xx = arith.extsi %x : i8 to i32
            %yy = arith.extsi %y : i8 to i32
            %p = arith.muli %xx,%yy : i32
            %sum = arith.addi %acc,%p : i32
            linalg.yield %sum : i32
        }} -> {c}
        %empty = tensor.empty() : {f}
        %cast = linalg.generic {{indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]}} ins(%dot : {c}) outs(%empty : {f}) {{
          ^bb0(%v: i32,%unused: bf16):
            %v2 = arith.sitofp %v : i32 to bf16
            linalg.yield %v2 : bf16
        }} -> {f}
        %empty2 = tensor.empty() : {f}
        %scaled = linalg.generic {{indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]}} ins(%cast,%scale : {f},{s}) outs(%empty2 : {f}) {{
          ^bb0(%v: bf16,%s: bf16,%unused: bf16):
            %v2 = arith.mulf %v,%s : bf16
            linalg.yield %v2 : bf16
        }} -> {f}
        {tail}
        func.return {output}{extra} : {f}{extra_type}
      }}
    }}''')
    function = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp) and o.sym_name.data == "test")
    return module, function, tuple(function.body.block.args), tuple(function.body.block.last_op.operands)


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 7, 5), (9, 2, 11)])
def test_integer_reduction_to_bf16_is_retained_without_rewriting(shape):
    module, _, inputs, outputs = fixture(*shape)
    before = str(module)
    boundary = analyze_observation_boundary(inputs=inputs, outputs=outputs)
    assert boundary.requested_boundary_closed
    assert boundary.observation_outputs == outputs
    names = [child.name for op in boundary.operations for child in op.walk()]
    assert names.index("arith.muli") < names.index("arith.addi") < names.index("arith.sitofp") < names.index("arith.mulf")
    assert str(module) == before
    validate_observation_boundary(boundary)


def test_unselected_scale_escape_remains_mandatory():
    _, _, inputs, outputs = fixture(escape=True)
    boundary = analyze_observation_boundary(inputs=inputs, outputs=outputs[:1])
    assert not boundary.requested_boundary_closed
    assert boundary.external_escapes == (inputs[2],)
    assert boundary.observation_outputs == outputs


@pytest.mark.parametrize("kind", ["scalar", "context", "use", "order"])
def test_source_mutations_refuse(kind):
    _, function, inputs, outputs = fixture()
    boundary = analyze_observation_boundary(inputs=inputs, outputs=outputs)
    if kind == "scalar":
        next(c for op in boundary.operations for c in op.walk() if c.name == "arith.addi").attributes["changed"] = UnitAttr()
    elif kind == "context":
        function.attributes["strictfp"] = UnitAttr()
    elif kind == "use":
        extra = func.CallOp("opaque", [outputs[0]], [outputs[0].type])
        function.body.block.insert_op_before(extra, function.body.block.last_op)
    else:
        operation = boundary.operations[0]
        block = function.body.block
        block.detach_op(operation)
        block.insert_op_before(operation, block.last_op)
    with pytest.raises(ValueError, match="changed after analysis"):
        validate_observation_boundary(boundary)


def test_unknown_effect_refuses():
    _, _, inputs, outputs = fixture(opaque=True)
    with pytest.raises(ValueError, match="unsupported"):
        analyze_observation_boundary(inputs=inputs, outputs=outputs)


def test_strict_context_refuses_initial_analysis():
    _, function, inputs, outputs = fixture()
    function.attributes["strictfp"] = UnitAttr()
    with pytest.raises(ValueError, match="strict FP"):
        analyze_observation_boundary(inputs=inputs, outputs=outputs)


def test_read_uninitialized_reduction_destination_refuses():
    _, _, inputs, outputs = fixture(uninitialized=True)
    with pytest.raises(ValueError, match="unsupported"):
        analyze_observation_boundary(inputs=inputs, outputs=outputs)


def test_missing_argument_and_duplicate_boundary_refuse():
    _, _, inputs, outputs = fixture()
    with pytest.raises(ValueError, match="unbound"):
        analyze_observation_boundary(inputs=inputs[:2], outputs=outputs)
    with pytest.raises(ValueError, match="distinct"):
        analyze_observation_boundary(inputs=(*inputs, inputs[0]), outputs=outputs)


def test_static_shape_required():
    _, _, inputs, outputs = fixture(m=0)
    with pytest.raises(ValueError, match="static"):
        analyze_observation_boundary(inputs=inputs, outputs=outputs)


def test_intermediate_external_use_is_an_observation():
    _, function, inputs, outputs = fixture()
    cast = next(o for o in function.body.block.ops if o.results and o.results[0].name_hint == "cast")
    extra = func.CallOp("opaque", [cast.results[0]], [cast.results[0].type])
    function.body.block.insert_op_before(extra, function.body.block.last_op)
    boundary = analyze_observation_boundary(inputs=inputs, outputs=outputs)
    assert boundary.external_escapes == (cast.results[0],)
    assert not boundary.requested_boundary_closed


def test_fastmath_scalar_refuses():
    _, function, inputs, outputs = fixture()
    scalar = next(o for o in function.walk() if o.name == "arith.mulf")
    scalar.attributes["fastmath"] = UnitAttr()
    with pytest.raises(ValueError, match="unsupported"):
        analyze_observation_boundary(inputs=inputs, outputs=outputs)


def test_unrelated_or_cross_block_inputs_refuse():
    _, _, inputs, outputs = fixture()
    _, _, other, other_outputs = fixture()
    with pytest.raises(ValueError, match="same block"):
        analyze_observation_boundary(inputs=(*inputs, other[0]), outputs=outputs)
    with pytest.raises(ValueError, match="share a function block"):
        analyze_observation_boundary(inputs=inputs, outputs=(*outputs, *other_outputs))


def test_unchanged_parameter_escape_is_not_a_replaced_source_escape():
    _, _, inputs, outputs = fixture(escape=True)
    boundary = analyze_observation_boundary(inputs=inputs, outputs=outputs[:1], source_values=inputs[:1])
    assert boundary.requested_boundary_closed
    assert boundary.source_values == inputs[:1]
    with pytest.raises(ValueError, match="source values"):
        analyze_observation_boundary(inputs=inputs, outputs=outputs, source_values=())


def test_modified_observation_record_cannot_discard_live_scale():
    from dataclasses import replace

    _, _, inputs, outputs = fixture(escape=True)
    boundary = analyze_observation_boundary(inputs=inputs, outputs=outputs[:1])
    with pytest.raises(ValueError, match="contents changed"):
        validate_observation_boundary(replace(boundary, external_escapes=()))
    with pytest.raises(ValueError, match="source changed"):
        validate_observation_boundary(replace(boundary, source_values=inputs[:1]))
