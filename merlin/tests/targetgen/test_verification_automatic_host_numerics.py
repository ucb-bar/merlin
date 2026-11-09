from __future__ import annotations

from verification_witnesses.automatic_host_numerics import (
    ArithFunctions,
    Interpreter,
    InterpreterFunctions,
    LinalgFunctions,
    _scalar_outputs,
    _ScalarConversions,
    _source,
    arith,
    builtin,
    cf,
    func,
    impl,
    ir_text,
    math,
    np,
    optimize_host_quantization,
    parse_mlir_text,
    prepare_automatic_host,
    pytest,
    register_impls,
    verify_quantization_rewrite,
)


@pytest.mark.parametrize("width", [8])
def test_realized_arithmetic_executes_ties_tails_and_saturation(width):
    source = parse_mlir_text(_source(width))
    original = ir_text(source)
    (rewritten, report) = optimize_host_quantization(source, numerical_policy="frozen-value-policy")
    values = np.array(
        [-np.inf, -10000, -127.5, -126.5, -2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, 126.5, 127.5, np.inf],
        dtype=np.float32,
    )
    expected = np.clip(np.rint(values), -127, 127).astype(np.int32)
    np.testing.assert_array_equal(_scalar_outputs(source, values), expected)
    np.testing.assert_array_equal(_scalar_outputs(rewritten, values), expected)
    assert ir_text(source) == original
    assert report["rewrites"] == 1 and report["round_sites_after"] == 0
    assert report["checks"][-1]["name"] == "arithmetic_conformance"


@pytest.mark.parametrize("fault", ["tie", "effect"])
def test_checker_reconstructs_actual_output_and_rejects_corruption(fault):
    source = parse_mlir_text(_source())
    (rewritten, _) = optimize_host_quantization(source, numerical_policy="frozen-value-policy")
    if fault == "tie":
        op = next((op for op in rewritten.walk() if op.name == "arith.cmpf" and op.predicate.value.data == 1))
        op.properties["predicate"] = builtin.IntegerAttr(3, builtin.i64)
    elif fault == "clamp":
        op = next((op for op in rewritten.walk() if op.name == "arith.minimumf"))
        op.operands = (op.operands[0], op.operands[0])
    elif fault == "sign":
        op = next((op for op in rewritten.walk() if op.name == "arith.select"))
        op.operands = (op.operands[0], op.operands[2], op.operands[1])
    elif fault == "output":
        op = next((op for op in rewritten.walk() if op.name == "linalg.yield"))
        op.operands = (op.parent.args[1],)
    else:
        op = next((op for op in rewritten.walk() if op.name == "linalg.yield"))
        op.parent.insert_op_before(builtin.UnregisteredOp.with_name("opaque.observe").create(), op)
    with pytest.raises(ValueError, match="arithmetic|unmodeled"):
        verify_quantization_rewrite(source, rewritten)
