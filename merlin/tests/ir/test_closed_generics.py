"""Closed elementwise generics become exact uniform fills; nothing else is folded."""

from mlir_fixtures import forward, generic, returned

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.closed_scalar import evaluate, fold_closed_generics, fold_unit_scales, interval
from merlin.targetgen.core_aten_numeric import normalize_closed_identities


def program(body, ins="%c"):
    prelude = "%z = arith.constant 0.0 : f32\n%c = tensor.splat %z : tensor<f32>\n"
    closed = generic([(ins, "tensor<f32>")], "tensor<bf16>", body, yields="%n")
    return forward("%x: tensor<f32>", "tensor<bf16>", prelude + closed)


def test_empty_reduction_narrowed_to_storage_is_a_literal_fill():
    module = program("%n = arith.truncf %a : f32 to bf16")
    assert fold_closed_generics(module) == 1
    module.verify()
    assert returned(module).owner.name == "tensor.splat"
    assert not any(op.name == "linalg.generic" for op in module.walk())


def test_runtime_input_is_not_folded():
    module = program("%n = arith.truncf %a : f32 to bf16", ins="%x")
    assert fold_closed_generics(module) == 0


def test_non_finite_closed_result_is_not_folded():
    module = program("%d = arith.divf %a, %a : f32\n%n = arith.truncf %d : f32 to bf16")
    assert fold_closed_generics(module) == 0


def test_unit_scale_is_an_exact_identity():
    for body, folded in (
        ("%k = arith.constant 1.0 : f32\n%m = arith.mulf %k, %a : f32\n%n = arith.truncf %m : f32 to bf16", 1),
        ("%k = arith.constant 1.0 : f32\n%m = arith.divf %a, %k : f32\n%n = arith.truncf %m : f32 to bf16", 1),
        ("%k = arith.constant 1.0 : f32\n%m = arith.divf %k, %a : f32\n%n = arith.truncf %m : f32 to bf16", 0),
        ("%k = arith.constant 2.0 : f32\n%m = arith.mulf %a, %k : f32\n%n = arith.truncf %m : f32 to bf16", 0),
        ("%k = arith.constant -1.0 : f32\n%m = arith.mulf %a, %k : f32\n%n = arith.truncf %m : f32 to bf16", 0),
    ):
        module = program(body, ins="%x")
        assert fold_unit_scales(module) == folded
        module.verify()


def test_round_trip_through_a_copied_unit_scale_is_the_input():
    # Frontend form of f16 x * 1.0: widen (and scale) in one generic, narrow in the next.
    prelude = "%one = arith.constant 1.0 : f32\n%s = tensor.splat %one : tensor<4xf32>\n"
    widen = generic(
        [("%x", "tensor<4xf16>"), ("%s", "tensor<4xf32>")],
        "tensor<4xf32>",
        "%c = arith.extf %a : f16 to f32\n%m = arith.mulf %c, %b : f32",
        yields="%m",
        name="%w",
    )
    narrow = generic(
        [("%w", "tensor<4xf32>")], "tensor<4xf16>", "%t = arith.truncf %a : f32 to f16", yields="%t", name="%n"
    )
    module = forward("%x: tensor<4xf16>", "tensor<4xf16>", f"{prelude}{widen}\n{narrow}", ret="%n")
    normalize_closed_identities(module)
    returned = next(op for op in module.walk() if op.name == "func.return")
    assert returned.operands[0] is next(op for op in module.walk() if op.name == "func.func").body.block.args[0]


def test_integer_conversions_reinterpret_the_source_word():
    # sitofp/extsi read the word as two's complement (i1 true is -1); uitofp/extui read it unsigned.

    module = parse_mlir_text("""module { func.func @f(%b: i1, %w: i8) -> (f32, i32, f32, i32, f32, f32) {
      %s = arith.sitofp %b : i1 to f32
      %e = arith.extsi %b : i1 to i32
      %u = arith.uitofp %b : i1 to f32
      %z = arith.extui %w : i8 to i32
      %n = arith.sitofp %w : i8 to f32
      %t = arith.constant true
      %k = arith.sitofp %t : i1 to f32
      func.return %s, %e, %u, %z, %n, %k : f32, i32, f32, i32, f32, f32
    } }""")
    function = next(op for op in module.walk() if op.name == "func.func")
    b, w = function.body.block.args
    s, e, u, z, n, k = function.body.block.last_op.operands
    assert [evaluate(v, {b: True, w: 255}) for v in (s, e, u, z, n)] == [-1.0, -1, 1.0, 255, -1.0]
    assert [evaluate(v, {b: True, w: -1}) for v in (z, n)] == [255, -1.0]
    assert interval(k) is None  # the literal word 1 is outside the signed i1 range; no unsound bound
