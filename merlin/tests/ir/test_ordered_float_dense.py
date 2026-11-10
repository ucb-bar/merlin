"""Exact source correspondence for explicitly selected floating contractions."""

from dataclasses import replace

import pytest

from merlin.transforms.ordered_float_dense import OrderedFloatPolicy, read_ordered_float_dense

POLICY = OrderedFloatPolicy("f8E4M3FN", "bf16", "test-ordered-fma-v1", "test-finite-normal-or-zero")


def source(m=32, k=64, n=96, *, transposed=True, policy=POLICY):
    t, acc, ident, domain = (policy.input_type, policy.accumulator_type, policy.identity, policy.input_domain)
    rhs = f"{n}x{k}" if transposed else f"{k}x{n}"
    maps = "d1,d2" if transposed else "d2,d1"
    return f'''module {{
  func.func @dense(%a: tensor<{m}x{k}x{t}> {{merlin.input_domain = "{domain}"}},
                   %b: tensor<{rhs}x{t}> {{merlin.input_domain = "{domain}"}}) -> tensor<{m}x{n}x{acc}> {{
    %zero = arith.constant 0.000000e+00 : {acc}
    %empty = tensor.empty() : tensor<{m}x{n}x{acc}>
    %init = linalg.fill ins(%zero : {acc}) outs(%empty : tensor<{m}x{n}x{acc}>) -> tensor<{m}x{n}x{acc}>
    %out = linalg.generic {{indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>,
      affine_map<(d0,d1,d2)->({maps})>, affine_map<(d0,d1,d2)->(d0,d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}}
      ins(%a, %b : tensor<{m}x{k}x{t}>, tensor<{rhs}x{t}>) outs(%init : tensor<{m}x{n}x{acc}>)
      attrs = {{merlin.numeric_semantics = "{ident}", merlin.input_domain = "{domain}",
               merlin.reduction_order = "k-ascending"}} {{
      ^bb0(%av: {t}, %bv: {t}, %c: {acc}):
        %ae = arith.extf %av : {t} to {acc}
        %be = arith.extf %bv : {t} to {acc}
        %sum = math.fma %ae, %be, %c : {acc}
        linalg.yield %sum : {acc}
    }} -> tensor<{m}x{n}x{acc}>
    func.return %out : tensor<{m}x{n}x{acc}>
  }}
}}'''


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("types", [("f8E4M3FN", "bf16"), ("f8E5M2", "f32"), ("f16", "f32")])
def test_geometry_and_policy_are_read_from_actual_source(transposed, types):
    policy = replace(POLICY, input_type=types[0], accumulator_type=types[1])
    selected = read_ordered_float_dense(source(transposed=transposed, policy=policy), entry="dense", policy=policy)
    assert (selected.m, selected.k, selected.n) == (32, 64, 96)
    assert selected.rhs_transposed is transposed
    assert selected.policy == policy


@pytest.mark.parametrize("old,new", [
    ("0.000000e+00", "-0.000000e+00"),
    ("0.000000e+00", "1.000000e+00"),
    ("math.fma %ae, %be, %c", "math.fma %ae, %ae, %c"),
    ("math.fma %ae, %be, %c", "math.fma %ae, %be, %ae"),
    ("linalg.yield %sum", "linalg.yield %c"),
    ("func.return %out", "func.return %init"),
    ("k-ascending", "unordered"),
    ("test-ordered-fma-v1", "different-policy"),
    ("test-finite-normal-or-zero", "unchecked"),
    ("%be = arith.extf %bv", "%be = arith.extf %av"),
    ("math.fma %ae, %be, %c : bf16", "math.fma %ae, %be, %c fastmath<reassoc> : bf16"),
])
def test_changed_recurrence_or_domain_is_rejected(old, new):
    original = source()
    assert old in original
    with pytest.raises(Exception):
        read_ordered_float_dense(original.replace(old, new), entry="dense", policy=POLICY)


def test_source_identity_changes_with_geometry():
    first = read_ordered_float_dense(source(), entry="dense", policy=POLICY)
    second = read_ordered_float_dense(source(k=32), entry="dense", policy=POLICY)
    assert first.source_sha256 != second.source_sha256


def test_fill_must_yield_the_actual_scalar_initializer():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    from xdsl.dialects.linalg.ops import FillOp

    module = parse_mlir_text(source())
    fill = next(op for op in module.walk() if type(op) is FillOp)
    fill.body.block.last_op.operands = [fill.body.block.args[1]]
    with pytest.raises(ValueError, match="fill body"):
        read_ordered_float_dense(text(module, generic=True), entry="dense", policy=POLICY)
