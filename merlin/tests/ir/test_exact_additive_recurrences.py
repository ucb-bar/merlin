"""Only a fully checked additive backedge gains rational reassociation."""

import pytest
from mlir_fixtures import assert_unchanged

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.exact_additive_recurrences import expose_exact_additive_recurrences


def source(*, n=6, stride=1, extent=3, decrement=None, extra="", element="f32", accum="f64"):
    decrement = stride if decrement is None else decrement
    return parse_mlir_text(f"""module {{ func.func @forward(%x:tensor<{n}x{element}>) -> tensor<{n}x{element}> {{
      %z = arith.constant 0 : index
      %one = arith.constant 1 : index
      %n = arith.constant {n} : index
      %s = arith.constant {stride} : index
      %back = arith.constant {decrement} : index
      %axis = arith.constant {extent} : index
      %zero = arith.constant 0.0 : {accum}
      %out = tensor.empty() : tensor<{n}x{element}>
      %acc = tensor.empty() : tensor<{n}x{accum}>
      %r:2 = scf.for %i = %z to %n step %one iter_args(%o=%out, %a=%acc)
        -> (tensor<{n}x{element}>,tensor<{n}x{accum}>) {{
        %v = tensor.extract %x[%i] : tensor<{n}x{element}>
        %wide = arith.extf %v : {element} to {accum}
        %group = arith.divui %i, %s : index
        %pos = arith.remui %group, %axis : index
        %reset = arith.cmpi eq, %pos, %z : index
        %prev = scf.if %reset -> ({accum}) {{
          scf.yield %zero : {accum}
        }} else {{
          %j = arith.subi %i, %back : index
          %old = tensor.extract %a[%j] : tensor<{n}x{accum}>
          scf.yield %old : {accum}
        }}
        %sum = arith.addf %prev, %wide : {accum}
        %narrow = arith.truncf %sum : {accum} to {element}
        %new_o = tensor.insert %narrow into %o[%i] : tensor<{n}x{element}>
        %new_a = tensor.insert %sum into %a[%i] : tensor<{n}x{accum}>
        {extra}
        scf.yield %new_o,%new_a : tensor<{n}x{element}>,tensor<{n}x{accum}>
      }}
      return %r#0 : tensor<{n}x{element}>
    }} }}""")


@pytest.mark.parametrize("stride,extent", [(1, 3), (2, 3), (3, 2)])
@pytest.mark.parametrize("element,accum", [("f16", "f32"), ("bf16", "f32"), ("f32", "f64")])
def test_exact_contract_exposes_every_prefix_dependency(stride, extent, element, accum):
    module = source(stride=stride, extent=extent, element=element, accum=accum)
    assert expose_exact_additive_recurrences(module, quantized_contract={}) == 1
    module.verify()
    kernel = next(op for op in module.walk() if op.name == "linalg.generic")
    weights = list(kernel.inputs[1].owner.value.get_values())
    for j in range(6):
        for i in range(6):
            expected = j in [i - k * stride for k in range((i // stride) % extent + 1)]
            assert weights[j * 6 + i] == int(expected)


@pytest.mark.parametrize("options", [dict(decrement=2), dict(extra="%hidden = arith.addf %sum, %wide : f64")])
def test_unproved_backedge_or_extra_computation_is_retained(options):
    assert_unchanged(source(**options), lambda module: expose_exact_additive_recurrences(module, quantized_contract={}))


def test_ordered_ieee_loop_never_gains_an_exact_contraction():
    assert_unchanged(source(), expose_exact_additive_recurrences)


def test_rational_oracle_keeps_the_exact_prefix_before_output_rounding():
    torch = pytest.importorskip("torch")
    from fractions import Fraction

    import numpy as np

    from merlin.targetgen.exact_arithmetic import ExactTensor
    from merlin.targetgen.exact_quantized_reference import evaluate_operation

    values = np.array([[Fraction(1), Fraction(1, 2**54), Fraction(-1)]], dtype=object)
    result = evaluate_operation(torch.ops.aten.cumsum.default, [ExactTensor(values, torch.float32), 1], {})
    assert result.values.tolist() == [[Fraction(1), Fraction(1) + Fraction(1, 2**54), Fraction(1, 2**54)]]
