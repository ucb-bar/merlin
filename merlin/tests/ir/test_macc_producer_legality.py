"""Reviewed producer legality; independent checker mutants retain old defects."""

import pytest
from lean_bf16_f32acc_cases import STANDARD_BODY, bmm, matmul
from lean_half_reduction_cases import reduction
from xdsl.dialects import arith, builtin

from merlin.llvmlower import passes_xdsl
from merlin.xdsl_dialects._common import text


@pytest.mark.parametrize("fmt", ["bf16", "f16"])
@pytest.mark.parametrize("case", ["nonzero", "negative-zero", "wrong-multiply", "wrong-add", "relaxed-source"])
def test_unsupported_policy_rejects_before_mutating_actual_module(fmt, case):
    source = bmm(
        fmt,
        init="1.000000e+00" if case == "nonzero" else "-0.000000e+00" if case == "negative-zero" else "0.000000e+00",
        body=STANDARD_BODY.replace("%x, %y", "%x, %x")
        if case == "wrong-multiply"
        else STANDARD_BODY.replace("%c, %m", "%m, %m")
        if case == "wrong-add"
        else STANDARD_BODY.replace("fastmath<none>", "fastmath<contract>")
        if case == "relaxed-source"
        else STANDARD_BODY,
    )
    module = passes_xdsl.parse_mlir_text(source)
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="requested f32-accumulation policy: unsupported contraction legality"):
        passes_xdsl.lower_bf16_matmul_f32acc(module)
    assert text(module, generic=True) == before


@pytest.mark.parametrize("fmt", ["bf16", "f16"])
def test_explicit_transposed_named_maps_survive_production_rewrite(fmt):
    module = passes_xdsl.parse_mlir_text(matmul(fmt, m=3, n=3, depth=3, transposed=True))
    original = next(op for op in module.walk() if op.name == "linalg.matmul")
    maps = original.indexing_maps
    assert passes_xdsl.lower_bf16_matmul_f32acc(module) == 1
    accumulated = next(
        op
        for op in module.walk()
        if op.name == "linalg.generic" and any(a.data.value == "reduction" for a in op.iterator_types)
    )
    assert accumulated.indexing_maps == maps
    module.verify()


def test_one_unsupported_half_site_prevents_partial_module_policy_mutation():
    first = matmul(m=3, n=3, depth=3).splitlines()
    second = bmm(init="1.000000e+00").splitlines()
    source = "\n".join(
        [
            *first[:-1],
            *[line.replace('sym_name = "subject"', 'sym_name = "unsupported"') for line in second[1:-1]],
            first[-1],
        ]
    )
    module = passes_xdsl.parse_mlir_text(source)
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="unsupported contraction legality"):
        passes_xdsl.lower_bf16_matmul_f32acc(module)
    assert text(module, generic=True) == before


def test_f32_operands_refuse_before_any_invalid_extf_is_emitted():
    module = passes_xdsl.parse_mlir_text(matmul(operand="f32"))
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="unsupported contraction legality"):
        passes_xdsl.lower_bf16_matmul_f32acc(module)
    assert text(module, generic=True) == before
    assert not any(type(op) is arith.ExtFOp for op in module.walk())


def test_unsupported_semantic_attribute_is_not_discarded():
    module = passes_xdsl.parse_mlir_text(matmul())
    source = next(op for op in module.walk() if op.name == "linalg.matmul")
    source.attributes["unknown.semantic"] = builtin.IntegerAttr(1, builtin.i32)
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="unsupported contraction legality"):
        passes_xdsl.lower_bf16_matmul_f32acc(module)
    assert text(module, generic=True) == before


@pytest.mark.parametrize("fmt", ["bf16", "f16", "f32"])
@pytest.mark.parametrize("operation", ["addf", "mulf"])
def test_ordinary_reductions_are_unchanged_by_explicit_macc_policy(tmp_path, fmt, operation):
    from merlin.verify import compiler_transforms as transforms

    module = passes_xdsl.parse_mlir_text(reduction(fmt, operation))
    module.verify()
    before = text(module, generic=True)
    assert passes_xdsl.lower_bf16_matmul_f32acc(module) == 0
    assert text(module, generic=True) == before
    # Exercise the public option path as well as the producer; preprocessing
    # alone does not invoke this optional policy.
    result, summary = transforms.apply(before, tmp_path, options=transforms.Options(macc_f32acc=True))
    assert summary["changed"] == []
    assert transforms.MACC in summary["unchanged"]
    assert not any(op.name == "arith.extf" for op in passes_xdsl.parse_mlir_text(result).walk())
