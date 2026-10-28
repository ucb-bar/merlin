"""Independent fixed bf16 policy qualification roster; no native calls at import.

Expected claims follow the zero-init policy candidate. xDSL checking is explicitly
not authoritative MLIR validation. Finite numerical equality is not used here.
"""

from dataclasses import dataclass

from lean_bf16_f32acc_cases import bmm, line_with, matmul, operands, replace_once, reverse_reduction, run_pass


@dataclass(frozen=True)
class Case:
    name: str
    before: str
    after: str
    expected: str
    sites: int
    reason: str


def cases() -> list[Case]:
    result = []
    for fmt in ("bf16", "f16"):
        for form, source in (
            ("named", matmul(fmt)),
            ("generic", bmm(fmt)),
            ("empty_k", bmm(fmt, depth=0)),
            ("tiny_k", bmm(fmt, depth=1)),
            ("empty_result", bmm(fmt, m=0)),
            ("generic_transpose", bmm(fmt, m=4, n=4, depth=4).replace("(d0, d3, d2)", "(d0, d2, d3)")),
        ):
            after, count = run_pass(source)
            result.append(
                Case(
                    fmt + "_" + form,
                    source,
                    after,
                    "raw_policy_match",
                    count,
                    "ordered f32 reference, zero initialization; human gate still closed",
                )
            )
    source = bmm()
    after, count = run_pass(source)
    mul, add = line_with(after, '"arith.mulf"'), line_with(after, '"arith.addf"')
    exts = [line for line in after.splitlines() if '"arith.extf"' in line]
    acc = line_with(after, '"linalg.generic"', "iterator_type<reduction>")
    fill = line_with(after, '"linalg.fill"')
    mutations = {
        "wrong_accumulator_f64": after.replace("f32", "f64"),
        "wrong_source_operand": replace_once(
            after, exts[1], exts[1].replace("(" + operands(exts[1])[0] + ")", "(" + operands(exts[0])[0] + ")")
        ),
        "wrong_positive_zero": replace_once(after, "<{value = 0.000000e+00 : f32}>", "<{value = 1.000000e+00 : f32}>"),
        "negative_zero": replace_once(after, "<{value = 0.000000e+00 : f32}>", "<{value = -0.000000e+00 : f32}>"),
        "uninitialized_accumulator": replace_once(after, acc, acc.replace(operands(acc)[2], operands(fill)[1])),
        "wrong_final_rounding": replace_once(
            after,
            line_with(after, '"arith.truncf"'),
            line_with(after, '"arith.truncf"').replace(") :", ") <{roundingmode = 3 : i32}> :"),
        ),
        "wrong_order": reverse_reduction(after),
        "wrong_add_order": replace_once(
            after,
            add,
            add.replace("(" + ", ".join(operands(add)) + ")", "(" + ", ".join(reversed(operands(add))) + ")"),
        ),
        "reassoc_flag": replace_once(after, add, add.replace("<contract>", "<reassoc,contract>")),
        "nnan_flag": replace_once(after, add, add.replace("<contract>", "<nnan,contract>")),
        "fast_flag": replace_once(after, mul, mul.replace("<contract>", "<fast>")),
        "dropped_site": source,
    }
    for name, target in mutations.items():
        result.append(
            Case(name, source, target, "raw_policy_refusal", 1, "one changed clause/complete boundary mismatch")
        )
    for name, target in (
        ("none_flags", after.replace("fastmath<contract>", "fastmath<none>")),
        ("add_only_contract", replace_once(after, mul, mul.replace("<contract>", "<none>"))),
    ):
        result.append(Case(name, source, target, "raw_policy_match", 1, "explicit permitted independent flags"))
    for name, source in (
        ("nonzero_source_F14", bmm(init="1.000000e+00")),
        ("negative_zero_source", bmm(init="-0.000000e+00")),
        ("named_transpose", matmul(m=4, n=4, depth=4, transposed=True)),
    ):
        target, count = run_pass(source)
        result.append(
            Case(
                name,
                source,
                target,
                "raw_policy_refusal",
                count,
                "source domain refusal or source-map preservation counterexample",
            )
        )
    # Source operand permutations are explicitly admitted by the source policy;
    # their target remains the canonical left/right/accumulator ordering.
    for name, source in (
        ("source_mul_swap", bmm().replace('"arith.mulf"(%x, %y)', '"arith.mulf"(%y, %x)')),
        ("source_add_swap", bmm().replace('"arith.addf"(%c, %m)', '"arith.addf"(%m, %c)')),
    ):
        target, count = run_pass(source)
        result.append(Case(name, source, target, "raw_policy_match", count, "source permutations admitted explicitly"))
    one = bmm(m=3, n=5, depth=5)
    one = one.replace('"linalg.generic"(%a, %b, %s)', '"linalg.generic"(%a, %s)')
    one = one.replace(", affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>", "")
    one = one.replace("array<i32: 2, 1>", "array<i32: 1, 1>")
    one = one.replace("%x: bf16, %y: bf16, %c: bf16", "%x: bf16, %c: bf16")
    one = one.replace('"arith.mulf"(%x, %y)', '"arith.mulf"(%x, %x)')
    one = one.replace(
        "(tensor<2x3x5xbf16>, tensor<2x5x5xbf16>, tensor<2x3x5xbf16>)", "(tensor<2x3x5xbf16>, tensor<2x3x5xbf16>)"
    )
    result.append(
        Case(
            "one_input_contraction",
            one,
            run_pass(one)[0],
            "outside_policy_domain",
            0,
            "one-input contraction lies outside the two-input policy",
        )
    )
    from lean_bf16_f32acc_cases import ITERS

    named_batch = (
        bmm().replace('"linalg.generic"', '"linalg.batch_matmul"').replace("iterator_types = " + ITERS + ", ", "")
    )
    result.append(
        Case(
            "named_batch_outside_domain",
            named_batch,
            run_pass(named_batch)[0],
            "outside_policy_domain",
            0,
            "named batch_matmul is outside the declared candidate scope; never gate coverage",
        )
    )
    # Same-typed suffix changes and a duplicated complete contraction are valid
    # graph mutations; structural cloning supplies fresh SSA names on printing.
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i32

    from merlin.llvmlower import passes_xdsl
    from merlin.xdsl_dialects._common import text

    for name in ("unrelated_suffix", "duplicated_accumulation"):
        module = passes_xdsl.parse_mlir_text(after)
        body = next(op for op in module.ops if op.name == "func.func").regions[0].block
        if name == "unrelated_suffix":
            body.insert_op_before(arith.ConstantOp(IntegerAttr(7, i32)), body.last_op)
        else:
            contraction = next(op for op in body.ops if op.name == "linalg.generic")
            body.insert_op_before(contraction.clone(), body.last_op)
        result.append(
            Case(
                name,
                bmm(),
                text(module, generic=True),
                "raw_policy_refusal",
                1,
                "complete-output correspondence, including unused added operations",
            )
        )
    first = next(iter(passes_xdsl.parse_mlir_text(bmm()).ops))
    second = first.clone()
    second.properties["sym_name"] = StringAttr("second")
    original = ModuleOp([first.clone(), second.clone()])
    before_multi = text(original, generic=True)
    after_multi, count = run_pass(before_multi)
    result.append(
        Case(
            "two_site_complete",
            before_multi,
            after_multi,
            "raw_policy_match",
            2,
            "every source site in both complete functions is checked",
        )
    )
    transformed = list(passes_xdsl.parse_mlir_text(after_multi).ops)
    dropped = ModuleOp([transformed[0].clone(), second.clone()])
    result.append(
        Case(
            "two_site_drop_second_rewrite",
            before_multi,
            text(dropped, generic=True),
            "raw_policy_refusal",
            2,
            "one unchanged eligible source site cannot be omitted",
        )
    )
    extra = transformed[1].clone()
    extra.properties["sym_name"] = StringAttr("extra")
    duplicated = ModuleOp([*(op.clone() for op in transformed), extra])
    result.append(
        Case(
            "two_site_duplicate_function",
            before_multi,
            text(duplicated, generic=True),
            "raw_policy_refusal",
            2,
            "an extra output function violates complete-program correspondence",
        )
    )
    return result
