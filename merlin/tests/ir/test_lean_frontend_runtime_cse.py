"""Exact same-interpretation CSE suffixes, with actual MLIR verification first."""

from __future__ import annotations

import copy

import pytest
from test_lean_frontend_positions import candidate_float, initial_boundary
from test_lean_frontend_runtime_checkpoint import driver, exported

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import differential as diff
from merlin.verify.lean import frontend_runtime as runtime
from merlin.xdsl_dialects._common import text as to_text


def pair(*, nested=False, callee=False, wrong=False) -> tuple[str, str]:
    sources = []
    for share in (False, True):
        constants = "%a = arith.constant 7 : i64\n"
        if not share:
            constants += "%b = arith.constant 7 : i64\n"
        elif wrong:
            constants = "%a = arith.constant 8 : i64\n"
        value = "%a" if share else "%b"
        if nested:
            body = f"""%v = scf.if %flag -> (i64) {{
              {constants} scf.yield {value} : i64
            }} else {{ %zero = arith.constant 0 : i64 scf.yield %zero : i64 }}
            return %v : i64"""
        else:
            body = f"{constants} return %a, {value} : i64, i64"
        args, outputs = ("%flag: i1", "i64") if nested else ("", "(i64, i64)")
        if callee:
            calls = "%r = func.call @helper(%flag) : (i1) -> i64\nreturn %r : i64"
            source = (
                f"builtin.module {{func.func @helper({args}) -> {outputs} {{{body}}}\n"
                f"func.func @subject(%flag: i1) -> i64 {{{calls}}}}}"
            )
        else:
            source = f"builtin.module {{func.func @subject({args}) -> {outputs} {{{body}}}}}"
        sources.append(source)
    return tuple(sources)


def suffix(before: str, after: str, work, tools) -> dict:
    source = exported(before, work / "source", tools)
    target = exported(after, work / "target", tools)
    return driver({"mode": "suffix", "before_export": str(source), "after_export": str(target)}, work / "proof", tools)


@pytest.mark.parametrize("nested,callee", [(False, False), (True, False), (True, True)])
def test_closed_cse_preserves_complete_program(tmp_path, fidelity_tools, nested, callee):
    before, after = pair(nested=nested, callee=callee)
    assert suffix(before, after, tmp_path, fidelity_tools)["accepted"]
    # Equally typed, MLIR-valid constant changes must fail the actual Lean checker.
    assert not suffix(before, pair(nested=nested, callee=callee, wrong=True)[1], tmp_path / "wrong", fidelity_tools)[
        "accepted"
    ]
    samples = [[["0"]], [["1"]]] if nested else [[]]
    previous = None
    for label, source in (("before", before), ("after", after)):
        lean, document, path = candidate_float(source, samples, tmp_path / label, fidelity_tools)
        inputs, outputs = diff.signature(document, "subject")
        native = diff.execute_mlir(
            path, samples, tmp_path / label / "native", fidelity_tools, inputs=inputs, outputs=outputs
        )
        diff.assert_equal(lean, native["results"], outputs, samples)
        if previous is not None:
            assert lean == previous
        previous = lean


@pytest.mark.parametrize("change", ["wrong-use", "non-adjacent", "changed-module-attribute"])
def test_compiler_valid_cse_refusals(tmp_path, fidelity_tools, change):
    before, after = pair()
    if change == "wrong-use":
        before = before.replace("return %a, %b", "%c = arith.constant 8 : i64\nreturn %a, %c")
    elif change == "non-adjacent":
        before = before.replace("%b =", "%c = arith.constant 8 : i64\n%b =")
        after = after.replace("return", "%c = arith.constant 8 : i64\nreturn")
    else:
        after = after.replace("builtin.module {", 'builtin.module attributes {frontend.policy = "changed"} {')
    assert not suffix(before, after, tmp_path, fidelity_tools)["accepted"]


@pytest.mark.parametrize("kind", ["unmodelled-attribute", "nonconstant"])
def test_unsupported_sharing_refuses(tmp_path, fidelity_tools, kind):
    from xdsl.dialects import arith, builtin

    if kind == "nonconstant":
        before = """builtin.module {func.func @subject() -> (tensor<1xi64>, tensor<1xi64>) {
          %a = tensor.empty() : tensor<1xi64>
          %b = tensor.empty() : tensor<1xi64>
          return %a, %b : tensor<1xi64>, tensor<1xi64> }}"""
        after = before.replace("%b = tensor.empty() : tensor<1xi64>", "").replace("return %a, %b", "return %a, %a")
        assert not suffix(before, after, tmp_path, fidelity_tools)["accepted"]
        return
    sources = []
    for text in pair():
        module = parse_mlir_text(text)
        for op in module.walk():
            if isinstance(op, arith.ConstantOp):
                op.attributes["qualification.unsupported"] = builtin.StringAttr("requires an interpretation")
        sources.append(to_text(module, generic=True))
    assert not suffix(*sources, tmp_path, fidelity_tools)["accepted"]


def extended_sources() -> tuple[str, str, str]:
    from xdsl.dialects import arith, builtin, func

    sources = []
    for text in initial_boundary():
        module = parse_mlir_text(text)
        fn = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
        block = fn.body.block
        ret = block.last_op
        a = arith.ConstantOp(builtin.IntegerAttr(7, builtin.i64))
        b = arith.ConstantOp(builtin.IntegerAttr(7, builtin.i64))
        block.insert_ops_before([a, b], ret)
        values = [*ret.operands, a.result, b.result]
        block.insert_op_before(func.ReturnOp(*values), ret)
        block.erase_op(ret)
        fn.properties["function_type"] = builtin.FunctionType.from_lists(
            list(fn.function_type.inputs), [v.type for v in values]
        )
        module.verify()
        sources.append(to_text(module, generic=True))
    # Share the actual adjacent pair, updating its real return uses.
    fn = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    b = list(fn.body.block.ops)[-2]
    a = list(fn.body.block.ops)[-3]
    b.result.replace_all_uses_with(a.result)
    fn.body.block.erase_op(b)
    module.verify()
    return (*sources, to_text(module, generic=True))


def test_repair_policy_composes_through_nonalpha_suffix(tmp_path, fidelity_tools):
    paths = [tmp_path / f"stage-{i}.mlir" for i in range(3)]
    for path, text in zip(paths, extended_sources(), strict=True):
        path.write_text(text)
    receipt = runtime.verify_candidate(
        *paths[:2],
        tmp_path / "evidence",
        function="subject",
        start=91,
        compiler_python=fidelity_tools.compiler_python,
        lake=fidelity_tools.lake,
        continuation_count=41,
        suffix=(paths[2],),
    )
    assert receipt["status"] == "verified_candidate", receipt.get("reason")
    assert receipt["answer"]["suffix"] and receipt["answer"]["continuation"]["accepted"]
    assert not receipt["ordinary_source_equivalence"] and not receipt["whole_function_defined"]
    runtime.validate_receipt(receipt)
    wrong = copy.deepcopy(receipt)
    wrong["artifacts"][1], wrong["artifacts"][2] = wrong["artifacts"][2], wrong["artifacts"][1]
    with pytest.raises(ValueError, match="receipt order or checkpoint"):
        runtime.validate_receipt(wrong)
