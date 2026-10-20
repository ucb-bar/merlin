"""Qualify the Lean-proved integer-body checker on Merlin's actual W8A8 output.

These are component checks, not complete W8A8 policy receipts. Mutations are made in actual
MLIR before authoritative export; neither the production pass nor the exporter is replaced.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8
from merlin.verify.lean.export import MLIR_EXPORT, export_file


def lowered(m=2, k=3, n=4):
    module = parse_mlir_text(f"""func.func @forward(%a: tensor<{m}x{k}xf32>,
        %b: tensor<{k}x{n}xf32>) -> tensor<{m}x{n}xf32> {{
      %e = tensor.empty() : tensor<{m}x{n}xf32>
      %z = arith.constant 0.0 : f32
      %f = linalg.fill ins(%z : f32) outs(%e : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
      %r = linalg.matmul ins(%a, %b : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>)
        outs(%f : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
      return %r : tensor<{m}x{n}xf32>
    }}""")
    assert lower_contraction_int8(module) == 1
    module.verify()
    function = next(op for op in module.walk() if op.name == "func.func")
    ops = list(function.body.block.ops)
    index, generic = next(
        (index, op)
        for index, op in enumerate(ops)
        if op.name == "linalg.generic" and [str(a.type) for a in op.body.block.args] == ["i8", "i8", "i32"]
    )
    return module, index, generic


def exported(module, tmp_path, tools, *, generic=False):
    source, document = tmp_path / "actual.mlir", tmp_path / "actual.json"
    module.verify()
    if generic:
        from io import StringIO

        from xdsl.printer import Printer

        stream = StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(module)
        source.write_text(stream.getvalue())
    else:
        source.write_text(str(module))
    # Explicitly verify mutations with the compiler's verifier, independently of xDSL.
    proc = subprocess.run(
        [
            str(tools.m2m_python),
            "-c",
            "from pathlib import Path\n"
            "import sys\n"
            "from torch_mlir import ir\n"
            "with ir.Context():\n"
            "    module = ir.Module.parse(Path(sys.argv[1]).read_text())\n"
            "    assert module.operation.verify()\n",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    export_file(source, document, backend="mlir", m2m_python=tools.m2m_python)
    return document


def check_body(document, index, samples=()):
    driver = repo_root() / "merlin/tests/fixtures/lean_w8a8_body.lean"
    proc = subprocess.run(
        ["lake", "env", "lean", "--run", str(driver), str(document), str(index), *map(str, samples)],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    assert proc.stdout.startswith("{"), proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["scope"] == "integer_body_only"
    assert (proc.returncode == 0) == result["matches"]
    return result


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (3, 17, 5)])
def test_actual_merlin_integer_body(tmp_path, lean_tools, shape):
    module, index, _ = lowered(*shape)
    assert check_body(exported(module, tmp_path, lean_tools), index)["matches"]


@pytest.mark.parametrize(
    "mutation",
    [
        "unsigned_extension",
        "add_product",
        "subtract_accumulator",
        "square_input",
        "drop_accumulator",
        "yield_accumulator",
        "nsw",
        "nuw",
    ],
)
def test_valid_mlir_integer_mutations_refused(tmp_path, lean_tools, mutation):
    from xdsl.dialects import arith

    module, index, generic = lowered()
    block = generic.body.block
    ext_a, _, product, addition, yielded = list(block.ops)
    if mutation == "unsigned_extension":
        ext_a.name = "arith.extui"
    elif mutation == "add_product":
        product.name = "arith.addi"
    elif mutation == "subtract_accumulator":
        addition.name = "arith.subi"
    elif mutation == "square_input":
        product.operands = [ext_a.results[0], ext_a.results[0]]
    elif mutation == "drop_accumulator":
        addition.operands = [product.results[0], product.results[0]]
    elif mutation == "yield_accumulator":
        yielded.operands = [block.args[2]]
    else:
        flag = arith.IntegerOverflowFlag.NSW if mutation == "nsw" else arith.IntegerOverflowFlag.NUW
        addition.properties["overflowFlags"] = arith.IntegerOverflowAttr([flag])
    assert not check_body(exported(module, tmp_path, lean_tools), index)["matches"]


def test_exported_ssa_renaming_and_alias_refusal(tmp_path, lean_tools):
    module, index, _ = lowered()
    document = exported(module, tmp_path, lean_tools)
    data = json.loads(document.read_text())
    region = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"][index]["regions"][0]
    block = region["blocks"][0]
    for arg in block["args"]:
        arg["id"] += 1000
    for op in block["ops"]:
        op["operands"] = [identifier + 1000 for identifier in op["operands"]]
        for result in op["results"]:
            result["id"] += 1000
    document.write_text(json.dumps(data))
    assert check_body(document, index)["matches"]
    # Malformed certificate input must refuse even though actual MLIR cannot contain this alias.
    block["ops"][0]["results"][0]["id"] = block["args"][0]["id"]
    document.write_text(json.dumps(data))
    assert not check_body(document, index)["matches"]


def scalarized(generic, a, b, acc):
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i8, i32
    from xdsl.ir import Region

    block = generic.clone().body.detach_block(0)
    constants = [arith.ConstantOp(IntegerAttr(v, ty)) for v, ty in ((a, i8), (b, i8), (acc, i32))]
    for arg, constant in zip(block.args, constants, strict=True):
        arg.replace_all_uses_with(constant.results[0])
    for arg in reversed(list(block.args)):
        block.erase_arg(arg)
    yielded = block.last_op
    returned = tuple(yielded.operands)
    block.erase_op(yielded)
    block.add_op(func.ReturnOp(*returned))
    block.insert_ops_before(constants, block.first_op)
    return ModuleOp([func.FuncOp.from_region("forward", [], [i32], Region(block))])


@pytest.mark.parametrize(
    "a,b,acc",
    [
        (-128, -128, 0),
        (-128, 127, 0),
        (127, -128, 17),
        (127, 127, -16384),
        (1, 1, 2147483647),
        (-1, 1, -2147483648),
        (0, -128, -9),
    ],
)
def test_actual_integer_body_against_mlir_folder(tmp_path, lean_tools, a, b, acc):
    module, index, generic = lowered()
    lean = check_body(exported(module, tmp_path, lean_tools), index, (a, b, acc))
    source, folded = tmp_path / "scalar.mlir", tmp_path / "folded.json"
    source.write_text(str(scalarized(generic, a, b, acc)))
    adapter = repo_root() / "merlin/tests/ir/lean_mlir_oracle.py"
    proc = subprocess.run(
        [
            str(lean_tools.m2m_python),
            str(adapter),
            str(MLIR_EXPORT),
            str(source),
            str(folded),
            "builtin.module(canonicalize)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ops = json.loads(folded.read_text())["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    returned = ops[-1]["operands"][0]
    constant = next(op for op in ops if any(r["id"] == returned for r in op["results"]))
    assert constant["name"] == "arith.constant"
    actual = int(constant["attrs"]["value"]["value"])
    expected = ((a * b + acc + (1 << 31)) % (1 << 32)) - (1 << 31)
    assert int(lean["value"]) == actual == expected
