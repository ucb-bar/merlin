"""Concrete integer semantics versus MLIR's independent folder and named-op generalizer.

UB is classified from the generated inputs, never inferred from Lean's answer. Unfolded cases
are counted explicitly and cannot replace measured coverage. This does not qualify IEEE floats.
"""

from __future__ import annotations

import itertools
import json
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.verify.lean.export import MLIR_EXPORT, export_file

OPS = (
    "addi",
    "subi",
    "muli",
    "andi",
    "ori",
    "xori",
    "maxsi",
    "minsi",
    "maxui",
    "minui",
    "divsi",
    "divui",
    "remsi",
    "remui",
    "ceildivsi",
    "ceildivui",
    "floordivsi",
    "shli",
    "shrsi",
    "shrui",
)
DIVS = {"divsi", "divui", "remsi", "remui", "ceildivsi", "ceildivui", "floordivsi"}
SIGNED_DIVS = {"divsi", "ceildivsi", "floordivsi"}


def evaluate(path):
    driver = repo_root() / "merlin" / "tests" / "fixtures" / "lean_eval.lean"
    proc = subprocess.run(
        ["lake", "env", "lean", "--run", str(driver), str(path)],
        cwd=repo_root() / "merlin" / "lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return {row["function"]: row["outcome"] for row in json.loads(proc.stdout)}


def mlir_pass(source, destination, pipeline, tools):
    adapter = repo_root() / "merlin" / "tests" / "ir" / "lean_mlir_oracle.py"
    proc = subprocess.run(
        [str(tools.m2m_python), str(adapter), str(MLIR_EXPORT), str(source), str(destination), pipeline],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(destination.read_text())


@pytest.mark.parametrize("width", [1, 2, 8, 32, 64])
def test_integer_semantics_against_mlir_folder(tmp_path, lean_tools, record_property, width):
    modulus = 1 << width
    values = sorted({0, 1, 2 % modulus, modulus - 1, modulus // 2, modulus // 2 - 1})
    cases = [(op, x, y, "") for op, x, y in itertools.product(OPS, values, values)]
    # Hit the exact shift legality boundary, not just very large shift amounts.
    cases += [
        (op, x, y, "")
        for op, x, y in itertools.product(("shli", "shrsi", "shrui"), values, (width - 1, width, width + 1))
        if y < modulus and y not in values
    ]
    # Independently exercise poison-producing overflow flags as well as wrapping arithmetic.
    cases += [
        (op, x, y, flag)
        for op, x, y, flag in itertools.product(("addi", "subi", "muli"), values, values, ("nsw", "nuw"))
    ]
    functions = []
    for i, (op, x, y, flag) in enumerate(cases):
        flags = f" overflow<{flag}>" if flag else ""
        functions.append(f"""func.func @c{i}() -> i{width} {{
  %a = arith.constant {x} : i{width}
  %b = arith.constant {y} : i{width}
  %r = arith.{op} %a, %b{flags} : i{width}
  return %r : i{width}
}}""")
    source = tmp_path / "integers.mlir"
    source.write_text("module {\n" + "\n".join(functions) + "\n}")
    original = tmp_path / "original.json"
    export_file(source, original, backend="mlir", m2m_python=lean_tools.m2m_python)
    actual = evaluate(original)
    folded = mlir_pass(source, tmp_path / "folded.json", "builtin.module(canonicalize)", lean_tools)
    measured = dict.fromkeys(OPS, 0)
    counts = {
        "cases": len(cases),
        "constants": 0,
        "poison": 0,
        "ub": 0,
        "poison_inputs": 0,
        "poison_refinements": 0,
        "not_folded": 0,
    }
    assert len(actual) == len(cases)
    assert len(folded["module"]["ops"]) == len(cases)
    for function in folded["module"]["ops"]:
        name = function["attrs"]["sym_name"]["value"]
        op, x, y, flag = cases[int(name[1:])]
        outcome = actual[name]
        ub = (op in DIVS and y == 0) or (op in SIGNED_DIVS and x == modulus // 2 and y == modulus - 1)
        assert (outcome["tag"] == "ub") == ub, (width, op, x, y, flag, outcome)
        if ub:
            counts["ub"] += 1
            continue  # MLIR may choose any result for independently established immediate UB.
        assert outcome["tag"] == "ok", (width, op, x, y, flag, outcome)
        body = function["regions"][0]["blocks"][0]["ops"]
        returned = body[-1]["operands"][0]
        producer = next(o for o in body if any(r["id"] == returned for r in o["results"]))
        value = outcome["values"][0]
        signed_x = x if x < modulus // 2 else x - modulus
        signed_y = y if y < modulus // 2 else y - modulus
        arithmetic = {"addi": lambda a, b: a + b, "subi": lambda a, b: a - b, "muli": lambda a, b: a * b}
        poison = op in {"shli", "shrsi", "shrui"} and y >= width
        if flag:
            result = arithmetic[op](signed_x, signed_y) if flag == "nsw" else arithmetic[op](x, y)
            poison = not (-modulus // 2 <= result < modulus // 2) if flag == "nsw" else not (0 <= result < modulus)
        assert (value["kind"] == "poison") == poison, (width, op, x, y, flag, value)
        counts["poison_inputs"] += int(poison)
        if producer["name"] == "arith.constant":
            if poison:
                # Constant folding is a refinement: replacing poison by a concrete value is legal.
                # Do not mistake that for disagreement, or count it as exact-value agreement.
                counts["poison_refinements"] += 1
                continue
            expected = int(producer["attrs"]["value"]["value"]) % modulus
            assert value == {"kind": "integer", "width": width, "unsigned": expected}, (
                width,
                op,
                x,
                y,
                flag,
                value,
                expected,
            )
            counts["constants"] += 1
        elif producer["name"] == "ub.poison":
            assert value == {"kind": "poison"}, (width, op, x, y, flag, value)
            counts["poison"] += 1
        else:
            counts["not_folded"] += 1
            continue
        measured[op] += 1
    record_property("differential_counts", json.dumps(counts, sort_keys=True))
    assert all(measured.values()), measured
    assert counts["constants"] + counts["poison"] > len(cases) // 2, counts
    assert counts["poison_inputs"] > 0, counts


NAMED = {
    "map": (
        """%a = arith.constant dense<[1, 2]> : tensor<2xi32>
%init = arith.constant dense<[10, 20]> : tensor<2xi32>
%r = linalg.map ins(%a : tensor<2xi32>) outs(%init : tensor<2xi32>)
  (%x: i32, %old: i32) { %v = arith.addi %x, %old : i32
    linalg.yield %v : i32 }""",
        "tensor<2xi32>",
        [2],
        [11, 22],
    ),
    "reduce": (
        """%a = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
%init = arith.constant dense<[10, 20]> : tensor<2xi32>
%r = linalg.reduce ins(%a : tensor<2x3xi32>) outs(%init : tensor<2xi32>) dimensions = [1]
  (%x: i32, %old: i32) { %v = arith.addi %x, %old : i32
    linalg.yield %v : i32 }""",
        "tensor<2xi32>",
        [2],
        [16, 35],
    ),
    "matmul": (
        """%a = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
%b = arith.constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi32>
%init = arith.constant dense<[[10, 20], [30, 40]]> : tensor<2x2xi32>
%r = linalg.matmul ins(%a, %b : tensor<2x2xi32>, tensor<2x2xi32>)
  outs(%init : tensor<2x2xi32>) -> tensor<2x2xi32>""",
        "tensor<2x2xi32>",
        [2, 2],
        [29, 42, 73, 90],
    ),
    "transpose": (
        """%a = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
%init = arith.constant dense<0> : tensor<3x2xi32>
%r = linalg.transpose ins(%a : tensor<2x3xi32>) outs(%init : tensor<3x2xi32>) permutation = [1, 0]""",
        "tensor<3x2xi32>",
        [3, 2],
        [1, 4, 2, 5, 3, 6],
    ),
}


@pytest.mark.parametrize("family", sorted(NAMED))
def test_named_linalg_matches_mlir_generalization(tmp_path, lean_tools, family):
    body, ty, shape, values = NAMED[family]
    source = tmp_path / "named.mlir"
    source.write_text(f"func.func @forward() -> {ty} {{\n{body}\n return %r : {ty}\n}}")
    original, generic = tmp_path / "named.json", tmp_path / "generic.json"
    export_file(source, original, backend="mlir", m2m_python=lean_tools.m2m_python)
    document = mlir_pass(source, generic, "builtin.module(linalg-generalize-named-ops)", lean_tools)
    ops = document["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    assert any(op["name"] == "linalg.generic" for op in ops)
    assert all(op["name"] != f"linalg.{family}" for op in ops)
    expected = {
        "forward": {
            "tag": "ok",
            "values": [
                {
                    "kind": "tensor",
                    "shape": shape,
                    "values": [{"kind": "integer", "width": 32, "unsigned": v} for v in values],
                }
            ],
        }
    }
    assert evaluate(original) == expected
    assert evaluate(generic) == expected
