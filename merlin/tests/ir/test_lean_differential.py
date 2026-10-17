"""Execute the proved Lean denotation and upstream MLIR on the same runtime input bits.

The corpus chooses defined inputs independently of Lean (division, shifts, casts and bounds).
Native UB is never used as an expected answer. Missing tools skip normally and FAIL with
MERLIN_LEAN_REQUIRED=1, matching the other Lean qualification suites.
"""

from __future__ import annotations

import itertools
import json
import random
import struct
from pathlib import Path

import pytest

from merlin.verify.lean import differential as diff


def _function(types, body, results, result_types=None, name="subject"):
    result_types = result_types or [types[0]] * len(results)
    args = ", ".join(f"%a{i}: {t}" for i, t in enumerate(types))
    return (
        f"func.func @{name}({args}) -> ({', '.join(result_types)}) {{\n"
        + "\n".join(body)
        + f"\nreturn {', '.join(results)} : {', '.join(result_types)}\n}}"
    )


def _integers(width):
    mask = (1 << width) - 1
    edges = sorted({n & mask for n in [0, 1, mask, mask >> 1, 1 << (width - 1), (1 << (width - 1)) + 1]})
    rng = random.Random(8121 + width)
    pairs = (
        list(itertools.product(range(1 << width), repeat=2)) if width <= 4 else list(itertools.product(edges, repeat=2))
    )
    pairs += [(rng.getrandbits(width), rng.getrandbits(width)) for _ in range(64)]
    return pairs


@pytest.mark.parametrize("width", [1, 4, 8, 13, 32, 64])
def test_integer_arithmetic_and_comparisons(tmp_path, fidelity_tools, width):
    ty = f"i{width}"
    names = ["addi", "subi", "muli", "andi", "ori", "xori", "maxsi", "minsi", "maxui", "minui"]
    predicates = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]
    body = [f"%r{i} = arith.{n} %a0, %a1 : {ty}" for i, n in enumerate(names)]
    body += [f"%p{i} = arith.cmpi {p}, %a0, %a1 : {ty}" for i, p in enumerate(predicates)]
    body += [f"%sel = arith.select %p2, %a0, %a1 : {ty}", f"%abs = math.absi %a0 : {ty}"]
    values = [f"%r{i}" for i in range(len(names))] + [f"%p{i}" for i in range(len(predicates))] + ["%sel", "%abs"]
    source = _function([ty, ty], body, values, [ty] * len(names) + ["i1"] * len(predicates) + [ty, ty])
    samples = [[[a], [b]] for a, b in _integers(width)]
    assert diff.compare(source, samples, tmp_path, fidelity_tools)["samples"] == len(samples)


@pytest.mark.parametrize("width", [1, 4, 8, 13, 32, 64])
def test_integer_division_on_independently_defined_inputs(tmp_path, fidelity_tools, width):
    ty = f"i{width}"
    names = ["divsi", "divui", "remsi", "remui", "ceildivsi", "ceildivui", "floordivsi"]
    body = [f"%r{i} = arith.{n} %a0, %a1 : {ty}" for i, n in enumerate(names)]
    source = _function([ty, ty], body, [f"%r{i}" for i in range(len(names))])
    # Exclude zero divisors and signed overflow independently. remsi(INT_MIN,-1) also excluded:
    # its MLIR/LLVM discrepancy is already recorded as ledger A2, and native srem may trap.
    samples = [
        [[a], [b]] for a, b in _integers(width) if b != 0 and not (a == 1 << (width - 1) and b == (1 << width) - 1)
    ]
    diff.compare(source, samples, tmp_path, fidelity_tools)


@pytest.mark.parametrize("width", [1, 4, 8, 13, 32, 64])
def test_shifts_with_legal_counts(tmp_path, fidelity_tools, width):
    ty = f"i{width}"
    names = ["shli", "shrsi", "shrui"]
    source = _function(
        [ty, ty], [f"%r{i} = arith.{n} %a0, %a1 : {ty}" for i, n in enumerate(names)], [f"%r{i}" for i in range(3)]
    )
    samples = [[[a], [b % width]] for a, b in _integers(width)]
    diff.compare(source, samples, tmp_path, fidelity_tools)


@pytest.mark.parametrize("source_width,target_width", [(1, 8), (4, 13), (8, 32), (13, 64), (32, 64)])
def test_integer_casts(tmp_path, fidelity_tools, source_width, target_width):
    small, large = f"i{source_width}", f"i{target_width}"
    source = _function(
        [small, large],
        [
            f"%s = arith.extsi %a0 : {small} to {large}",
            f"%u = arith.extui %a0 : {small} to {large}",
            f"%t = arith.trunci %a1 : {large} to {small}",
        ],
        ["%s", "%u", "%t"],
        [large, large, small],
    )
    samples = [[[a], [b]] for a, b in _integers(target_width)]
    diff.compare(source, samples, tmp_path, fidelity_tools)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_index_casts_and_arithmetic(tmp_path, fidelity_tools, index_bits):
    source = _function(
        ["index", "i64"],
        [
            "%s = arith.index_cast %a0 : index to i64",
            "%u = arith.index_castui %a0 : index to i64",
            "%i = arith.index_cast %a1 : i64 to index",
            "%j = arith.index_castui %a1 : i64 to index",
            "%a = arith.addi %a0, %i : index",
        ],
        ["%s", "%u", "%i", "%j", "%a"],
        ["i64", "i64", "index", "index", "index"],
    )
    diff.compare(source, [[[a], [b]] for a, b in _integers(64)], tmp_path, fidelity_tools, index_bits=index_bits)


@pytest.mark.parametrize("width", [8, 32, 64])
def test_unviolated_integer_promises(tmp_path, fidelity_tools, width):
    ty = f"i{width}"
    source = _function(
        [ty, ty],
        [
            f"%a = arith.addi %a0, %a1 overflow<nsw, nuw> : {ty}",
            f"%m = arith.muli %a0, %a1 overflow<nsw, nuw> : {ty}",
            f"%s = arith.subi %m, %a0 overflow<nsw, nuw> : {ty}",
        ],
        ["%a", "%m", "%s"],
    )
    diff.compare(source, [[[a], [b]] for a in range(8) for b in range(1, 8)], tmp_path, fidelity_tools)


def _float_bits(value, width):
    return int.from_bytes(struct.pack("<f" if width == 32 else "<d", value), "little")


def _floats(width):
    fraction, exponent = (23, 8) if width == 32 else (52, 11)
    infinity = ((1 << exponent) - 1) << fraction
    sign = 1 << (width - 1)
    values = [_float_bits(x, width) for x in [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, 2.5, -2.5, 3.25]]
    values += [
        1,
        sign | 1,
        1 << fraction,
        infinity - 1,
        infinity,
        sign | infinity,
        infinity | (1 << (fraction - 1)),
        infinity | sign | (1 << (fraction - 1)),
    ]
    return values


@pytest.mark.parametrize("width", [32, 64])
def test_float_arithmetic_and_rounding(tmp_path, fidelity_tools, width):
    ty = f"f{width}"
    binary = ["arith.addf", "arith.subf", "arith.mulf", "arith.divf"]
    unary = [
        "arith.negf",
        "math.absf",
        "math.sqrt",
        "math.ceil",
        "math.floor",
        "math.round",
        "math.roundeven",
        "math.trunc",
    ]
    body = [f"%b{i} = {op} %a0, %a1 : {ty}" for i, op in enumerate(binary)]
    body += [f"%u{i} = {op} %a0 : {ty}" for i, op in enumerate(unary)]
    body += [f"%fma = math.fma %a0, %a1, %a0 : {ty}"]
    results = [f"%b{i}" for i in range(len(binary))] + [f"%u{i}" for i in range(len(unary))] + ["%fma"]
    source = _function([ty, ty], body, results)
    rng = random.Random(412 + width)
    pairs = list(itertools.product(_floats(width), repeat=2))
    pairs += [(rng.getrandbits(width), rng.getrandbits(width)) for _ in range(64)]
    diff.compare(source, [[[a], [b]] for a, b in pairs], tmp_path, fidelity_tools, arithmetic_nans=True)


@pytest.mark.parametrize("width", [32, 64])
def test_all_float_comparison_predicates(tmp_path, fidelity_tools, width):
    ty = f"f{width}"
    predicates = [
        "false",
        "oeq",
        "ogt",
        "oge",
        "olt",
        "ole",
        "one",
        "ord",
        "ueq",
        "ugt",
        "uge",
        "ult",
        "ule",
        "une",
        "uno",
        "true",
    ]
    source = _function(
        [ty, ty],
        [f"%p{i} = arith.cmpf {p}, %a0, %a1 : {ty}" for i, p in enumerate(predicates)],
        [f"%p{i}" for i in range(16)],
        ["i1"] * 16,
    )
    diff.compare(source, [[[a], [b]] for a, b in itertools.product(_floats(width), repeat=2)], tmp_path, fidelity_tools)


def test_float_format_and_integer_conversions(tmp_path, fidelity_tools):
    source = _function(
        ["f32", "f64", "i32"],
        [
            "%e = arith.extf %a0 : f32 to f64",
            "%t = arith.truncf %a1 : f64 to f32",
            "%s = arith.sitofp %a2 : i32 to f32",
            "%u = arith.uitofp %a2 : i32 to f32",
            "%d = arith.sitofp %a2 : i32 to f64",
            "%v = arith.uitofp %a2 : i32 to f64",
            "%i = arith.fptosi %a0 : f32 to i32",
            "%j = arith.fptoui %a1 : f64 to i32",
            "%bits = arith.bitcast %a0 : f32 to i32",
            "%back = arith.bitcast %a2 : i32 to f32",
        ],
        ["%e", "%t", "%s", "%u", "%d", "%v", "%i", "%j", "%bits", "%back"],
        ["f64", "f32", "f32", "f32", "f64", "f64", "i32", "i32", "i32", "f32"],
    )
    # float-to-integer values safely inside the destination range, independently of the model.
    samples = [
        [[_float_bits(a, 32)], [_float_bits(b, 64)], [c]]
        for a, b, c in itertools.product(
            [-1024.5, -1.5, -0.0, 0.5, 100.75], [0.0, 0.5, 1.5, 1024.25], [0, -1, 16777217, -(1 << 31), (1 << 31) - 1]
        )
    ]
    diff.compare(source, samples, tmp_path, fidelity_tools)


TENSOR_PROGRAMS = {
    "reshape": """func.func @subject(%a0: tensor<2x3xi32>) -> (tensor<6xi32>, tensor<3x2xi32>) {
      %c = tensor.collapse_shape %a0 [[0, 1]] : tensor<2x3xi32> into tensor<6xi32>
      %e = tensor.expand_shape %c [[0, 1]] output_shape [3, 2] : tensor<6xi32> into tensor<3x2xi32>
      return %c, %e : tensor<6xi32>, tensor<3x2xi32>
    }""",
    "strided_slices_concat": """func.func @subject(%a0: tensor<3x4xi32>)
        -> (tensor<2x2xi32>, tensor<4xi32>, tensor<2x4xi32>) {
      %s = tensor.extract_slice %a0[0, 0] [2, 2] [2, 2] : tensor<3x4xi32> to tensor<2x2xi32>
      %r = tensor.extract_slice %a0[1, 0] [1, 4] [1, 1] : tensor<3x4xi32> to tensor<4xi32>
      %c = tensor.concat dim(1) %s, %s : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x4xi32>
      return %s, %r, %c : tensor<2x2xi32>, tensor<4xi32>, tensor<2x4xi32>
    }""",
    "insert_slice": """func.func @subject(%a0: tensor<2x2xi32>, %a1: tensor<3x4xi32>) -> tensor<3x4xi32> {
      %r = tensor.insert_slice %a0 into %a1[0, 0] [2, 2] [2, 2] : tensor<2x2xi32> into tensor<3x4xi32>
      return %r : tensor<3x4xi32>
    }""",
    "insert_extract": """func.func @subject(%a0: tensor<2x3xi32>, %a1: i32, %a2: index, %a3: index)
        -> (tensor<2x3xi32>, i32) {
      %t = tensor.insert %a1 into %a0[%a2, %a3] : tensor<2x3xi32>
      %s = tensor.extract %t[%a2, %a3] : tensor<2x3xi32>
      return %t, %s : tensor<2x3xi32>, i32
    }""",
    "splat_from_elements": """func.func @subject(%a0: i32, %a1: i32) -> (tensor<2x3xi32>, tensor<2xi32>, tensor<i32>) {
      %s = tensor.splat %a0 : tensor<2x3xi32>
      %t = tensor.from_elements %a0, %a1 : tensor<2xi32>
      %r = tensor.from_elements %a1 : tensor<i32>
      return %s, %t, %r : tensor<2x3xi32>, tensor<2xi32>, tensor<i32>
    }""",
    "empty_dimension": """func.func @subject(%a0: tensor<0x3xi32>) -> tensor<0xi32> {
      %r = tensor.collapse_shape %a0 [[0, 1]] : tensor<0x3xi32> into tensor<0xi32>
      return %r : tensor<0xi32>
    }""",
    "dense_constants": """func.func @subject() -> (tensor<2x3xi32>, tensor<2x2xi1>, tensor<2xf32>) {
      %a = arith.constant dense<[[1, -2, 3], [-2147483648, 0, 2147483647]]> : tensor<2x3xi32>
      %b = arith.constant dense<[[true, false], [false, true]]> : tensor<2x2xi1>
      %c = arith.constant dense<[1.0, -0.0]> : tensor<2xf32>
      return %a, %b, %c : tensor<2x3xi32>, tensor<2x2xi1>, tensor<2xf32>
    }""",
    "tensor_select": """func.func @subject(%a0: tensor<3xi1>, %a1: tensor<3xi32>, %a2: tensor<3xi32>, %a3: i1)
        -> (tensor<3xi32>, tensor<3xi32>) {
      %t = arith.select %a0, %a1, %a2 : tensor<3xi1>, tensor<3xi32>
      %s = arith.select %a3, %a1, %a2 : tensor<3xi32>
      return %t, %s : tensor<3xi32>, tensor<3xi32>
    }""",
}


def _tensor_samples(source, tmp_path, fidelity_tools):
    from merlin.verify.lean.export import export_file

    path, exported = tmp_path / "signature.mlir", tmp_path / "signature.json"
    path.write_text(source)
    export_file(path, exported, backend="mlir", m2m_python=fidelity_tools.compiler_python)
    inputs, _ = diff.signature(json.loads(exported.read_text()), "subject")
    rng = random.Random(712)
    return [[[rng.getrandbits(t.width(64)) for _ in range(t.count)] for t in inputs] for _ in range(8)]


@pytest.mark.parametrize("name", sorted(TENSOR_PROGRAMS))
def test_tensor_operations(tmp_path, fidelity_tools, name):
    source = TENSOR_PROGRAMS[name]
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    if name == "insert_extract":
        for i, sample in enumerate(samples):
            sample[2], sample[3] = [i % 2], [i % 3]
    diff.compare(source, samples, tmp_path, fidelity_tools)


def test_internal_calls_and_passing_assertion(tmp_path, fidelity_tools):
    source = """func.func private @twice(%x: i32) -> i32 {
      %r = arith.addi %x, %x : i32
      return %r : i32
    }
    func.func @subject(%a0: i32) -> i32 {
      %true = arith.constant true
      cf.assert %true, "always true"
      %r = func.call @twice(%a0) : (i32) -> i32
      return %r : i32
    }"""
    diff.compare(source, [[[x]] for x in [0, 1, -1, (1 << 31) - 1, -(1 << 31)]], tmp_path, fidelity_tools)


STRUCTURED_PROGRAMS = {
    **{
        f"fill_{name}": f"""func.func @subject(%a0: {scalar}) -> tensor<{shape}> {{
          %e = tensor.empty() : tensor<{shape}>
          %r = linalg.fill ins(%a0 : {scalar}) outs(%e : tensor<{shape}>) -> tensor<{shape}>
          return %r : tensor<{shape}>
        }}"""
        for name, shape, scalar in [
            ("rank_zero", "i32", "i32"),
            ("empty", "0x3xi32", "i32"),
            ("f32", "2x3xf32", "f32"),
            ("f64", "2x3x2xf64", "f64"),
        ]
    },
    "fill": """func.func @subject(%a0: i32) -> tensor<2x3xi32> {
      %e = tensor.empty() : tensor<2x3xi32>
      %r = linalg.fill ins(%a0 : i32) outs(%e : tensor<2x3xi32>) -> tensor<2x3xi32>
      return %r : tensor<2x3xi32>
    }""",
    "matmul": """func.func @subject(%a0: tensor<2x3xi8>, %a1: tensor<3x4xi8>, %a2: tensor<2x4xi32>) -> tensor<2x4xi32> {
      %r = linalg.matmul ins(%a0, %a1 : tensor<2x3xi8>, tensor<3x4xi8>) outs(%a2 : tensor<2x4xi32>) -> tensor<2x4xi32>
      return %r : tensor<2x4xi32>
    }""",
    "batch_matmul": """func.func @subject(%a0: tensor<2x3x4xi32>, %a1: tensor<2x4x2xi32>, %a2: tensor<2x3x2xi32>)
        -> tensor<2x3x2xi32> {
      %r = linalg.batch_matmul ins(%a0, %a1 : tensor<2x3x4xi32>, tensor<2x4x2xi32>)
        outs(%a2 : tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
      return %r : tensor<2x3x2xi32>
    }""",
    "quantized_matmul": """func.func @subject(%a0: tensor<2x3xi8>, %a1: tensor<3x2xi8>, %a2: i32, %a3: i32,
        %a4: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %r = linalg.quantized_matmul ins(%a0, %a1, %a2, %a3 : tensor<2x3xi8>, tensor<3x2xi8>, i32, i32)
        outs(%a4 : tensor<2x2xi32>) -> tensor<2x2xi32>
      return %r : tensor<2x2xi32>
    }""",
    "copy": """func.func @subject(%a0: tensor<2x3xi8>, %a1: tensor<2x3xi32>) -> tensor<2x3xi32> {
      %r = linalg.copy ins(%a0 : tensor<2x3xi8>) outs(%a1 : tensor<2x3xi32>) -> tensor<2x3xi32>
      return %r : tensor<2x3xi32>
    }""",
    "transpose": """func.func @subject(%a0: tensor<2x3x4xi32>, %a1: tensor<3x4x2xi32>) -> tensor<3x4x2xi32> {
      %r = linalg.transpose ins(%a0 : tensor<2x3x4xi32>) outs(%a1 : tensor<3x4x2xi32>) permutation = [1, 2, 0]
      return %r : tensor<3x4x2xi32>
    }""",
    "broadcast": """func.func @subject(%a0: tensor<2x3xi32>, %a1: tensor<4x2x5x3xi32>) -> tensor<4x2x5x3xi32> {
      %r = linalg.broadcast ins(%a0 : tensor<2x3xi32>) outs(%a1 : tensor<4x2x5x3xi32>) dimensions = [0, 2]
      return %r : tensor<4x2x5x3xi32>
    }""",
    "reduce": """func.func @subject(%a0: tensor<2x3x4xi32>, %a1: tensor<3xi32>) -> tensor<3xi32> {
      %r = linalg.reduce ins(%a0 : tensor<2x3x4xi32>) outs(%a1 : tensor<3xi32>) dimensions = [0, 2]
        (%x: i32, %acc: i32) {
          %s = arith.addi %x, %acc : i32
          linalg.yield %s : i32
        }
      return %r : tensor<3xi32>
    }""",
    "map": """func.func @subject(%a0: tensor<2x3xi32>, %a1: tensor<2x3xi32>, %a2: tensor<2x3xi32>) -> tensor<2x3xi32> {
      %r = linalg.map ins(%a0, %a1 : tensor<2x3xi32>, tensor<2x3xi32>) outs(%a2 : tensor<2x3xi32>)
        (%x: i32, %y: i32, %init: i32) {
          %s = arith.subi %x, %y : i32
          linalg.yield %s : i32
        }
      return %r : tensor<2x3xi32>
    }""",
    "affine_generic_index": """func.func @subject(%a0: tensor<2x3xi32>, %a1: tensor<4x6xi32>) -> tensor<4x6xi32> {
      %r = linalg.generic {
        indexing_maps = [affine_map<(d0,d1)->(d0 floordiv 2,d1 mod 3)>, affine_map<(d0,d1)->(d0,d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%a0 : tensor<2x3xi32>) outs(%a1 : tensor<4x6xi32>) {
      ^bb0(%x: i32, %acc: i32):
        %i = linalg.index 0 : index
        %j = linalg.index 1 : index
        %ii = arith.index_cast %i : index to i32
        %jj = arith.index_cast %j : index to i32
        %s = arith.addi %x, %ii : i32
        %t = arith.subi %s, %jj : i32
        linalg.yield %t : i32
      } -> tensor<4x6xi32>
      return %r : tensor<4x6xi32>
    }""",
}


@pytest.mark.parametrize("name", sorted(STRUCTURED_PROGRAMS))
def test_structured_operations_and_named_generalization(tmp_path, fidelity_tools, name):

    source = STRUCTURED_PROGRAMS[name]
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    diff.compare(source, samples, tmp_path / "original", fidelity_tools)
    # Cross-check the transcribed named-op maps/body against MLIR's own generalization, then run
    # that generic form through both evaluators too. The named implementation is never our oracle.
    request = tmp_path / "generalize.json"
    request.write_text(json.dumps({"source": str(tmp_path / "signature.mlir"), "generalize": True}))
    proc = diff._run(
        [str(fidelity_tools.compiler_python), str(Path(diff.__file__).with_name("mlir_execute.py")), str(request)]
    )
    generic = json.loads(proc.stdout)["mlir"]
    diff.compare(generic, samples, tmp_path / "generalized", fidelity_tools)
    original = json.loads((tmp_path / "original/results.json").read_text())["lean"]
    generalized = json.loads((tmp_path / "generalized/results.json").read_text())["lean"]
    assert original == generalized


@pytest.mark.parametrize(
    "shape,element,inputs,mode",
    [
        ("2x3x", "i32", 0, "read"),
        ("2x3x", "i32", 1, "ignore"),
        ("2x3x", "i32", 2, "read"),
        ("2x3x", "i32", 2, "shorthand"),
        ("", "i32", 1, "read"),
        ("0x3x", "i32", 2, "read"),
        ("2x", "f32", 1, "read"),
        ("2x", "f32", 1, "ignore"),
        ("2x", "i32", 1, "capture"),
        ("2x", "i32", 1, "overwrite"),
    ],
)
def test_map_contract_and_native_generalization(tmp_path, fidelity_tools, shape, element, inputs, mode):
    ty = f"tensor<{shape}{element}>"
    arguments = [f"%a{i}: {ty}" for i in range(inputs)]
    if mode != "overwrite":
        arguments.append(f"%init: {ty}")
    ins = (
        ("ins(" + ", ".join(f"%a{i}" for i in range(inputs)) + " : " + ", ".join([ty] * inputs) + ")") if inputs else ""
    )
    block_args = ", ".join([f"%x{i}: {element}" for i in range(inputs)] + [f"%old: {element}"])
    prefix = f"%init = tensor.empty() : {ty}" if mode == "overwrite" else ""
    if mode == "capture":
        prefix += "\n%captured = arith.constant 7 : i32"
    if mode == "shorthand":
        payload, body = "{ arith.addi }", ""
    else:
        value = "%old" if mode == "read" else "%captured" if mode == "capture" else "%x0"
        calculation = ""
        if mode == "read" and inputs:
            operation = "arith.addf" if element == "f32" else "arith.addi"
            calculation = f"%sum = {operation} %old, %x0 : {element}"
            value = "%sum"
        payload, body = "", f"({block_args}) {{ {calculation} linalg.yield {value} : {element} }}"
    source = f"""func.func @subject({", ".join(arguments)}) -> {ty} {{
      {prefix}
      %r = linalg.map {payload} {ins} outs(%init : {ty}) {body}
      return %r : {ty}
    }}"""
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    arithmetic_nans = element == "f32" and mode == "read"
    diff.compare(source, samples, tmp_path / "map", fidelity_tools, arithmetic_nans=arithmetic_nans)
    request = tmp_path / "generalize.json"
    request.write_text(json.dumps({"source": str(tmp_path / "signature.mlir"), "generalize": True}))
    result = diff._run(
        [str(fidelity_tools.compiler_python), str(Path(diff.__file__).with_name("mlir_execute.py")), str(request)]
    )
    generic = json.loads(result.stdout)
    assert generic["map_contract"]["contract"] == "map-inputs-then-destination-v1"
    assert [c["body_arguments"] for c in generic["map_contract"]["cases"]] == [1, 2, 3]
    assert generic["map_contract"]["shorthand_arguments"] == 3
    assert len(generic["map_contract"]["mlir_capi"]["sha256"]) == 64
    diff.compare(generic["mlir"], samples, tmp_path / "generic", fidelity_tools, arithmetic_nans=arithmetic_nans)
    original = json.loads((tmp_path / "map/results.json").read_text())
    generalized = json.loads((tmp_path / "generic/results.json").read_text())
    assert original["lean"] == generalized["lean"]
    assert original["native"]["results"] == generalized["native"]["results"]


def test_map_generalization_with_closed_constant_motion(tmp_path, fidelity_tools):
    from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain

    source = """func.func @subject(%input: tensor<2xi32>) -> (tensor<2xi32>, i32) {
      %init = tensor.empty() : tensor<2xi32>
      %captured = arith.constant 7 : i32
      %r = linalg.map ins(%input : tensor<2xi32>) outs(%init : tensor<2xi32>)
        (%x: i32, %old: i32) {
          %sum = arith.addi %x, %captured : i32
          linalg.yield %sum : i32
        }
      return %r, %captured : tensor<2xi32>, i32
    }"""
    stage = tmp_path / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text(source)
    request = tmp_path / "generalize.json"
    request.write_text(json.dumps({"source": str(before), "generalize": True}))
    result = diff._run(
        [str(fidelity_tools.compiler_python), str(Path(diff.__file__).with_name("mlir_execute.py")), str(request)]
    )
    target = json.loads(result.stdout)["mlir"]
    after.write_text(target)
    # This must exercise actual operation motion, in addition to map expansion.
    assert source.index("tensor.empty") < source.index("arith.constant")
    assert target.index('"arith.constant"') < target.index('"tensor.empty"')
    (stage / "meta.json").write_text(json.dumps({"boundary": "map-and-constant-motion"}))
    compiler = fidelity_tools.compiler_python
    assert check_typed_stage(stage, m2m_python=compiler)["verdict"] == "verified"
    assert check_support(before, "subject", tmp_path / "support", m2m_python=compiler)["verdict"] == "verified"
    assert check_definedness(before, "subject", tmp_path / "defined", m2m_python=compiler)["verdict"] == "verified"
    chain = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "subject",
        tmp_path / "chain.json",
        m2m_python=compiler,
        definedness=tmp_path / "defined/receipt.json",
    )
    assert chain["verdict"] == "verified", chain.get("reason")
    samples = [[[0, 1]], [[0xFFFFFFFF, 0x7FFFFFFF]], [[0x80000000, 12345]]]
    expected = [[str((x + 7) % (1 << 32)) for x in sample[0]] + ["7"] for sample in samples]
    for name, program in [("before", source), ("after", target)]:
        work = tmp_path / name
        diff.compare(program, samples, work, fidelity_tools)
        results = json.loads((work / "results.json").read_text())
        assert results["native"]["results"] == expected
    # Changing the moved constant must still fail transformation verification.
    assert "value = 7 : i32" in target
    bad = tmp_path / "changed-constant"
    bad.mkdir()
    (bad / "before.mlir").write_text(source)
    (bad / "after.mlir").write_text(target.replace("value = 7 : i32", "value = 8 : i32"))
    (bad / "meta.json").write_text(json.dumps({"boundary": "changed-moved-constant"}))
    assert check_typed_stage(bad, m2m_python=compiler)["verdict"] == "abstracted"


def test_map_reading_poison_never_runs_natively(tmp_path, fidelity_tools, monkeypatch):
    source = """func.func @subject(%a: tensor<2xi32>) -> tensor<2xi32> {
      %init = tensor.empty() : tensor<2xi32>
      %r = linalg.map ins(%a : tensor<2xi32>) outs(%init : tensor<2xi32>)
        (%x: i32, %old: i32) { linalg.yield %old : i32 }
      return %r : tensor<2xi32>
    }"""

    def forbidden(*_args, **_kwargs):
        pytest.fail("map reading a dirty destination reached native execution")

    monkeypatch.setattr(diff, "execute_mlir", forbidden)
    with pytest.raises(AssertionError, match="independently defined"):
        diff.compare(source, [[[1, 2]]], tmp_path, fidelity_tools)


def test_map_initializer_copy_survives_bufferization(tmp_path, fidelity_tools):
    from merlin.llvmlower import pipeline
    from merlin.verify.lean.export import export_file

    # Protect both production defaults. The oracle prefix below also comes from its actual PASSES.
    for stages in [pipeline._upstream_pipeline(frozenset()), pipeline._parallel_pipeline(frozenset())]:
        assert stages.index("linalg-generalize-named-ops") < stages.index("one-shot-bufferize")
    source = tmp_path / "copy.mlir"
    # The store makes the initializer read observable; yielding the input prevents an identity-map
    # elimination from concealing the upstream missing-copy bug. This IR-only probe is never JITed.
    source.write_text("""func.func @probe(%buffer: memref<1xi16>) {
      %c0 = arith.constant 0 : index
      %input = arith.constant dense<0> : tensor<1xi16>
      %init = arith.constant dense<11> : tensor<1xi16>
      %r = linalg.map ins(%input : tensor<1xi16>) outs(%init : tensor<1xi16>)
        (%x: i16, %old: i16) {
          memref.store %old, %buffer[%c0] : memref<1xi16>
          linalg.yield %x : i16
        }
      return
    }""")
    request = tmp_path / "bufferize.json"
    request.write_text(json.dumps({"source": str(source), "bufferize_only": True}))
    result = diff._run(
        [str(fidelity_tools.compiler_python), str(Path(diff.__file__).with_name("mlir_execute.py")), str(request)]
    )
    answer = json.loads(result.stdout)
    bufferized, exported = tmp_path / "bufferized.mlir", tmp_path / "bufferized.json"
    bufferized.write_text(answer["mlir"])
    assert answer["pipeline"].index("linalg-generalize-named-ops") < answer["pipeline"].index("one-shot-bufferize")
    assert '"linalg.map"' not in answer["mlir"]
    assert answer["mlir"].count('"memref.copy"') == 1
    assert '"linalg.generic"' in answer["mlir"]
    assert '"memref.store"' in answer["mlir"]
    # Follow SSA edges: the copy must initialize this generic's destination from dense<11>,
    # and the observable store must consume this destination's block argument.
    export_file(bufferized, exported, backend="mlir", m2m_python=fidelity_tools.compiler_python)
    module_ops = json.loads(exported.read_text())["module"]["ops"]
    function = next(op for op in module_ops if op["name"] == "func.func")
    ops = function["regions"][0]["blocks"][0]["ops"]
    copy = next(op for op in ops if op["name"] == "memref.copy")
    generic = next(op for op in ops if op["name"] == "linalg.generic")
    assert copy["operands"][1] == generic["operands"][-1]
    getter = next(op for op in ops if any(r["id"] == copy["operands"][0] for r in op["results"]))
    assert getter["name"] == "memref.get_global"
    symbol = getter["attrs"]["name"]["root"]
    initializer = next(
        op for op in module_ops if op["name"] == "memref.global" and op["attrs"]["sym_name"]["value"] == symbol
    )
    assert initializer["attrs"]["initial_value"]["hex"] == "0b00"
    block = generic["regions"][0]["blocks"][0]
    store = next(op for op in block["ops"] if op["name"] == "memref.store")
    assert store["operands"][0] == block["args"][-1]["id"]


@pytest.mark.parametrize("width", [32, 64])
def test_float_reduction_order(tmp_path, fidelity_tools, width):
    ty = f"f{width}"
    source = f"""func.func @subject(%a0: tensor<1x4x{ty}>, %a1: tensor<4x1x{ty}>, %a2: tensor<1x1x{ty}>)
        -> tensor<1x1x{ty}> {{
      %r = linalg.matmul ins(%a0, %a1 : tensor<1x4x{ty}>, tensor<4x1x{ty}>)
        outs(%a2 : tensor<1x1x{ty}>) -> tensor<1x1x{ty}>
      return %r : tensor<1x1x{ty}>
    }}"""

    def b(x):
        return _float_bits(x, width)

    samples = [
        [[b(x) for x in [1e20, 1.0, -1e20, 4.0]], [b(1.0)] * 4, [b(0.0)]],
        [[b(x) for x in [1e20, -1e20, 1.0, 4.0]], [b(1.0)] * 4, [b(0.0)]],
    ]
    diff.compare(source, samples, tmp_path, fidelity_tools)


@pytest.mark.parametrize("operation,args", [("arith.divsi", [7, 0]), ("arith.shli", [1, 32])])
def test_undefined_or_poison_results_never_run_natively(tmp_path, fidelity_tools, monkeypatch, operation, args):
    source = _function(["i32", "i32"], [f"%r = {operation} %a0, %a1 : i32"], ["%r"])

    def forbidden(*_args, **_kwargs):
        pytest.fail("undefined/poison case reached native execution")

    monkeypatch.setattr(diff, "execute_mlir", forbidden)
    with pytest.raises(AssertionError, match="independently defined"):
        diff.compare(source, [[[x] for x in args]], tmp_path, fidelity_tools)


def test_unsupported_float_model_fails_explicitly(tmp_path, fidelity_tools):
    source = _function(["f32", "f32"], ["%r = arith.remf %a0, %a1 : f32"], ["%r"])
    with pytest.raises(RuntimeError, match="test float model does not implement arith.remf"):
        diff.compare(source, [[[_float_bits(7.0, 32)], [_float_bits(2.0, 32)]]], tmp_path, fidelity_tools)


def test_seeded_semantic_fault_is_detected(tmp_path, fidelity_tools):
    from merlin.verify.lean.export import export_file

    source = _function(["i32", "i32"], ["%r = arith.addi %a0, %a1 : i32"], ["%r"])
    diff.compare(source, [[[7], [2]]], tmp_path / "valid", fidelity_tools)
    # Actually evaluate a wrong model program, rather than fabricating its output or using an
    # expected constant. Native code still executes the original add, so a copied subtraction
    # error in the Lean model would be caught by precisely this comparison.
    mutant, exported = tmp_path / "mutant.mlir", tmp_path / "mutant.json"
    mutant.write_text(source.replace("arith.addi", "arith.subi"))
    export_file(mutant, exported, backend="mlir", m2m_python=fidelity_tools.compiler_python)
    wrong = diff.evaluate_lean(exported, [[["7"], ["2"]]], tmp_path, fidelity_tools)
    native = json.loads((tmp_path / "valid/results.json").read_text())["native"]["results"]
    with pytest.raises(AssertionError, match="Lean bits=.*native MLIR bits="):
        diff.assert_equal(wrong, native, [diff.ValueType("i32")], [[[7], [2]]])


@pytest.mark.parametrize("left,right", [(0x7FC00001, 0x7FC00002), (0, 0x80000000)])
def test_bit_observation_preserves_nan_payloads_and_signed_zero(left, right):
    result = [{"status": "ok", "values": [{"shape": None, "bits": [str(left)]}]}]
    with pytest.raises(AssertionError, match="native MLIR bits"):
        diff.assert_equal(result, [[str(right)]], [diff.ValueType("f32")], [[]])


def test_arithmetic_nan_policy_is_explicit():
    result = [{"status": "ok", "values": [{"shape": None, "bits": [str(0x7FC00001)]}]}]
    diff.assert_equal(result, [[str(0xFFC00002)]], [diff.ValueType("f32")], [[]], arithmetic_nans=True)
    with pytest.raises(AssertionError, match="native MLIR bits"):
        diff.assert_equal(result, [["0"]], [diff.ValueType("f32")], [[]], arithmetic_nans=True)


@pytest.mark.parametrize("ty", ["i32", "f32"])
def test_elementwise_tensor_arithmetic(tmp_path, fidelity_tools, ty):
    op = "arith.addi" if ty == "i32" else "arith.addf"
    source = _function([f"tensor<2x3x{ty}>"] * 2, [f"%r = {op} %a0, %a1 : tensor<2x3x{ty}>"], ["%r"])
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    diff.compare(source, samples, tmp_path, fidelity_tools, arithmetic_nans=ty == "f32")


def test_cli_native_comparison_is_empirical(tmp_path, fidelity_tools, capsys):
    from merlin.verify.cli import main

    program, samples, work = tmp_path / "input.mlir", tmp_path / "samples.json", tmp_path / "comparison"
    program.write_text(_function(["i32", "i32"], ["%r = arith.addi %a0, %a1 : i32"], ["%r"]))
    # Unflagged modular addition is defined for these integer inputs independently of Lean.
    samples.write_text(json.dumps([[[7], [2]], [[0xFFFFFFFF], [1]]]))
    command = [
        "lean",
        "differential",
        str(program),
        "--samples",
        str(samples),
        "--work",
        str(work),
        "--function",
        "subject",
        "--compiler-python",
        str(fidelity_tools.compiler_python),
        "--lean-exe",
        str(fidelity_tools.lean_exe),
        "--lake",
        str(fidelity_tools.lake),
        "--runner-utils",
        str(fidelity_tools.runner_utils),
    ]
    with pytest.raises(SystemExit) as missing_assumption:
        main(command)
    assert missing_assumption.value.code == 2 and not work.exists()
    capsys.readouterr()
    command.append("--assume-defined-inputs")
    assert main(command) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "matched_samples" and result["samples"] == 2
    assert "verdict" not in result
    assert json.loads((work / "comparison.json").read_text()) == result
    # Reuse cannot make an older matching artifact look like a new successful attempt.
    assert main(command) == 2
    assert json.loads(capsys.readouterr().out)["verdict"] == "unmeasured"


@pytest.mark.parametrize(
    "shape,element,target_element,dirty",
    [
        ((), "i32", None, True),
        ((0, 3), "i8", None, True),
        ((2, 3), "i13", None, True),
        ((2, 3), "i8", "i32", True),
        ((2, 3), "f32", None, True),
        ((2,), "f64", None, False),
    ],
)
def test_copy_generalization_native_and_independent_reference(
    tmp_path, fidelity_tools, shape, element, target_element, dirty
):
    from types import SimpleNamespace

    from test_lean_copy_generalize import _source
    from test_lean_generalize import _generalize

    source = _source(shape, element, target_element, dirty=dirty, escape=True).replace("@forward", "@subject")
    _, before, after = _generalize(tmp_path, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), source)
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    if element.startswith("f"):
        # Exercise exact data movement of NaN payloads, signed zero, infinities and subnormals.
        edges = _floats(int(element[1:]))
        edges.append(0x7FC01234 if element == "f32" else 0x7FF8000000001234)
    else:
        width = int(element[1:])
        edges = [0, 1, (1 << width) - 1, 1 << (width - 1)]
    samples.extend([[argument.copy() for argument in samples[0]] for _ in range(max(0, len(edges) - len(samples)))])
    for index, sample in enumerate(samples):
        sample[0] = [edges[(index + j) % len(edges)] for j in range(len(sample[0]))]
    for name, program in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(program.read_text(), samples, work, fidelity_tools)
        actual = json.loads((work / "native-results.json").read_text())["results"]
        expected = []
        for sample in samples:
            values = sample[0]
            if target_element is not None:
                # Default integer widening is signed; derive the reference directly from input bits.
                sign, modulus = 1 << (int(element[1:]) - 1), 1 << int(target_element[1:])
                copied = [str(((value ^ sign) - sign) % modulus) for value in values]
            else:
                copied = [str(value) for value in values]
            expected.append(copied + [str(value) for value in values])
        assert actual == expected


def test_copy_wrong_input_is_detected_by_native_execution(tmp_path, fidelity_tools):
    from types import SimpleNamespace

    from test_lean_copy_generalize import MUTATE, _source
    from test_lean_generalize import _generalize

    source = _source((2, 2), dirty=False).replace("@forward", "@subject")
    _, before, after = _generalize(tmp_path, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), source)
    samples = [[[7] * 4, [13] * 4, [19] * 4]]
    diff.compare(before.read_text(), samples, tmp_path / "original", fidelity_tools)
    diff._run([str(fidelity_tools.compiler_python), "-c", MUTATE, str(after), "input"])
    diff.compare(after.read_text(), samples, tmp_path / "mutated", fidelity_tools)
    original = json.loads((tmp_path / "original/native-results.json").read_text())["results"]
    mutated = json.loads((tmp_path / "mutated/native-results.json").read_text())["results"]
    assert original == [["7"] * 4] and mutated == [["13"] * 4]
