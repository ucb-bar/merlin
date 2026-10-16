"""First lowering obligation: the scalar body, with independently specified arithmetic order.

These tests deliberately do not report the enclosing tensor lowering as verified. The current
per-channel body must be refused until its operation order is corrected in the compiler.
"""

from __future__ import annotations

import copy
import json
import struct
import subprocess

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import lower_quant_ext
from merlin.verify.lean.export import MLIR_EXPORT, export_file


def lowered(kind, shape=(2,), axis=0):
    quantized_shape = "".join(f"{extent}x" for extent in shape)
    parameter_shape = "" if kind == "per_tensor" else f"{shape[axis]}x"
    attrs = "" if kind == "per_tensor" else f"axis = {axis} : i64"
    module = parse_mlir_text(f"""func.func @forward(%q: tensor<{quantized_shape}i8>,
        %s: tensor<{parameter_shape}f32>, %z: tensor<{parameter_shape}i32>) -> tensor<{quantized_shape}f32> {{
      %r = "quant_ext.dequantize_{kind}"(%q,%s,%z) <{{{attrs}}}> :
        (tensor<{quantized_shape}i8>, tensor<{parameter_shape}f32>, tensor<{parameter_shape}i32>)
          -> tensor<{quantized_shape}f32>
      return %r : tensor<{quantized_shape}f32>
    }}""")
    assert lower_quant_ext(module) == 1
    module.verify()
    return module, next(op for op in module.walk() if op.name == "linalg.generic")


def corrected_channel_body(generic):
    # Proposed replacement only: the production pass remains the subject under examination.
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import f32, i32
    from xdsl.dialects.linalg.ops import YieldOp

    block = generic.body.block
    for op in reversed(list(block.ops)):
        block.erase_op(op)
    promoted = arith.ExtSIOp(block.args[0], i32)
    diff = arith.SubiOp(promoted.result, block.args[2])
    converted = arith.SIToFPOp(diff.result, f32)
    scaled = arith.MulfOp(converted.result, block.args[1])
    block.add_ops([promoted, diff, converted, scaled, YieldOp(scaled.result)])


def export_module(module, root, tools):
    source, target = root / "lowered.mlir", root / "lowered.json"
    source.write_text(str(module))
    export_file(source, target, backend="mlir", m2m_python=tools.m2m_python)
    return target


def check_body(document, kind):
    driver = repo_root() / "merlin" / "tests" / "fixtures" / "lean_dequant_scalar.lean"
    proc = subprocess.run(
        ["lake", "env", "lean", "--run", str(driver), str(document), kind],
        cwd=repo_root() / "merlin" / "lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stderr
    assert proc.stdout.startswith("{"), proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["scope"] == "scalar_body_only"
    assert (proc.returncode == 0) == result["matches"]
    return result["matches"]


@pytest.mark.parametrize("kind", ["per_tensor", "per_channel"])
def test_actual_lowering_scalar_body(tmp_path, lean_tools, kind):
    module, _ = lowered(kind)
    document = export_module(module, tmp_path, lean_tools)
    assert check_body(document, kind) == (kind == "per_tensor")


def test_integer_subtraction_channel_body_matches_proof(tmp_path, lean_tools):
    module, generic = lowered("per_channel")
    corrected_channel_body(generic)
    module.verify()
    assert check_body(export_module(module, tmp_path, lean_tools), "per_channel")


@pytest.mark.parametrize("mutation", ["subtract_order", "scale_wiring", "fastmath", "conversion", "result", "extra_op"])
def test_body_checker_rejects_mutations(tmp_path, lean_tools, mutation):
    module, _ = lowered("per_tensor")
    document = export_module(module, tmp_path, lean_tools)
    data = json.loads(document.read_text())
    generic = next(
        op for op in data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"] if op["name"] == "linalg.generic"
    )
    block = generic["regions"][0]["blocks"][0]
    ops = block["ops"]
    if mutation == "subtract_order":
        ops[2]["operands"].reverse()
    elif mutation == "scale_wiring":
        ops[3]["operands"][1] = block["args"][3]["id"]
    elif mutation == "fastmath":
        ops[2]["attrs"]["fastmath"]["value"] = "reassoc"
    elif mutation == "conversion":
        ops[0]["name"] = "arith.uitofp"
    elif mutation == "result":
        ops[-1]["operands"] = ops[2]["operands"][:1]
    else:
        ops.append(copy.deepcopy(ops[-1]))
    document.write_text(json.dumps(data))
    assert not check_body(document, "per_tensor")


def scalarize(generic, q, zero, scale):
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, ModuleOp, f32, i8, i32
    from xdsl.ir import Region

    block = generic.body.detach_block(0)
    constants = [
        arith.ConstantOp(IntegerAttr(q, i8)),
        arith.ConstantOp(FloatAttr(scale, f32)),
        arith.ConstantOp(IntegerAttr(zero, i32)),
        arith.ConstantOp(FloatAttr(-999.0, f32)),
    ]
    for arg, constant in zip(block.args, constants, strict=True):
        arg.replace_all_uses_with(constant.results[0])
    for arg in reversed(list(block.args)):
        block.erase_arg(arg)
    last = block.last_op
    returned = tuple(last.operands)
    block.erase_op(last)
    block.add_op(func.ReturnOp(*returned))
    block.insert_ops_before(constants, block.first_op)
    module = ModuleOp([func.FuncOp.from_region("forward", [], [f32], Region(block))])
    module.verify()
    return module


CASES = [
    (1, 16777217, 1.0),
    (-128, 0, 0.125),
    (127, -3, -0.5),
    (-128, -(1 << 31), 1.0),
    (1, (1 << 31) - 1, 1.0),
    (0, 0, -0.0),
]


def folded_bits(module, tmp_path, lean_tools):
    source, destination = tmp_path / "scalar.mlir", tmp_path / "folded.json"
    source.write_text(str(module))
    adapter = repo_root() / "merlin" / "tests" / "ir" / "lean_mlir_oracle.py"
    proc = subprocess.run(
        [
            str(lean_tools.m2m_python),
            str(adapter),
            str(MLIR_EXPORT),
            str(source),
            str(destination),
            "builtin.module(canonicalize)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    ops = json.loads(destination.read_text())["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    returned = ops[-1]["operands"][0]
    folded = next(op for op in ops if any(r["id"] == returned for r in op["results"]))
    assert folded["name"] == "arith.constant"
    return int(folded["attrs"]["value"]["bits"], 16)


@pytest.mark.parametrize("q,zero,scale", CASES)
@pytest.mark.parametrize("channel", [False, True])
def test_scalar_arithmetic_against_mlir_folder(tmp_path, lean_tools, q, zero, scale, channel):
    _, generic = lowered("per_channel" if channel else "per_tensor")
    if channel:
        corrected_channel_body(generic)
    module = scalarize(generic, q, zero, scale)
    if channel:
        signed_difference = ((q - zero + (1 << 31)) % (1 << 32)) - (1 << 31)
        expected = np.float32(signed_difference) * np.float32(scale)
    else:
        expected = (np.float32(q) - np.float32(zero)) * np.float32(scale)
    expected_bits = struct.unpack("<I", struct.pack("<f", float(expected)))[0]
    assert folded_bits(module, tmp_path, lean_tools) == expected_bits


def test_current_per_channel_large_zero_point_counterexample(tmp_path, lean_tools):
    _, generic = lowered("per_channel")
    actual = folded_bits(scalarize(generic, 1, 16777217, 1.0), tmp_path, lean_tools)
    current_bits = struct.unpack("<I", struct.pack("<f", -16777215.0))[0]
    source_bits = struct.unpack("<I", struct.pack("<f", -16777216.0))[0]
    assert actual == current_bits
    assert actual != source_bits
