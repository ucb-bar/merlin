"""The Lean dequantization contract covers the quantization Merlin actually captures.

Every `quant_ext.dequantize_per_tensor` in the tracked capsules (i8 input, rank-0 f32 scale, rank-0
i64 zero point, quant ranges [-128, 127] and [-127, 127]) is lowered by `lower_quant_ext` and proved
against `MerlinVerify.Spec.Dequant` (PyTorch v2.10.0); wrong lowerings of the same ops are refused.
Needs the built checker and the gate tools; skips without them, or fails when ``MERLIN_LEAN_REQUIRED``.
"""

from __future__ import annotations

import collections

import pytest
from test_lean_pass_gate import job

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_xdsl import lower_quant_ext, parse_mlir_text
from merlin.verify import gate_checks
from merlin.verify.gate_errors import GateError
from merlin.xdsl_dialects._common import text

PER_TENSOR = '"quant_ext.dequantize_per_tensor"'


def captured_forms() -> tuple[list[tuple[str, str]], int]:
    """Each distinct (signature, properties) of a captured per-tensor dequantization, with one exact
    captured op line, and the number of captured ops. Provenance attributes are semantics-free."""
    forms: dict[tuple[str, str], str] = {}
    total = 0
    for path in sorted((repo_root() / "merlin/contract/capsules/model").glob("*/capsule.interface.mlir")):
        for line in path.read_text().splitlines():
            if PER_TENSOR + "(" not in line:
                continue
            total += 1
            head, _, signature = line.rpartition(" : ")
            properties = head[head.index("<{") : head.index("}>") + 2]
            forms.setdefault((signature.strip(), properties), line.strip())
    return [(signature, line) for (signature, _), line in forms.items()], total


def wrap(forms: list[tuple[str, str]]) -> str:
    """One function applying every form to its own arguments, exactly as captured."""
    params, body, results, types = [], [], [], []
    for i, (signature, line) in enumerate(forms):
        operands, _, result = signature.partition(" -> ")
        q, s, z = (t.strip() for t in operands.strip("()").split(", "))
        params += [f"%q{i}: {q}", f"%s{i}: {s}", f"%z{i}: {z}"]
        call = line[line.index(PER_TENSOR) :]
        call = f"{PER_TENSOR}(%q{i}, %s{i}, %z{i}){call[call.index(')') + 1 :]}"
        body.append(f"  %r{i} = {call}")
        results.append(f"%r{i}")
        types.append(result.strip())
    return (
        f"func.func @forward({', '.join(params)}) -> ({', '.join(types)}) {{\n"
        + "\n".join(body)
        + f"\n  return {', '.join(results)} : {', '.join(types)}\n}}\n"
    )


def lowered(source: str, mutate=None) -> tuple[str, str]:
    module = parse_mlir_text(source)
    before = text(module, generic=True)
    assert lower_quant_ext(module) > 0
    if mutate is not None:
        mutate(module)
    return before, text(module, generic=True)


def verdict(tmp_path, tools, before: str, after: str) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        gate_checks.tensor_refinement(job(tmp_path, before, after, tools))
        return "accepted"
    except GateError as error:
        return type(error).__name__


def test_every_captured_per_tensor_dequantization_is_proved(tmp_path, pass_gate_tools):
    forms, total = captured_forms()
    assert total >= 600 and len(forms) >= 80  # the capsules carry 627 such ops in 83 forms
    signatures = collections.Counter(signature.split(", ")[2].split(")")[0] for signature, _ in forms)
    assert set(signatures) == {"tensor<i64>"}
    before, after = lowered(wrap(forms))
    detail = gate_checks.tensor_refinement(job(tmp_path, before, after, pass_gate_tools))
    assert detail["claim"].startswith("strict refinement")


SMALL = (
    "(tensor<2x3xi8>, tensor<f32>, tensor<i64>) -> tensor<2x3xf32>",
    f"%r = {PER_TENSOR}(%q, %s, %z) <{{quant_min = -127 : i64, quant_max = 127 : i64}}> "
    '{prov.op = "dequantize_per_tensor"} : (tensor<2x3xi8>, tensor<f32>, tensor<i64>) -> tensor<2x3xf32>',
)


def _swap_subtraction(module):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    sub = next(op for op in module.walk() if isinstance(op, arith.SubfOp))
    Rewriter.replace_op(sub, arith.SubfOp(sub.rhs, sub.lhs))


def _ignore_zero_point(module):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    sub = next(op for op in module.walk() if isinstance(op, arith.SubfOp))
    Rewriter.replace_op(sub, arith.SubfOp(sub.lhs, sub.lhs))


def _add_scale(module):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    mul = next(op for op in module.walk() if isinstance(op, arith.MulfOp))
    Rewriter.replace_op(mul, arith.AddfOp(mul.lhs, mul.rhs))


@pytest.mark.parametrize("mutate", [_swap_subtraction, _ignore_zero_point, _add_scale])
def test_a_wrong_i64_per_tensor_lowering_is_refused(tmp_path, pass_gate_tools, mutate):
    source = wrap([SMALL])
    assert verdict(tmp_path / "right", pass_gate_tools, *lowered(source)) == "accepted"
    assert verdict(tmp_path / "wrong", pass_gate_tools, *lowered(source, mutate)) != "accepted"


PER_CHANNEL = (
    "func.func @forward(%w: tensor<4x3xi8>, %s: tensor<3xf32>, %z: tensor<3xZERO>) -> tensor<4x3xf32> {\n"
    '  %r = "quant_ext.dequantize_per_channel"(%w, %s, %z) <{axis = 1 : i64, RANGE}> '
    ": (tensor<4x3xi8>, tensor<3xf32>, tensor<3xZERO>) -> tensor<4x3xf32>\n"
    "  return %r : tensor<4x3xf32>\n}\n"
)


@pytest.mark.parametrize(
    ("zero", "low", "expected"),
    [
        ("i32", -128, "accepted"),
        ("i32", -127, "accepted"),  # PyTorch validates the range against i8 only; arithmetic unchanged
        ("i32", -129, "refused"),  # outside i8: PyTorch raises
        ("i64", -128, "refused"),  # per-channel i64 zero points stay outside the contract
    ],
)
def test_per_channel_contract_follows_pytorch(tmp_path, pass_gate_tools, zero, low, expected):
    source = PER_CHANNEL.replace("ZERO", zero).replace("RANGE", f"quant_min = {low} : i64, quant_max = 127 : i64")
    result = verdict(tmp_path, pass_gate_tools, *lowered(source))
    assert (result == "accepted") == (expected == "accepted"), result
