"""Quant-inner tensors reach the COMPILED path, and unwritten memory fails the build.

The defect these gate: a torchao capture parks a quantized subclass's inner tensors in
``extra.npz`` and leaves an uninitialized ``tensor.empty`` in the graph. The numpy interpreter binds
them while it walks the module; a compiled binary has no such moment, so it computed on whatever the
allocator last left there while the interpreter gated ``cos 1.0`` on the same bundle.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import qinner
from merlin.xdsl_dialects._common import text as to_text


def _module(*, tag: bool, shape_only: bool = False) -> str:
    """A one-op module whose second `ins` operand is an uninitialized `tensor.empty`.

    ``tag`` puts the quant-inner provenance on the consumer (what the capture emits);
    ``shape_only`` leaves the operand's block argument UNUSED, which is how a pooling window's
    extent is carried and must not be mistaken for data.
    """
    body = ("      %m = arith.mulf %a, %b : f32\n"
            "      linalg.yield %m : f32\n") if not shape_only else (
            "      %m = arith.mulf %a, %a : f32\n"
            "      linalg.yield %m : f32\n")
    attrs = ' attrs = {prov.quant_inner_1 = "fc.weight.tensor_impl.scale"}' if tag else ""
    return (
        "builtin.module {\n"
        "  func.func @forward(%arg0: tensor<4xf32>) -> tensor<4xf32> {\n"
        "    %e = tensor.empty() : tensor<4xf32>\n"
        "    %d = tensor.empty() : tensor<4xf32>\n"
        "    %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, "
        "affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "
        'iterator_types = ["parallel"]} '
        "ins(%arg0, %e : tensor<4xf32>, tensor<4xf32>) "
        f"outs(%d : tensor<4xf32>){attrs} {{\n"
        "    ^bb0(%a: f32, %b: f32, %c: f32):\n"
        f"{body}"
        "    } -> tensor<4xf32>\n"
        "    func.return %r : tensor<4xf32>\n"
        "  }\n"
        "}\n")


def test_gate_refuses_a_tensor_empty_that_reaches_computation():
    module = parse_mlir_text(_module(tag=True))
    findings = qinner.uninitialized_reads(module)
    assert len(findings) == 1, findings
    assert "quant-inner key" in findings[0]
    with pytest.raises(qinner.QinnerError, match="uninitialized"):
        qinner.require_initialized(module)


def test_gate_ignores_destinations_and_shape_only_operands():
    # `%d` is the generic's `outs` destination and `%e`'s block argument is never read: neither is
    # data the model computes on. Flagging either would refuse every DPS op and every pooling model.
    module = parse_mlir_text(_module(tag=False, shape_only=True))
    assert qinner.uninitialized_reads(module) == []
    qinner.require_initialized(module)


def test_lift_makes_the_tensor_an_argument_and_clears_the_gate():
    module = parse_mlir_text(_module(tag=True))
    appended = qinner.lift(module)
    assert [a.key for a in appended] == ["fc.weight.tensor_impl.scale"]
    assert appended[0].shape == (4,) and appended[0].dtype == "f32"
    assert qinner.uninitialized_reads(module) == []

    # the rewritten module must survive a print/parse round trip -- it is written to disk between
    # preparation and lowering -- and carry the appended argument in its function type.
    reparsed = parse_mlir_text(to_text(module))
    fn = next(op for op in reparsed.walk() if op.name == "func.func")
    assert len(fn.function_type.inputs.data) == 2
    assert len(fn.body.blocks[0].args) == 2


def test_plan_matches_what_lift_appends():
    # The compiled path derives the argument list from the BUNDLE while the object is built from the
    # lifted module. They are only consistent because both come from this one derivation.
    assert qinner.plan(parse_mlir_text(_module(tag=True))) == \
        qinner.lift(parse_mlir_text(_module(tag=True)))


def test_a_module_without_a_quantized_subclass_is_untouched():
    text = _module(tag=False)
    module = parse_mlir_text(text)
    before = to_text(module)
    assert qinner.lift(module) == []
    assert to_text(module) == before


def test_resolve_refuses_a_missing_or_mismatched_tensor():
    arg = qinner.QinnerArg("fc.scale", (4,), "f32")
    with pytest.raises(qinner.QinnerError, match="absent from extra.npz"):
        qinner.resolve({}, [arg])
    wrong = {"qinner::fc.scale": np.zeros((5,), np.float32)}
    with pytest.raises(qinner.QinnerError, match="in the IR"):
        qinner.resolve(wrong, [arg])
    right = {"qinner::fc.scale": np.arange(4, dtype=np.float32)}
    assert np.array_equal(qinner.resolve(right, [arg])[0], np.arange(4, dtype=np.float32))
