"""Qualify the explicit RNE underflow branch against the selected scalar libm."""

from __future__ import annotations

import ctypes
import subprocess
from decimal import Decimal, localcontext

import numpy as np
import pytest
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from merlin.llvmlower import toolchain
from merlin.llvmlower.cpu_softmax import build_libm_softmax_exp_rne


def test_cutoff_is_strictly_below_half_minimum_subnormal():
    with localcontext() as ctx:
        ctx.prec = 100
        assert Decimal(-104).exp() < Decimal(2) ** -150


@pytest.mark.parametrize(
    "changes",
    [
        {"domain": "unknown"},
        {"rounding_mode": "upward"},
        {"floating_effects": "observed"},
    ],
)
def test_requires_explicit_domain_and_effect_contract(changes):
    value = Block(arg_types=[builtin.f32]).args[0]
    options = dict(domain="nonpositive", rounding_mode="rne", floating_effects="unobserved")
    with pytest.raises(ValueError):
        build_libm_softmax_exp_rne(value, **(options | changes))


def test_refuses_non_f32():
    value = Block(arg_types=[builtin.f64]).args[0]
    with pytest.raises(TypeError):
        build_libm_softmax_exp_rne(value, domain="nonpositive", rounding_mode="rne", floating_effects="unobserved")


def underflow_module():
    block = Block(arg_types=[builtin.f32])
    ops, result = build_libm_softmax_exp_rne(
        block.args[0],
        domain="nonpositive",
        rounding_mode="rne",
        floating_effects="unobserved",
    )
    block.add_ops([*ops, func.ReturnOp(result)])
    module = builtin.ModuleOp([func.FuncOp("evaluate", ([builtin.f32], [builtin.f32]), Region(block))])
    module.verify()
    return module


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_native_return_bits_match_libm(tmp_path, optimization):
    if not toolchain.available():
        pytest.skip("native toolchain unavailable")
    from merlin.llvmlower.entry_weight_projection import module_text
    from merlin.llvmlower.lower import lower_model_file

    source = tmp_path / "model.mlir"
    source.write_text(module_text(underflow_module()))
    lowered = lower_model_file(source, tmp_path / "lower", targets=(), textual=True)
    shared = tmp_path / "model.so"
    subprocess.run(
        [str(toolchain.clang()), optimization, "-shared", "-fPIC", str(lowered.ll_path), "-lm", "-o", str(shared)],
        check=True,
        capture_output=True,
    )
    compiled = ctypes.CDLL(str(shared)).evaluate
    reference = ctypes.CDLL("libm.so.6").expf
    for f in [compiled, reference]:
        f.argtypes = [ctypes.c_float]
        f.restype = ctypes.c_float
    edges = np.array(
        [-np.inf, -np.finfo(np.float32).max, -104.0, -103.972084, -103.9, -87.3, -1.0, -0.0, 0.0], np.float32
    )
    with np.errstate(over="ignore"):
        values = np.concatenate(
            [
                edges,
                np.nextafter(edges, np.float32(-np.inf)),
                np.nextafter(edges, np.float32(0)),
                -np.random.default_rng(21).uniform(0, 150, 2048).astype(np.float32),
            ]
        )
    observed = np.array([compiled(float(v)) for v in values], np.float32)
    expected = np.array([reference(float(v)) for v in values], np.float32)
    np.testing.assert_array_equal(observed.view("u4"), expected.view("u4"))
    assert "fcmp ole" in lowered.ll_path.read_text()
