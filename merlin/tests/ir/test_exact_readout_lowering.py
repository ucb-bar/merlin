"""Plain exact rational readouts: extremes, ties, subnormals, executable arithmetic, literal and runtime biases."""

import json
import math
import random
import struct
from fractions import Fraction

import numpy as np
import pytest
from exact_readout_source import build_readout
from mlir_fixtures import native_library, run_block
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IntegerAttr,
    ModuleOp,
    StringAttr,
    TensorType,
    UnitAttr,
    f32,
    f64,
    i32,
    i64,
)
from xdsl.ir import Block, Region

from merlin.targetgen.exact_arithmetic import round_binary
from merlin.tests.ir.test_fixed_point_ir import reference_bits


def fixture(scales, *, size=513, floating=False, bias=False, **kwargs):
    denominator = math.lcm(*(s.denominator // (s.denominator & -s.denominator) for s in scales))
    output_scale = None if floating else float(denominator)
    block = Block(arg_types=[TensorType(i32, [size, len(scales)]), *([TensorType(f32, [len(scales)])] if bias else [])])
    ops, result = build_readout(
        block.args[0],
        [s * Fraction(output_scale or 1) for s in scales],
        bias=block.args[1] if bias else None,
        output_scale=output_scale,
        **kwargs,
    )
    reader = result.owner
    reader.attributes.update(
        {
            "prov.role": StringAttr("readout"),
            "prov.quant.arithmetic": StringAttr("exact"),
            "prov.quant.scale_numerator": ArrayAttr([StringAttr(str(s.numerator)) for s in scales]),
            "prov.quant.scale_denominator": ArrayAttr([StringAttr(str(s.denominator)) for s in scales]),
        }
    )
    if not floating:
        reader.attributes["prov.quant.output_scale"] = FloatAttr(output_scale, f64)
        for k, default in (("zero_point", 0), ("quant_min", -128), ("quant_max", 127)):
            reader.attributes[f"prov.quant.{k}"] = IntegerAttr(kwargs.get(k, default), i64)
        reader.attributes["prov.quant.relu"] = StringAttr(str(kwargs.get("relu", False)).lower())
        reader.attributes["prov.quant.storage_dtype"] = StringAttr("uint8" if kwargs.get("unsigned") else "int8")
    block.add_ops([*ops, func.ReturnOp(result)])
    fn = func.FuncOp("forward", ([v.type for v in block.args], [result.type]), Region(block))
    fn.attributes["llvm.emit_c_interface"] = UnitAttr()
    return ModuleOp([fn]), reader


@pytest.mark.parametrize(
    "scale",
    [
        Fraction(1, 2),
        Fraction(-1, 2),
        Fraction(1, 3),
        Fraction(123456789, 987654321),
        Fraction(0.03) * Fraction(0.02) / Fraction(0.01),
        Fraction(2) ** -100,
        Fraction(2) ** 60,
    ],
)
@pytest.mark.parametrize("zp,lo,hi,relu", [(0, -128, 127, False), (7, -127, 127, True), (131, 0, 255, False)])
def test_emitted_requantizer_exact_at_extremes_ties_and_random(scale, zp, lo, hi, relu):
    m, r = fixture([scale], zero_point=zp, quant_min=lo, quant_max=hi, relu=relu, unsigned=lo == 0)
    m.verify()
    rng = random.Random(421)
    points = {-(1 << 31), (1 << 31) - 1, 0, -1, 1}
    points.update(rng.randrange(-(1 << 31), 1 << 31) for _ in range(200))
    for k in range(lo - zp - 1, hi - zp + 2):
        x = int(Fraction(2 * k + 1, 2) / scale)
        points.update(range(x - 2, x + 3))
    for v in points:
        if not -(1 << 31) <= v < 1 << 31:
            continue
        code = round(v * scale) + zp
        if relu:
            code = max(zp, code)
        assert run_block(r.body.block, [v & 0xFFFFFFFF, 0]) == min(hi, max(lo, code)) & 255


def test_complete_small_numerator_domain():
    scale = Fraction(23, 98)
    m, r = fixture([scale])
    m.verify()
    for v in range(-32768, 32769):
        assert run_block(r.body.block, [v & 0xFFFFFFFF, 0]) == min(127, max(-128, round(v * scale))) & 255


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "scale",
    [
        Fraction(1, 2),
        Fraction(-0.03) * Fraction(0.02),
        Fraction(float(np.float32(-0.03))) * Fraction(float(np.float32(0.02))),
        Fraction(2) ** -149,
        Fraction(2) ** 110,
    ],
)
def test_float_readout_bits(scale, bias):
    m, r = fixture([scale], floating=True, bias=bias)
    m.verify()
    rng = random.Random(731)
    inputs = [-(1 << 31), (1 << 31) - 1, 0, -1, 1] + [rng.randrange(-(1 << 31), 1 << 31) for _ in range(100)]
    for v in inputs:
        bs = (
            rng.choice([0, 1, 0x7F7FFFFF, 0x80000000, 0x00800000, 0x80000001, 0xBF800000, rng.randrange(0x7F800000)])
            if bias
            else 0
        )
        bv = struct.unpack("<f", bs.to_bytes(4, "little"))[0]
        exact = v * scale + Fraction(bv)
        expected = reference_bits(exact.numerator, 0, exact.denominator, "f32")
        args = [v & 0xFFFFFFFF, *([bs] if bias else []), 0]
        assert run_block(r.body.block, args) == expected, (v, bs, scale)


@pytest.mark.parametrize("floating,bias", [(False, False), (True, False), (True, True)])
def test_native_readout_against_rational_reference(tmp_path, floating, bias):
    scales = [
        Fraction(float(np.float32(0.03))) * Fraction(float(np.float32(0.02))),
        Fraction(float(np.float32(-0.017))),
        Fraction(1, 8),
    ]
    if not floating:
        scales = [s / Fraction(0.01) for s in scales]
    module, reader = fixture(scales, floating=floating, bias=bias)
    run = native_library(module, tmp_path)
    rng = np.random.default_rng(141)
    data = rng.integers(-(1 << 31), 1 << 31, (513, 3), dtype=np.int32)
    data[:7] = np.array([-(1 << 31), (1 << 31) - 1, -1, 0, 1, 2, 3], dtype=np.int32)[:, None]
    biases = np.array([0.1, 1e-40, -1e20], dtype=np.float32)
    expected = np.empty(data.shape, dtype=np.float32 if floating else np.int8)
    for ix in np.ndindex(data.shape):
        exact = int(data[ix]) * scales[ix[-1]] + (Fraction(float(biases[ix[-1]])) if bias else 0)
        expected[ix] = round_binary(exact) if floating else min(127, max(-128, round(exact)))
    actual = np.zeros_like(expected)
    run(data, *([biases] if bias else []), actual)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("floating", [False, True])
def test_captured_trace_annotations_do_not_change_scalar_semantics(floating):
    module, reader = fixture([Fraction(1, 2)], floating=floating)
    for inner in reader.body.walk():
        inner.attributes.update(
            {
                "prov.source_node_ids": ArrayAttr([StringAttr("source:5")]),
                "prov.origin_node_ids": ArrayAttr([StringAttr("origin:2")]),
                "prov.trace_role": StringAttr("lowering"),
            }
        )
    module.verify()


def freeze(module, reader, tmp_path, values, literal):
    values = np.asarray(values, np.float32)
    if literal:
        old = reader.inputs[1]
        const = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(old.type, values.tolist()))
        reader.parent_block().insert_op_before(const, reader)
        reader.operands = tuple(const.result if v is old else v for v in reader.operands)
    else:
        data = values.tobytes()
        header = json.dumps({"bias": {"dtype": "F32", "shape": [len(values)], "data_offsets": [0, len(data)]}}).encode()
        path = tmp_path / "weights.safetensors"
        path.write_bytes(len(header).to_bytes(8, "little") + header + data)
        path.with_suffix(".safetensors.manifest.json").write_text(
            json.dumps({"1": dict(weight="bias", kind="param", dtype="float32", shape=[len(values)])})
        )
        module.attributes["prov.weights_file"] = StringAttr(str(path))


@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("literal", [False, True])
def test_frozen_per_channel_bias_native_and_changed_parameter(tmp_path, floating, literal):
    scales = [Fraction(1, 2), Fraction(17, 31) if not floating else Fraction(float(np.float32(0.03))), Fraction(-3, 8)]
    module, reader = fixture(scales, size=129, bias=True, floating=floating)
    biases = np.array([0.0, -0.25, 1.0], np.float32)
    freeze(module, reader, tmp_path, biases, literal)
    module.verify()
    rng = np.random.default_rng(443)
    x = rng.integers(-(1 << 31), 1 << 31, (129, 3), dtype=np.int32)
    x[:7] = np.asarray([-(1 << 31), (1 << 31) - 1, -1, 0, 1, 2, 3], np.int32)[:, None]
    output = np.zeros(x.shape, dtype=np.float32 if floating else np.int8)
    run = native_library(module, tmp_path)
    output_scale = Fraction(1) if floating else Fraction(31)
    for mutate in (False, True):
        if mutate:
            biases[:] = [0.125, 8.0, -1.0]
        run(x, biases, output)
        effective = np.array([0.0, -0.25, 1.0], np.float32) if literal else biases
        expected = np.empty_like(output)
        for ix in np.ndindex(x.shape):
            real = int(x[ix]) * scales[ix[-1]] + Fraction(float(effective[ix[-1]])) / output_scale
            expected[ix] = round_binary(real) if floating else min(127, max(-128, round(real)))
        assert output.tobytes() == expected.tobytes()
