"""Execute generated host limb arithmetic at native wraparound boundaries."""

import json

import numpy as np
import pytest
from mlir_fixtures import assert_unchanged, forward, generic, limbs, pointwise, run_ops, signed
from xdsl.dialects.builtin import IntegerType
from xdsl.ir.affine import AffineDimExpr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.boolean_components import rewrite_boolean_components
from merlin.llvmlower.integer_components import rewrite_integer_components
from merlin.llvmlower.integer_width_boundaries import preserve_integer_width_boundaries
from merlin.llvmlower.native_sign_components import rewrite_native_sign_components


def module(dtype, expression):
    typ = f"tensor<4x{dtype}>"
    return pointwise([typ, typ], typ, expression)


def evaluate(module, arrays):
    """Small integer tensor interpreter over storage words; all generated tensors have identity maps."""
    for op in module.walk():
        assert op.name != "arith.cmpi" or op.predicate.value.data in (1, 2, 8)
        assert op.name != "arith.cmpf" or (str(op.operands[0].type), op.predicate.value.data) == ("f16", 4)
    function = next(op for op in module.walk() if op.name == "func.func")
    values = {
        arg: np.frompyfunc(lambda v, t=arg.type.element_type: word(v, t), 1, 1)(np.array(a, dtype=object))
        for arg, a in zip(function.body.block.args, arrays)
    }
    for op in function.body.block.ops:
        if op.name == "func.return":
            typ = op.operands[0].type.element_type
            convert = signed if isinstance(typ, IntegerType) else lambda v, t: v
            return np.frompyfunc(lambda v: convert(v, typ), 1, 1)(values[op.operands[0]]).tolist()
        if op.name == "arith.constant":
            run_ops([op], values)
        elif op.name == "tensor.empty":
            values[op.results[0]] = np.zeros(op.results[0].type.get_shape(), object)
        elif op.name == "tensor.splat":
            values[op.results[0]] = np.full(op.results[0].type.get_shape(), values[op.operands[0]], object)
        elif op.name == "linalg.generic":
            output = values[op.outputs[0]].copy()
            maps = [attr.data for attr in op.get_indexing_maps()]
            extents = [None] * len(op.get_iterator_types())
            for value, amap in zip(op.operands, maps):
                for extent, expr in zip(value.type.get_shape(), amap.results):
                    assert isinstance(expr, AffineDimExpr)
                    previous = extents[expr.position]
                    assert previous is None or previous == extent
                    extents[expr.position] = extent
            assert all(n is not None for n in extents)
            for index in np.ndindex(tuple(extents)):
                output_index = tuple(maps[-1].eval(index, ()))
                env = {
                    arg: (
                        output[output_index] if i == len(op.inputs) else values[source][tuple(maps[i].eval(index, ()))]
                    )
                    for i, (arg, source) in enumerate(zip(op.body.block.args, op.operands))
                }
                output[output_index] = run_ops(op.body.block.ops, env)
            values[op.results[0]] = output
        else:
            raise AssertionError(op.name)
    raise AssertionError("no return")


def word(value, typ):
    return int(value) % (1 << typ.width.data) if isinstance(typ, IntegerType) else value


@pytest.mark.parametrize("bits", [8, 32, 64])
@pytest.mark.parametrize("coefficient", [-3, 1, 7])
def test_linear_components_recombine_with_native_wraparound(bits, coefficient):
    source = module(
        f"i{bits}",
        f"%s = arith.constant {coefficient} : i{bits}\n"
        f"%p = arith.muli %a, %s : i{bits}\n%v = arith.addi %p, %b : i{bits}",
    )
    limit = 1 << (bits - 1)
    x, y = [-limit, limit - 1, -129 % limit, 128 % limit], [1, -1, 127, -128]
    plans = limbs(source, [limit, 128])
    assert plans and all(p["radix"] == 256 for p in plans)
    assert all(
        json.loads(op.attributes["test.limb"].data)["family"] == "F2"
        for op in source.walk()
        if "test.limb" in op.attributes
    )
    expected = [((coefficient * a + b + limit) % (2 * limit)) - limit for a, b in zip(x, y)]
    assert evaluate(source, [x, y]) == expected


def test_boolean_or_stays_in_the_host_source():
    source = module("i1", "%v = arith.ori %a, %b : i1")
    original = str(source)
    bounds, count = rewrite_boolean_components(source)
    assert count == 0 and bounds == {}
    assert str(source) == original
    assert not rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[1, 1], known_bounds=bounds)
    assert str(source) == original


def test_widening_keeps_the_carry_above_the_original_storage_width():
    source = forward(
        "%x:tensor<4xi64>,%y:tensor<4xi64>",
        "tensor<4xi128>",
        generic([("%x", "tensor<4xi64>")], "tensor<4xi128>", "%v = arith.extsi %a : i64 to i128"),
    )
    values = [(1 << 63) - 1, -(1 << 63), 32767, -32768]
    plans = limbs(source, [1 << 63, 0])
    assert len(plans) == 8  # Source-width storage is normalized before the signed extension.
    assert evaluate(source, [values, [0] * 4]) == values


def test_native_sign_transform_keeps_all_ieee_payload_bits_in_integer_literals():
    source = pointwise(["tensor<4xf16>"], "tensor<4xf16>", "%v = arith.negf %a : f16")
    bounds, proofs = rewrite_native_sign_components(source)
    assert proofs == [
        dict(
            source="arith.negf",
            representation="ieee_storage_bits",
            width=16,
            arithmetic="sign_bit_offset_modulo_storage_width",
            nan_payload="preserved",
        )
    ]
    assert not any(op.name == "arith.negf" for op in source.walk())
    limbs(source, [0], bounds)
    for words in [[0, 0x8000, 0x7C00, 0xFC00], [0x7E01, 0xFE7F, 0x7C01, 0xFC01], [0x0001, 0x8001, 0x3C00, 0xBC00]]:
        assert [value & 0xFFFF for value in evaluate(source, [words])] == [word ^ 0x8000 for word in words]


@pytest.mark.parametrize("dtype,predicate,zero", [("f16", "olt", "0.0"), ("i64", "slt", "0")])
def test_clamp_epilogue_preserves_source_comparison_and_payloads(dtype, predicate, zero):
    cmp = "cmpf" if dtype == "f16" else "cmpi"
    source = pointwise(
        [f"tensor<4x{dtype}>"],
        f"tensor<4x{dtype}>",
        f"""%zero=arith.constant {zero} : {dtype}
        %p=arith.{cmp} {predicate},%a,%zero : {dtype}
        %v=arith.select %p,%zero,%a : {dtype}""",
    )
    bounds, proofs = rewrite_native_sign_components(source)
    assert proofs[0]["arithmetic"] == "source_compare_select_retained_host"
    limbs(source, [1 << 63], bounds)
    if dtype == "f16":
        for words, expected in [
            ([0, 0x8000, 0x7C00, 0xFC00], [0, 0x8000, 0x7C00, 0]),
            ([0x7E01, 0xFE7F, 0x7C01, 0xFC01], [0x7E01, 0xFE7F, 0x7C01, 0xFC01]),
            ([0x0001, 0x8001, 0x3C00, 0xBC00], [0x0001, 0, 0x3C00, 0]),
        ]:
            assert [v & 0xFFFF for v in evaluate(source, [words])] == expected
    else:
        assert evaluate(source, [[-(1 << 63), -1, 0, (1 << 63) - 1]]) == [0, 0, 0, (1 << 63) - 1]


def test_window_reduction_ignores_unread_shape_operand():
    prelude = "%shape = tensor.empty() : tensor<2xi64>\n%z = arith.constant 0 : i64\n"
    prelude += "%seed = tensor.splat %z : tensor<2xi64>\n"
    maps = [f"affine_map<(d0,d1)->({m})>" for m in ("d0*2+d1", "d1", "d0")]
    ins = [("%x", "tensor<4xi64>"), ("%shape", "tensor<2xi64>")]
    window = generic(
        ins,
        "tensor<2xi64>",
        "%v = arith.addi %a, %acc : i64",
        maps=maps,
        iterators=["parallel", "reduction"],
        out="%seed",
        acc="%acc",
    )
    source = forward("%x: tensor<4xi64>", "tensor<2xi64>", prelude + window)
    limbs(source, [128])
    for op in source.walk():
        if "test.limb" in op.attributes:
            assert len(op.inputs) == 2
            assert len(op.get_indexing_maps()) == 3
            assert not op.body.block.args[1].uses
            plan = json.loads(op.attributes["test.limb"].data)
            assert {entry["operand"] for entry in plan["inputs"]} == {0, 1}
            data = next(entry for entry in plan["inputs"] if entry["operand"] == 0)
            assert next(entry for entry in plan["inputs"] if entry["operand"] == 1) == dict(
                operand=1, index=0, count=1, exponent=8 * data["index"] + data["exponent"]
            )


def test_declared_bfloat_nan_negation_uses_device_words_then_cpu_canonicalization():
    source = pointwise(["tensor<4xbf16>"], "tensor<4xbf16>", "%v=arith.negf %a : bf16")
    bounds, proofs = rewrite_native_sign_components(source, canonical_bf16_nan=True)
    assert proofs[0]["nan_payload"] == "canonical_positive_quiet"
    limbs(source, [0], bounds)
    for payload in range(1, 128):
        words = [0x7F80 | payload, 0xFF80 | payload, 0, 0x8000]
        assert [value & 0xFFFF for value in evaluate(source, [words])] == [0x7FC0, 0x7FC0, 0x8000, 0]


def test_binary32_sign_words_preserve_nonfinite_payloads_without_quantization():
    source = pointwise(["tensor<4xf32>"], "tensor<4xf32>", "%v=arith.negf %a : f32")
    bounds, proofs = rewrite_native_sign_components(source)
    assert proofs[0]["nan_payload"] == "preserved"
    limbs(source, [0], bounds)
    for words in [
        [0, 0x80000000, 0x7F800000, 0xFF800000],
        [0x7FC00001, 0xFFC00012, 0x7F800001, 0xFF800001],
        [1, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF],
    ]:
        assert [v & 0xFFFFFFFF for v in evaluate(source, [words])] == [v ^ 0x80000000 for v in words]


@pytest.mark.parametrize("bits", [8, 16, 32])
@pytest.mark.parametrize("cast", ["extui", "extsi"])
@pytest.mark.parametrize("wrap", [False, True])
def test_width_changes_observe_the_source_ring(bits, cast, wrap):
    limit = 1 << (bits - 1)
    body = f"%s=arith.addi %a,%b : i{bits}\n" if wrap else ""
    body += f"%v=arith.{cast} " + ("%s" if wrap else "%a") + f" : i{bits} to i64"
    source = pointwise([f"tensor<4xi{bits}>"] * 2, "tensor<4xi64>", body)
    values = [-1, -limit, 0, limit - 1]
    other = [1, -1, limit - 1, 1]
    expected = evaluate(source, [values, other])
    limbs(source, [limit, limit])
    assert evaluate(source, [values, other]) == expected
    if not wrap and cast == "extui":
        assert expected == [(v % (1 << bits)) for v in values]
    if wrap and cast == "extsi":
        assert expected[-1] == -limit


@pytest.mark.parametrize("bits", [8, 16, 32])
def test_truncation_before_signed_extension_keeps_its_modular_boundary(bits):
    source = module("i64", f"%t=arith.trunci %a : i64 to i{bits}\n%v=arith.extsi %t : i{bits} to i64")
    values = [-(1 << 63), (1 << bits) - 1, (1 << (bits - 1)), (1 << (bits - 1)) - 1]
    expected = evaluate(source, [values, [0] * 4])
    limbs(source, [1 << 63, 0])
    assert evaluate(source, [values, [0] * 4]) == expected


def test_real_frontend_uint8_cast_review_repro():
    torch = pytest.importorskip("torch")
    m2m = pytest.importorskip("m2m")

    class Cast(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten._to_copy.default(x, dtype=torch.int32)

    values = torch.tensor([255, 128, 0, 127], dtype=torch.uint8)
    original = parse_mlir_text(str(m2m.convert(Cast(), (values,), backend="fx_importer", fully_standard=True).module))
    limbs(original, [128])
    assert evaluate(original, [[-1, -128, 0, 127]]) == [255, 128, 0, 127]


def test_late_width_pass_does_not_split_generated_quotient_plumbing():
    source = module(
        "i64", "%q=arith.divsi %a,%b : i64\n%t=arith.trunci %q : i64 to i32\n%v=arith.extsi %t : i32 to i64"
    )
    assert_unchanged(source, preserve_integer_width_boundaries)


@pytest.mark.parametrize("coefficient", [(1 << 40) + 3, -(1 << 63) + 1, (1 << 63) - 1])
def test_full_width_scalar_coefficients_use_exact_literal_planes(coefficient):
    source = module("i64", f"%s=arith.constant {coefficient}:i64\n%v=arith.muli %a,%s:i64")
    limit = 1 << 63
    values = [-limit, limit - 1, -129, 128]
    limbs(source, [limit, limit])
    expected = [((coefficient * a + limit) % (2 * limit)) - limit for a in values]
    assert evaluate(source, [values, [0] * 4]) == expected


@pytest.mark.parametrize("bits", [8, 32, 64])
def test_named_sum_exposes_full_width_limb_reduction_and_modular_wrap(bits):
    source = parse_mlir_text(
        """module {
      func.func @forward(%x:tensor<3x4xTYPE>) -> tensor<4xTYPE> {
        %z=arith.constant 0:TYPE
        %s=tensor.splat %z:tensor<4xTYPE>
        %r=linalg.reduce ins(%x:tensor<3x4xTYPE>) outs(%s:tensor<4xTYPE>) dimensions=[0]
          (%a:TYPE,%b:TYPE) { %v=arith.addi %a,%b:TYPE
            linalg.yield %v:TYPE }
        return %r:tensor<4xTYPE>
      }}""".replace("TYPE", f"i{bits}")
    )
    limit = 1 << (bits - 1)
    values = [[limit - 1, -limit, 127, -128]] * 3
    plans = limbs(source, [limit])
    assert plans and not any(op.name == "linalg.reduce" for op in source.walk())
    assert max(p["inputs"][0]["count"] for p in plans) == bits // 8
    assert evaluate(source, [values]) == [((3 * a + limit) % (2 * limit)) - limit for a in values[0]]
