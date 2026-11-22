"""Closed BF16 source compositions: collapse and checked host C against exact references."""

import ctypes
import shutil
import subprocess
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from math import prod
from pathlib import Path
from random import Random

import pytest

from merlin.compiler.lower.c.pointwise_host import emit_host_pointwise_c
from merlin.compiler.semantic.linalg.linalg_bridge import LinalgBridgeError, translate_linalg_text
from merlin.compiler.semantic.linalg.source_compositions import row_reduction_maps
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

DATA = Path(__file__).parents[1] / "data" / "host_compositions"
FIXTURE = DATA / "norm_gelu_bf16.mlir"
_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798")


def _request(inputs, result, constants=()):
    return KernelRequest((*inputs, result), (result.id,), ("ram",),
                         tuple((node.id, "ram") for node in inputs if node.effect == "input"), "fixture",
                         constants=constants)


def _compile(tmp_path, source, name):
    record, files = emit_host_pointwise_c(source, name)
    directory = tmp_path / name
    directory.mkdir()
    for filename, text in files.items():
        (directory / filename).write_text(text)
    environment = directory / "fp_environment.c"
    environment.write_text("#include <fenv.h>\nint test_round_down(void) { return fesetround(FE_DOWNWARD); }\n"
                           "int test_round_nearest(void) { return fesetround(FE_TONEAREST); }\n")
    compiler = shutil.which("cc")
    assert compiler, "generated host C validation requires a local C compiler"
    library = directory / "region.so"
    subprocess.run(["nice", "-n", "10", compiler, "-std=gnu11", "-O2", "-fPIC", "-shared", "-fno-fast-math",
                    "-ffp-contract=off", "-Werror=frame-larger-than=4096",
                    *(str(directory / f) for f in files if f.endswith(".c")), str(environment),
                    "-o", str(library), "-lm"], check=True, capture_output=True)
    loaded = ctypes.CDLL(str(library))
    function = getattr(loaded, name)
    function.environment = loaded
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    function.restype = ctypes.c_int
    return record, function


def _call(function, arrays, output):
    inputs = (ctypes.c_void_p * len(arrays))(*(ctypes.addressof(a) for a in arrays))
    return function(None, inputs, (ctypes.c_void_p * 1)(ctypes.addressof(output)))


# Exact reference arithmetic, independent of the emitted C and of any float library.
def _value(bits, width=32):
    sign, exponent, fraction = (bits >> 31, (bits >> 23) & 255, bits & 0x7FFFFF) if width == 32 else \
        (bits >> 15, (bits >> 7) & 255, bits & 127)
    mantissa = 23 if width == 32 else 7
    assert exponent != 255
    magnitude = Fraction(fraction if not exponent else fraction | 1 << mantissa) * Fraction(2) ** (
        max(exponent, 1) - 127 - mantissa)
    return -magnitude if sign else magnitude


def _f32(value, zero_sign=0):
    """Binary32 RNE of an exact rational, with gradual underflow and overflow to infinity."""
    sign = 1 if value < 0 else 0
    value = abs(Fraction(value))
    if not value:
        return zero_sign << 31
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if Fraction(2) ** exponent > value:
        exponent -= 1
    quantum = Fraction(2) ** (max(exponent, -126) - 23)
    q, rest = divmod(value / quantum, 1)
    q = int(q) + (rest > Fraction(1, 2) or rest == Fraction(1, 2) and int(q) & 1)
    scaled = q * quantum
    if not q:  # underflow to zero keeps the sign of the exact value
        return sign << 31
    if scaled >= Fraction(2) ** 128:
        return sign << 31 | 0x7F800000
    exponent = scaled.numerator.bit_length() - scaled.denominator.bit_length()
    if Fraction(2) ** exponent > scaled:
        exponent -= 1
    if exponent < -126:
        return sign << 31 | int(scaled / Fraction(2) ** -149)
    return sign << 31 | (exponent + 127) << 23 | int(scaled / Fraction(2) ** (exponent - 23)) - (1 << 23)


def _bf16(bits):
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def _op(a, b, operation):
    """One IEEE binary32 operation on finite bit patterns, including signed-zero results."""
    sa, sb = a >> 31, b >> 31
    infinite = [(bits & 0x7FFFFFFF) == 0x7F800000 for bits in (a, b)]
    if any(infinite):  # Only the overflow cases the GELU bodies reach; NaN results are not admitted.
        if operation == "mul":
            assert all(bits & 0x7FFFFFFF for bits in (a, b))
            return (sa ^ sb) << 31 | 0x7F800000
        assert operation == "add" and not all(infinite)
        return a if infinite[0] else b
    x, y = _value(a), _value(b)
    if operation == "sub":
        y, sb = -y, sb ^ 1
    if operation in {"add", "sub"}:
        exact, zero = x + y, sa & sb
    else:
        exact, zero = (x * y, sa ^ sb) if operation == "mul" else (x / y, sa ^ sb)
    return _f32(exact, zero)


def _tanh(value):
    with localcontext() as context:
        context.prec = 110
        x = Decimal(value.numerator) / Decimal(value.denominator)
        if abs(x) > 20:
            return Fraction(1 if x > 0 else -1)
        e = (2 * abs(x)).exp()
        result = (e - 1) / (e + 1)
        return Fraction(-result if x < 0 else result)


def _erf(value):
    with localcontext() as context:
        context.prec = 110
        x = Decimal(value.numerator) / Decimal(value.denominator)
        if abs(x) > 6:
            return Fraction(1 if x > 0 else -1)
        term, total, n = x, x, 0
        while abs(term) > Decimal(10) ** -105 or n < 2:
            n += 1
            term = -term * x * x / n
            total += term / (2 * n + 1)
        return Fraction(total * 2 / _PI.sqrt())


def _fma(a, b, c):
    """IEEE binary32 fused multiply-add RN(a*b+c) on finite bit patterns, with signed-zero results."""
    product = _value(a) * _value(b)
    exact = product + _value(c)
    zero = ((a >> 31) ^ (b >> 31)) & (c >> 31) if not product and not _value(c) else 0
    return _f32(exact, zero)


def _sqrt(value):
    with localcontext() as context:
        context.prec = 60
        return Fraction((Decimal(value.numerator) / Decimal(value.denominator)).sqrt())


HALF, ONE, CUBIC, ROOT, INVERSE = 0x3F000000, 0x3F800000, 0x3D372713, 0x3F4C422A, 0x3F3504F3


def _math(function, bits):
    """Correctly rounded binary32 result; odd functions keep the sign of zero."""
    if (bits & 0x7FFFFFFF) == 0x7F800000:
        return bits >> 31 << 31 | ONE
    return _f32(function(_value(bits)), bits >> 31)


def _gelu(code, approximate):
    x = code << 16
    if approximate == "tanh":
        cube = _op(_op(x, x, "mul"), x, "mul")
        t = _math(_tanh, _op(ROOT, _op(x, _op(CUBIC, cube, "mul"), "add"), "mul"))
    else:
        t = _math(_erf, _op(x, INVERSE, "mul"))
    return _bf16(_op(_op(x, HALF, "mul"), _op(ONE, t, "add"), "mul"))


def _f32_int(value):
    return _f32(Fraction(value))


def _layer_norm(rows, weight, bias, epsilon_bits):
    """The pinned torch wheel's dispatched CPU BF16 layer_norm (7 fused, 2 rounded multiply-adds) in exact binary32 arithmetic."""
    lanes, chunk = 8, 16
    weights = [_f32(Fraction(1, j + 1)) for j in range(chunk)]

    def add_vec(add, m1a, m2a, m0, m1, m2):  # AddMomentsVec: rounded m1 update, fused m2 update
        n = m0 + add
        c = _op(_f32_int(add), _f32_int(n), "div") if n else 0
        out1, out2 = [], []
        for l in range(lanes):
            delta = _op(m1a[l], m1[l], "sub")
            cd = _op(c, delta, "mul")
            out1.append(_op(m1[l], cd, "add"))
            out2.append(_fma(_op(delta, _f32_int(m0), "mul"), cd, _op(m2[l], m2a[l], "add")))
        return n, out1, out2

    result = []
    for row in rows:
        x = [bits << 16 for bits in row]
        width = len(x)
        vectors = width // (2 * lanes)
        chunks = -(-vectors // chunk)
        depth = (chunks - 1).bit_length() if chunks > 1 else 0
        stack = [[0, [0] * lanes, [0] * lanes] for _ in range(max(depth, 1))]
        for i in range(chunks):
            count = min(chunk, vectors - i * chunk)
            for h in range(2):
                a1, a2 = [0] * lanes, [0] * lanes
                for j in range(count):
                    for l in range(lanes):
                        v = x[(i * chunk + j) * 2 * lanes + h * lanes + l]
                        delta = _op(v, a1[l], "sub")
                        a1[l] = _fma(delta, weights[j], a1[l])
                        a2[l] = _fma(delta, _op(v, a1[l], "sub"), a2[l])
                stack[0] = list(add_vec(count, a1, a2, *stack[0]))
            mask, j = i + 1, 1
            while j < depth and not mask & 1:
                stack[j] = list(add_vec(stack[j - 1][0], stack[j - 1][1], stack[j - 1][2], *stack[j]))
                stack[j - 1] = [0, [0] * lanes, [0] * lanes]
                mask >>= 1
                j += 1
        for j in range(1, depth):
            stack[0] = list(add_vec(stack[j][0], stack[j][1], stack[j][2], *stack[0]))
        m0, m1, m2 = 0, 0, 0
        for v in x[vectors * 2 * lanes:]:
            delta = _op(v, m1, "sub")
            m0 += 1
            m1 = _op(m1, _op(delta, _f32_int(m0), "div"), "add")
            m2 = _op(m2, _op(delta, _op(v, m1, "sub"), "mul"), "add")
        for l in range(lanes):  # AddMoments
            add = 2 * vectors
            n = m0 + add
            c = _op(_f32_int(add), _f32_int(n), "div") if n else 0
            delta = _op(stack[0][1][l], m1, "sub")
            m1 = _fma(c, delta, m1)
            m2 = _op(m2, _fma(_op(_op(delta, delta, "mul"), c, "mul"), _f32_int(m0), stack[0][2][l]), "add")
            m0 = n
        variance = _op(m2, _f32_int(width), "div")
        rstd = _op(ONE, _f32(_sqrt(_value(_op(variance, epsilon_bits, "add")))), "div")
        offset = _op(m1 ^ 0x80000000, rstd, "mul")
        for j, v in enumerate(x):
            out = _fma(_fma(v, rstd, offset), weight[j] << 16 if weight is not None else ONE,
                       bias[j] << 16 if bias is not None else 0)
            result.append(_bf16(out))
    return result


def _layer_norm_node(shape, k, roles, epsilon=0x358637BD):
    rank = len(shape)
    typ = TensorType(shape, "bf16", "raw-bit-pattern")
    operands = (SemanticNode("x", "input", (), typ, effect="input"),
                *(SemanticNode(role, "input", (), replace(typ, shape=shape[rank - k:]), effect="input") for role in roles))
    node = SemanticNode("norm", "bf16_layer_norm", tuple(n.id for n in operands), typ,
                        attrs=(("affine_operands", ",".join(roles)), ("epsilon_f32_bits", epsilon),
                               ("input_domain", "finite-bf16"), ("normalized_rank", k),
                               ("numeric_semantics", "torch-2.11-cpu-fma-bf16-layer-norm-v1")),
                        index_maps=row_reduction_maps(rank, k, len(roles)))
    return operands, node


def test_bf16_gelu_and_layer_norm_host_c_match_exact_references(tmp_path):
    rng = Random(20261120)
    # Every BF16 value of magnitude in [2^-7, 2^7), both signs, plus wide random exponents and edges.
    codes = [sign | exponent << 7 | fraction for sign in (0, 0x8000) for exponent in range(120, 134)
             for fraction in range(128)]
    codes += [rng.choice((0, 0x8000)) | rng.randrange(1, 255) << 7 | rng.randrange(128) for _ in range(512)]
    codes += [0x0001, 0x8001, 0x0080, 0x7F7F, 0xFF7F, 0x4B80, 0xCB80]
    words = ctypes.c_uint16 * len(codes)
    typ = TensorType((len(codes),), "bf16", "raw-bit-pattern")
    x = SemanticNode("x", "input", (), typ, effect="input")
    for approximate in ("tanh", "none"):
        gelu = SemanticNode("gelu", "bf16_gelu", ("x",), typ,
                            attrs=(("approximate", approximate), ("input_domain", "finite-bf16"),
                                   ("numeric_semantics", "bf16-gelu-f32-opmath-v1")),
                            index_maps=pointwise_identity_maps(1, 1))
        record, function = _compile(tmp_path, _request((x,), gelu), "gelu_" + approximate)
        assert record["host_math_symbols"] == ["merlin_host_" + ("tanhf" if approximate == "tanh" else "erff")]
        values, output = words(*codes), words(*([0x5555] * len(codes)))
        assert _call(function, [values], output) == 0
        expected = [_gelu(code, approximate) for code in codes]
        assert sum(1 for code in expected if code & 0x7FFF) > len(codes) // 2
        assert list(output) == expected
        for special in (0x7F80, 0xFF80, 0x7FC1):
            values[3] = special
            output[:] = [0x5555] * len(codes)
            assert _call(function, [values], output) == -2 and set(output) == {0x5555}
        values[3] = codes[3]
        assert _call(function, [values], values) == -1
        try:
            assert function.environment.test_round_down() == 0
            assert _call(function, [values], output) == -3 and set(output) == {0x5555}
        finally:
            assert function.environment.test_round_nearest() == 0
        with pytest.raises(ValueError, match="semantics differ"):
            emit_host_pointwise_c(_request((x,), replace(gelu, attrs=(("approximate", "sigmoid"),) + gelu.attrs[1:])), "changed")

    def bf16_word(scale):
        return rng.choice((0, 0x8000)) | (127 + rng.randrange(-scale, scale + 1)) << 7 | rng.randrange(128)

    # Mean 100 with BF16 spacing 0.5 exercises cancellation; 1,100 columns reach a cascade depth of three.
    def near_hundred():
        return 0x42C8 + rng.randrange(-4, 5)

    def spread():
        return bf16_word(6)

    cases = (((3, 2, 40), 1, ("weight", "bias"), spread), ((2, 4, 10), 2, ("weight",), spread),
             ((5, 24), 1, (), spread), ((2, 3, 600), 1, ("bias",), near_hundred), ((2, 1100), 1, ("weight", "bias"), near_hundred))
    for shape, k, roles, draw in cases:
        operands, node = _layer_norm_node(shape, k, roles)
        record, function = _compile(tmp_path, _request(operands, node), f"layer_norm_{len(roles)}_{k}_{shape[-1]}")
        width = prod(shape[len(shape) - k:])
        rows = [[draw() for _ in range(width)] for _ in range(prod(shape) // width)]
        terms = {role: [bf16_word(2) for _ in range(width)] for role in roles}
        flat = [code for row in rows for code in row]
        buffers = [(ctypes.c_uint16 * len(flat))(*flat), *((ctypes.c_uint16 * width)(*terms[r]) for r in roles)]
        output = (ctypes.c_uint16 * len(flat))(*([0x5555] * len(flat)))
        assert _call(function, buffers, output) == 0
        assert list(output) == _layer_norm(rows, terms.get("weight"), terms.get("bias"), 0x358637BD)
        assert record["inputs"] == ["x", *roles] and record["workspace_bytes"] == 0
        # Overflowing statistics refuse with no store; a constant row with zero epsilon has no inverse.
        buffers[0][0], buffers[0][1] = 0x7F7F, 0xFF7F
        output[:] = [0x5555] * len(flat)
        assert _call(function, buffers, output) == -2 and set(output) == {0x5555}
        buffers[0][0] = 0x7F80
        assert _call(function, buffers, output) == -2 and set(output) == {0x5555}
        assert _call(function, buffers, buffers[0]) == -1
    # Deployment links every region into one unit; embedded supplier sources must coexist.
    combined = tmp_path / "combined.c"
    combined.write_text("\n".join((tmp_path / d / (d + ".c")).read_text().replace(f'#include "{d}.h"', (tmp_path / d / (d + ".h")).read_text())
                                   for d in ("gelu_tanh", "gelu_none", "layer_norm_2_1_40")))
    subprocess.run([shutil.which("cc"), "-std=gnu11", "-O2", "-ffp-contract=off", "-c", str(combined),
                    "-o", str(tmp_path / "combined.o")], check=True, capture_output=True)
    # Sigmoid, softmax, and both GELU forms can share a unit, so every repeated supplier typedef must be renamed.
    from merlin.runtime import host_math_supplier
    from merlin.runtime.host_math_supplier.embedded import embedded_host_math
    supplier = tmp_path / "supplier.c"
    supplier.write_text("#include <stdint.h>\n#include <string.h>\n" + embedded_host_math(host_math_supplier, ("expf", "tanhf", "erff"))[0] + "\n")
    subprocess.run([shutil.which("cc"), "-std=gnu11", "-O2", "-ffp-contract=off", "-c", str(supplier),
                    "-o", str(tmp_path / "supplier.o")], check=True, capture_output=True)
    operands, node = _layer_norm_node((2, 8), 1, (), epsilon=0)
    _, function = _compile(tmp_path, _request(operands, node), "layer_norm_zero_epsilon")
    constant_rows = (ctypes.c_uint16 * 16)(*([0x3F80] * 16))
    output = (ctypes.c_uint16 * 16)(*([0x5555] * 16))
    assert _call(function, [constant_rows], output) == -2 and set(output) == {0x5555}
    for change, message in ((("normalized_rank", 2), "policy differs"),
                            (("numeric_semantics", "bf16-layer-norm-f32-opmath-v1"), "policy differs"),
                            (("epsilon_f32_bits", 0x7F800000), "epsilon")):
        operands, node = _layer_norm_node((3, 8), 1, ("weight", "bias"))
        attrs = tuple(change if key == change[0] else (key, value) for key, value in node.attrs)
        with pytest.raises(ValueError, match=message):
            emit_host_pointwise_c(_request(operands, replace(node, attrs=attrs)), "changed")
    # Affine terms bound as constants are embedded and give the same exact result.
    operands, node = _layer_norm_node((3, 40), 1, ("weight", "bias"))
    terms = {role: [bf16_word(2) for _ in range(40)] for role in ("weight", "bias")}
    from merlin.compiler.semantic.model import ConstantBinding

    constant = tuple(replace(n, effect="constant", op="constant") if n.id in terms else n for n in operands)
    bindings = tuple(ConstantBinding(role, "bf16-le-raw", b"".join(v.to_bytes(2, "little") for v in terms[role]).hex())
                     for role in ("weight", "bias"))
    record, function = _compile(tmp_path, _request(constant, node, bindings), "layer_norm_constant_terms")
    rows = [[bf16_word(6) for _ in range(40)] for _ in range(3)]
    flat = (ctypes.c_uint16 * 120)(*[v for row in rows for v in row])
    output = (ctypes.c_uint16 * 120)()
    assert record["inputs"] == ["x"] and _call(function, [flat], output) == 0
    assert list(output) == _layer_norm(rows, terms["weight"], terms["bias"], 0x358637BD)
    operands, node = _layer_norm_node((3, 8), 1, ("weight",))
    wide = replace(operands[1], type=replace(operands[1].type, dtype="f32"))
    with pytest.raises(ValueError, match="affine terms must be BF16"):
        emit_host_pointwise_c(_request((operands[0], wide), node), "changed")


def test_closed_bf16_decompositions_collapse_and_other_forms_stay_decomposed():
    source = FIXTURE.read_text()

    def ops(text):
        return [n.op for n in translate_linalg_text(text, entry="forward", target_identity="fixture").request.nodes
                if n.effect == "pure"]

    graph = translate_linalg_text(source, entry="forward", target_identity="fixture").request
    norm, tanh_gelu, erf_gelu = (graph.node(key) for key in ("op41r0_s0", "op47r0_s0", "op53r0_s0"))
    assert [n.op for n in graph.nodes if n.effect == "pure"] == ["bf16_layer_norm", "bf16_gelu", "bf16_gelu"]
    assert graph.outputs == (tanh_gelu.id, erf_gelu.id) and not graph.constants
    assert norm.inputs == ("arg2", "arg0", "arg1") and dict(norm.attrs) == {
        "affine_operands": "weight,bias", "epsilon_f32_bits": 0x358637BD, "input_domain": "finite-bf16",
        "normalized_rank": 1, "numeric_semantics": "torch-2.11-cpu-fma-bf16-layer-norm-v1"}
    assert tanh_gelu.inputs == (norm.id,) and dict(tanh_gelu.attrs)["approximate"] == "tanh"
    assert erf_gelu.inputs == ("arg2",) and dict(erf_gelu.attrs)["approximate"] == "none"
    # A changed literal or operation, or an escaping intermediate, keeps the original nodes or refuses.
    escape = ("func.return %121, %143 : tensor<2x3x8xbf16>, tensor<2x3x8xbf16>",
              "-> (tensor<2x3x8xbf16>, tensor<2x3x8xbf16>) {")
    mutations = (
        ("4.471500e-02", "4.471600e-02", None),  # tanh coefficient: no admitted f32 body
        ("0.707106769", "0.707106709", None),  # erf scale: no admitted f32 body
        ("%74 = math.rsqrt %72", "%74 = math.exp %72", (0, 2)),
        ("%24 = arith.constant 8.000000e+00", "%24 = arith.constant 7.000000e+00", (0, 2)),  # mean divisor
        ("1.000000e-06 : f32", "-1.000000e-06 : f32", (0, 2)),  # negative epsilon
        ("%46 = arith.constant 0.000000e+00", "%46 = arith.constant 1.000000e+00", (0, 2)),  # nonzero seed
        (escape[0], "func.return %35, %121, %143 : tensor<2x3x8xf32>, tensor<2x3x8xbf16>, tensor<2x3x8xbf16>", (0, 2)),
        (escape[0], "func.return %104, %121, %143 : tensor<2x3x8xf32>, tensor<2x3x8xbf16>, tensor<2x3x8xbf16>", (1, 1)),
    )
    for old, new, expected in mutations:
        assert old in source, old
        text = source.replace(old, new, 1)
        if new.startswith("func.return %"):
            text = text.replace(escape[1], "-> (tensor<2x3x8xf32>, tensor<2x3x8xbf16>, tensor<2x3x8xbf16>) {", 1)
        if expected is None:
            with pytest.raises(LinalgBridgeError):
                ops(text)
            continue
        translated = ops(text)
        assert (translated.count("bf16_layer_norm"), translated.count("bf16_gelu")) == expected, old


def test_bf16_negation_is_an_exact_sign_flip_for_every_pattern(tmp_path):
    source = """builtin.module {
      func.func @forward(%x: tensor<65536xbf16>) -> tensor<65536xbf16> {
        %empty = tensor.empty() : tensor<65536xbf16>
        %out = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                               iterator_types = ["parallel"]} ins(%x : tensor<65536xbf16>) outs(%empty : tensor<65536xbf16>) {
          ^bb0(%a: bf16, %unused: bf16):
            %n = arith.negf %a : bf16
            linalg.yield %n : bf16
        } -> tensor<65536xbf16>
        return %out : tensor<65536xbf16>
      }
    }"""
    graph = translate_linalg_text(source, entry="forward", target_identity="fixture").request
    node = graph.node(graph.outputs[0])
    assert node.op == "bf16_neg" and dict(node.attrs) == {"numeric_semantics": "bf16-sign-bit-flip-v1"}
    record, function = _compile(tmp_path, graph, "negation")
    assert record["operation_order"] == "Flip only the sign bit, including NaN payloads."
    words = ctypes.c_uint16 * 65536
    values, output = words(*range(65536)), words()
    assert _call(function, [values], output) == 0
    assert list(output) == [code ^ 0x8000 for code in range(65536)]
    for old, new in (("arith.negf %a : bf16", "arith.negf %a fastmath<nnan> : bf16"),
                     ("%n = arith.negf %a : bf16\n            linalg.yield %n", "%n = arith.negf %a : bf16\n            linalg.yield %a")):
        with pytest.raises(LinalgBridgeError):
            translate_linalg_text(source.replace(old, new), entry="forward", target_identity="fixture")


def test_unfolded_bf16_convolution_collapses_and_matches_exact_reference(tmp_path):
    source = (DATA / "patch_conv_bf16.mlir").read_text()
    graph = translate_linalg_text(source, entry="forward", target_identity="fixture").request
    biased, plain = (graph.node(key) for key in graph.outputs)
    assert [n.op for n in graph.nodes if n.effect == "pure"] == ["bf16_affine_contraction"] * 2
    assert dict(biased.attrs)["loop_extents"] == "8,3,4,4,1,2,2" and dict(biased.attrs)["reduction_loops"] == "1,2,3"
    assert biased.inputs == ("arg3", "arg0", "arg1") and plain.inputs == ("arg3", "arg2")
    rng = Random(7)

    def codes(count, scale):
        return [rng.choice((0, 0x8000)) | (127 + rng.randrange(-scale, scale + 1)) << 7 | rng.randrange(128)
                for _ in range(count)]

    x = codes(3 * 8 * 8, 4)
    for node, (cout, kernel) in ((biased, (8, 4)), (plain, (5, 2))):
        from merlin.compiler.partition.semantic_regions import extract_semantic_region

        region = extract_semantic_region(graph, (node.id,))
        record, function = _compile(tmp_path, region, "conv_" + node.id)
        assert record["numeric_semantics"] == "bf16-exact-product-f32-ltr-rne-bf16-rne-v1"
        w, b = codes(cout * 3 * kernel * kernel, 3), codes(cout, 2)
        side = 8 // kernel
        buffers = [(ctypes.c_uint16 * len(v))(*v) for v in (x, w, *((b,) if node is biased else ()))]
        output = (ctypes.c_uint16 * (cout * side * side))(*([0x5555] * (cout * side * side)))
        assert _call(function, buffers, output) == 0
        expected = []
        for co in range(cout):
            for oh in range(side):
                for ow in range(side):
                    total = 0
                    for ci in range(3):
                        for kh in range(kernel):
                            for kw in range(kernel):
                                pixel = x[(ci * 8 + oh * kernel + kh) * 8 + ow * kernel + kw] << 16
                                weight = w[((co * 3 + ci) * kernel + kh) * kernel + kw] << 16
                                total = _op(total, _op(pixel, weight, "mul"), "add")
                    if node is biased:
                        total = _op(total, b[co] << 16, "add")
                    expected.append(_bf16(total))
        assert list(output) == expected and sum(1 for v in expected if v & 0x7FFF) == len(expected)
        # Non-finite operands and a possible binary32 overflow refuse before stores.
        for index, code in ((0, 0x7F80), (1, 0x7F7F)):
            saved = buffers[index][0]
            buffers[index][0] = code
            assert _call(function, buffers, output) == -2
            buffers[index][0] = saved
        assert _call(function, buffers, buffers[0]) == -1
    for old, new in (("arith.constant 0.000000e+00 : f32", "arith.constant 1.000000e+00 : f32"),):
        assert old in source
        changed = translate_linalg_text(source.replace(old, new, 1), entry="forward", target_identity="fixture").request
        assert [n.op for n in changed.nodes if n.effect == "pure"].count("bf16_affine_contraction") == 1
    attrs = tuple(("reduction_loops", "1,2") if key == "reduction_loops" else (key, value) for key, value in biased.attrs)
    with pytest.raises(ValueError, match="indexing differs"):
        emit_host_pointwise_c(extract_semantic_region(replace(graph, nodes=tuple(
            replace(n, attrs=attrs) if n.id == biased.id else n for n in graph.nodes)), (biased.id,)), "changed")
