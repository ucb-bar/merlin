"""Execute checked BF16 host regions through their generated opaque-pointer ABI."""

import ctypes
import shutil
import struct
import subprocess
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from merlin.compiler.lower.c.pointwise_host import emit_host_pointwise_c
from merlin.compiler.semantic.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps


def _request(inputs, result):
    return KernelRequest(
        (*inputs, result), (result.id,), ("ram",), tuple((node.id, "ram") for node in inputs), "fixture"
    )


def _compile(tmp_path, source, name):
    record, files = emit_host_pointwise_c(source, name)
    directory = tmp_path / name
    directory.mkdir()
    for filename, text in files.items():
        (directory / filename).write_text(text)
    environment = directory / "fp_environment.c"
    environment.write_text(
        "#include <fenv.h>\n"
        "int test_round_down(void) { return fesetround(FE_DOWNWARD); }\n"
        "int test_round_nearest(void) { return fesetround(FE_TONEAREST); }\n"
    )
    compiler = shutil.which("cc")
    assert compiler, "generated host C validation requires a local C compiler"
    library = directory / "region.so"
    subprocess.run(
        [
            "nice",
            "-n",
            "10",
            compiler,
            "-std=c11",
            "-O2",
            "-fPIC",
            "-shared",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-Werror=frame-larger-than=4096",
            str(directory / (name + ".c")),
            str(environment),
            "-o",
            str(library),
            "-lm",
        ],
        check=True,
        capture_output=True,
    )
    loaded = ctypes.CDLL(str(library))
    function = getattr(loaded, name)
    function.environment = loaded
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    function.restype = ctypes.c_int
    return record, function


def _call(function, arrays, output):
    inputs = (ctypes.c_void_p * len(arrays))(*(ctypes.addressof(a) for a in arrays))
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output))
    return function(None, inputs, outputs)


def test_checked_bf16_host_regions_execute_and_refuse_changed_contracts(tmp_path):
    # Two channels give exact alpha=(2,-1), beta=(1,2). Independent rational
    # arithmetic gives y=(2*x+1, -x+2), all representable in BF16 here.
    activation = TensorType((1, 2, 1, 3), "bf16", "raw-bit-pattern")
    channel_type = TensorType((2,), "bf16", "raw-bit-pattern")
    operands = (
        SemanticNode("x", "input", (), activation, effect="input"),
        *(SemanticNode(role, "input", (), channel_type, effect="input") for role in ("mean", "var", "weight", "bias")),
    )
    identity = pointwise_identity_maps(4, 1)[0]
    channel = IndexMap(4, ((0, 1, 0, 0),), (0,))
    norm = SemanticNode(
        "norm",
        "bf16_batch_norm_inference",
        tuple(n.id for n in operands),
        activation,
        attrs=(
            ("channel_operands", "mean,var,weight,bias"),
            ("epsilon_f32_bits", 0),
            ("input_domain", "finite-bf16"),
            ("numeric_semantics", "torch-cpu-bf16-batch-norm-inference-v1"),
        ),
        index_maps=(identity, channel, channel, channel, channel, identity),
    )
    source = _request(operands, norm)
    record, function = _compile(tmp_path, source, "batch_norm")
    words = ctypes.c_uint16 * 6
    x = words(0xBF80, 0, 0x3FC0, 0xC000, 0x3F80, 0x4000)
    terms = [(ctypes.c_uint16 * 2)(*v) for v in ((0, 0x3F80), (0x3F80, 0x4080), (0x4000, 0xC000), (0x3F80, 0x3F80))]
    output = words(*([0x5555] * 6))
    assert _call(function, [x, *terms], output) == 0
    assert list(output) == [0xBF80, 0x3F80, 0x4080, 0x4080, 0x3F80, 0]
    assert record["inputs"] == ["x", "mean", "var", "weight", "bias"]
    assert record["workspace_bytes"] == 0
    # Compiler-enforced stack bound with thousands of channels rejects the
    # former per-channel alpha/beta arrays. No large tensor allocation is needed
    # to exercise the generated function's resource requirement.
    wide_activation = replace(activation, shape=(1, 8192, 1, 1))
    wide_operands = (
        replace(operands[0], type=wide_activation),
        *(replace(term, type=replace(channel_type, shape=(8192,))) for term in operands[1:]),
    )
    wide_record, _ = _compile(tmp_path, _request(wide_operands, replace(norm, type=wide_activation)), "wide_batch_norm")
    assert wide_record["workspace_bytes"] == 0
    x[5] = 0x7F80
    output[:] = [0x5555] * 6
    assert _call(function, [x, *terms], output) == -2 and list(output) == [0x5555] * 6
    x[5] = 0x4000
    assert _call(function, [x, *terms], x) == -1
    # Statistics cannot alias output even though channel terms precede stores.
    assert _call(function, [x, *terms], terms[0]) == -1
    changed = replace(norm, attrs=tuple((k, "changed" if k == "numeric_semantics" else v) for k, v in norm.attrs))
    with pytest.raises(ValueError, match="policy differs"):
        emit_host_pointwise_c(_request(operands, changed), "changed")

    # Dilated NCHW windows with left padding exercise affine geometry and the
    # first-equal signed-zero rule independently of any tensor framework.
    typ = TensorType((1, 1, 1, 4), "bf16", "raw-bit-pattern")
    operand = SemanticNode("x", "input", (), typ, effect="input")
    window = SemanticNode(
        "pool",
        "bf16_window_max",
        ("x",),
        typ,
        attrs=(
            ("input_domain", "non-nan-bf16"),
            ("numeric_semantics", "ordered-window-max-first-equal-v1"),
            ("window_h", 1),
            ("window_w", 2),
        ),
        index_maps=(
            IndexMap(
                6, ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 1, 0), (0, 0, 0, 1, 0, 2)), (0, 0, 0, -1)
            ),
            IndexMap(6, ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0)), (0, 0, 0, 0)),
        ),
    )
    _, pool = _compile(tmp_path, _request((operand,), window), "window_max")
    row = (ctypes.c_uint16 * 4)(0x8000, 0x3F80, 0, 0xBF80)
    pooled = (ctypes.c_uint16 * 4)()
    assert _call(pool, [row], pooled) == 0
    assert list(pooled) == [0x3F80, 0x8000, 0x3F80, 0]
    row[2] = 0x7F80
    assert _call(pool, [row], pooled) == 0 and list(pooled) == [0x3F80, 0x7F80, 0x3F80, 0x7F80]
    row[3] = 0x7FC1
    pooled[:] = [0x5555] * 4
    assert _call(pool, [row], pooled) == -2 and list(pooled) == [0x5555] * 4
    with pytest.raises(ValueError, match="indexing, or policy differs"):
        emit_host_pointwise_c(_request((operand,), replace(window, index_maps=window.index_maps[::-1])), "changed")

    relu = SemanticNode(
        "relu",
        "relu",
        ("x",),
        typ,
        attrs=(("input_domain", "finite-normal-or-zero-bf16"), ("numeric_semantics", "bf16-sign-clamp-v1")),
        index_maps=pointwise_identity_maps(4, 1),
    )
    # The rejected sign-clamp declaration has no host route; only the ordered select emits.
    with pytest.raises(ValueError, match="indexing, or policy differs"):
        emit_host_pointwise_c(_request((operand,), relu), "relu")

    sigmoid_type = replace(typ, shape=(6,))
    sigmoid_input = replace(operand, type=sigmoid_type)
    sigmoid = SemanticNode(
        "sigmoid",
        "bf16_sigmoid",
        ("x",),
        sigmoid_type,
        attrs=(("input_domain", "non-nan-bf16"), ("numeric_semantics", "torch-cpu-bf16-sigmoid-v1")),
        index_maps=pointwise_identity_maps(1, 1),
    )
    sigmoid_source = _request((sigmoid_input,), sigmoid)
    record, logistic = _compile(tmp_path, sigmoid_source, "sigmoid")
    values = words(0, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80)
    output[:] = [0x5555] * 6
    assert _call(logistic, [values], output) == 0

    # Independent decimal exponential, then explicit binary32 scalar steps.
    def reference(value):
        def f32(x):
            return struct.unpack("<f", struct.pack("<f", x))[0]

        with localcontext() as context:
            context.prec = 100
            exponential = f32(float((-Decimal(value)).exp()))
        result = f32(1.0 / f32(1.0 + exponential))
        bits = struct.unpack("<I", struct.pack("<f", result))[0]
        return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16

    assert list(output) == [0x3F00, 0x3F00, reference(1), reference(-1), 0x3F80, 0]
    assert record["qualification"] == "requires-exact-receipt"
    values[5] = 0x7FC1
    output[:] = [0x5555] * 6
    assert _call(logistic, [values], output) == -2 and list(output) == [0x5555] * 6
    assert _call(logistic, [values], values) == -1
    values[5] = 0xFF80
    try:
        assert logistic.environment.test_round_down() == 0
        assert _call(logistic, [values], output) == -3 and list(output) == [0x5555] * 6
    finally:
        assert logistic.environment.test_round_nearest() == 0
    with pytest.raises(ValueError, match="semantics differ"):
        emit_host_pointwise_c(_request((sigmoid_input,), replace(sigmoid, attrs=())), "changed")

    _check_ordered_select(tmp_path)
    _check_post_readout_bias(tmp_path)


def _check_post_readout_bias(tmp_path):
    from merlin.compiler.partition.semantic_regions import extract_semantic_region
    from merlin.compiler.semantic.source_preparation import prepare_source

    typ = TensorType((1, 2), "bf16", "raw-bit-pattern")
    operands = tuple(
        SemanticNode(key, "input", (), replace(typ, shape=(2,)) if key == "bias" else typ, effect="input")
        for key in ("x", "xe", "w", "we", "bias")
    )
    contraction = SemanticNode(
        "linear",
        "fixture_contraction",
        tuple(n.id for n in operands),
        typ,
        attrs=(("bias_order", "after_output_cast"), ("semantic_kind", "prequantized_contraction")),
    )
    prepared, _ = prepare_source(_request(operands, contraction))
    region = extract_semantic_region(prepared, ("linear",))
    record, add = _compile(tmp_path, region, "bias_add")
    assert record["numeric_semantics"] == "ieee754-bf16-add-rne-v1"
    assert record["inputs"] == ["linear.readout", "linear.bias"]
    readout = (ctypes.c_uint16 * 2)(0x3F80, 0x3F81)
    bias = (ctypes.c_uint16 * 2)(0x3B80, 0x3B80)  # Half an ULP at one: ties to even.
    output = (ctypes.c_uint16 * 2)()
    assert _call(add, [readout, bias], output) == 0 and list(output) == [0x3F80, 0x3F82]


def _check_ordered_select(tmp_path):
    from merlin.compiler.frontend.patterns.narrow_float import SELECT_ATTRS
    from merlin.compiler.semantic.reference import TensorValue, evaluate_graph
    typ = TensorType((65536,), "bf16", "raw-bit-pattern")
    x = SemanticNode("x", "input", (), typ, effect="input")
    relu = SemanticNode("relu", "relu", ("x",), typ,
                        attrs=tuple(sorted(SELECT_ATTRS.items())), index_maps=pointwise_identity_maps(1, 1))
    source = _request((x,), relu)
    # IEEE comparison on exactly widened BF16 values is independent of the byte emitter.
    expected = [0 if struct.unpack('<f', struct.pack('<I', b << 16))[0] < 0 else b
                for b in range(65536)]
    actual, = evaluate_graph(source, {'x': TensorValue(typ, tuple(range(65536)))})
    assert list(actual.elements) == expected
    _, function = _compile(tmp_path, source, "ordered_select")
    words = ctypes.c_uint16 * 65536
    inputs, output = words(*range(65536)), words()
    assert _call(function, [inputs], output) == 0
    assert list(output) == expected
    assert _call(function, [inputs], inputs) == -1
