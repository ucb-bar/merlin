import ctypes
import hashlib
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.integer_readout import derive, emit_readout, evaluate, fixedpoint_candidate
from merlin.llvmlower.requantization import quantized


@pytest.mark.parametrize(
    "scales,relu",
    [
        ([0.5, 0.5], False),
        ([0.1525018811225891, 0.0011392629239708185, 4.235067367553711], True),
        ([1.7959643602371216, 0.00084189377957955, 0.45207950472831726], True),
    ],
)
def test_integer_readout_against_compiled_original_source(tmp_path, scales, relu):
    proof = derive(scales, relu=relu)
    candidate = fixedpoint_candidate(proof)
    assert candidate and candidate["max_estimate_output_error"] <= 1
    source = emit_readout(proof, "binary") + emit_readout(proof, "fixed", fixedpoint=True)
    optimized = []
    for fixedpoint in (False, True):
        for packet in (1, 2, 4, 8):
            name = f"optimized_{int(fixedpoint)}_{packet}"
            source += emit_readout(proof, name, fixedpoint=fixedpoint, saturation_first=True, packet=packet)
            optimized.append(name)
    body = "volatile float value=(float)x;"
    for s in scales:
        body += f"value=value*{float(s).hex()}f;"
    low = 0 if relu else -128
    source += f"""\n#include <math.h>
void original(const int32_t*a,int8_t*b,size_t n){{for(size_t i=0;i<n;i++){{int32_t x=a[i];{body}
float y=nearbyintf(value);if(y<{low})y={low};if(y>127)y=127;b[i]=(int8_t)y;}}}}
"""
    (tmp_path / "readout.c").write_text(source)
    library = tmp_path / ("readout_" + hashlib.sha256(source.encode()).hexdigest() + ".so")
    subprocess.run(
        ["cc", "-O2", "-ffp-contract=off", "-fPIC", "-shared", str(tmp_path / "readout.c"), "-lm", "-o", str(library)],
        check=True,
    )
    boundaries = {-(1 << 31), (1 << 31) - 1, 0}
    for t in proof["thresholds"]:
        boundaries.update(x for x in (t - 1, t, t + 1) if -(1 << 31) <= x < (1 << 31))
    # Exhaust all unsaturated transition intervals, plus full i32 extremes.
    interior = np.arange(
        max(-(1 << 31), proof["thresholds"][0] - 2), min((1 << 31), proof["thresholds"][-1] + 2), dtype=np.int32
    )
    a = np.unique(np.concatenate([interior, np.array(sorted(boundaries), dtype=np.int32)]))
    lib = ctypes.CDLL(str(library))
    outputs = []
    for name in ("binary", "fixed", "original", *optimized):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        output = np.full(a.size + 8, -77, dtype=np.int8)
        fn(a.ctypes.data, output.ctypes.data, a.size)
        assert np.all(output[a.size :] == -77)
        outputs.append(output[: a.size])
    assert np.array_equal(outputs[0], outputs[2])
    assert np.array_equal(outputs[1], outputs[2])
    for result in outputs[3:]:
        assert np.array_equal(result, outputs[2])


def test_packet_tail_and_overlapping_storage_preserve_source_order(tmp_path):
    proof = derive([0.25])
    source = emit_readout(proof, "control", fixedpoint=True)
    source += emit_readout(proof, "packet", fixedpoint=True, saturation_first=True, packet=8)
    path = tmp_path / "readout.c"
    library = tmp_path / "readout.so"
    path.write_text(source)
    subprocess.run(["cc", "-O2", "-fPIC", "-shared", str(path), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    for name in ("control", "packet"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    original = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64], dtype=np.int32)
    for count in (0, 1, 7, 8, 9, 15, 16, 17):
        outputs = []
        for name in ("control", "packet"):
            result = np.full(count + 8, -77, dtype=np.int8)
            getattr(lib, name)(original.ctypes.data, result.ctypes.data, count)
            assert np.all(result[count:] == -77)
            outputs.append(result)
        assert np.array_equal(*outputs)
        # A char store into the next input's low byte affects the next read.
        # A packet that preloads lanes would silently change this behavior.
        images = []
        for name in ("control", "packet"):
            image = original.copy()
            getattr(lib, name)(image.ctypes.data, image.ctypes.data + 4, count)
            images.append(image.view(np.uint8))
        assert np.array_equal(*images)


@pytest.mark.parametrize(
    "scales,lo,hi,relu",
    [([0.01], -1, 1, False), ([1e-9], 0, 10, True), ([10.0], 20, 30, False), ([10.0], -30, -20, False)],
)
def test_saturation_with_repeated_narrow_domain_thresholds(tmp_path_factory, scales, lo, hi, relu):
    # Keep loaded libraries' files until process exit. Successful per-test temp
    # cleanup can otherwise recycle an inode that dlopen still has cached.
    tmp_path = tmp_path_factory.mktemp("narrow-readout")
    proof = derive(scales, lo, hi, relu)
    path = tmp_path / "readout.c"
    library = tmp_path / "readout.so"
    path.write_text(emit_readout(proof, "readout", saturation_first=True, packet=4))
    subprocess.run(["cc", "-O2", "-fPIC", "-shared", str(path), "-o", str(library)], check=True)
    fn = ctypes.CDLL(str(library)).readout
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    values = np.arange(lo, hi + 1, dtype=np.int32)
    result = np.full(values.size + 8, -77, dtype=np.int8)
    fn(values.ctypes.data, result.ctypes.data, values.size)
    assert result[: values.size].tolist() == [quantized(int(x), scales, relu) for x in values]
    assert np.all(result[values.size :] == -77)


@pytest.mark.parametrize("options", [{"packet": True}, {"packet": 3}, {"packet": 0}, {"saturation_first": 1}])
def test_refuses_invalid_readout_options(options):
    with pytest.raises(ValueError):
        emit_readout(derive([0.25]), "bad", **options)


def test_narrow_domain_repeated_sentinel_and_proof_tampering():
    p = derive([0.01], -1, 1, False)
    assert [evaluate(x, p) for x in (-1, 0, 1)] == [0, 0, 0]
    with pytest.raises(ValueError):
        evaluate(2, p)
    p["thresholds"][0] += 1
    with pytest.raises(ValueError, match="changed"):
        emit_readout(p, "bad")


@pytest.mark.parametrize("scales", [[], [-1], [0], [float("inf")], [float("nan")]])
def test_refuse_nonmonotone_or_invalid_scales(scales):
    with pytest.raises(ValueError):
        derive(scales)


@pytest.mark.parametrize("scale", [1.0, 0.0625, 0.125, 0.4307985291804076, 0.75])
@pytest.mark.parametrize("relu", [False, True])
def test_rational_scalar_ir_consumed_with_complete_transition_boundary_inputs(scale, relu):
    from merlin.llvmlower.integer_readout import emit_rational_scalar_ir
    from merlin.compile.scheduling.dense_contracts import DenseIntegerReadout
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.runtime.linalg_numpy import evaluate as evaluate_ir

    ir, receipt = emit_rational_scalar_ir([scale], relu=relu)
    graph = DenseIntegerReadout(ir, "i8", "neutral_readout", ()).graph()
    assert graph.input_bits == (32,) and graph.output_bits == 8
    points = {-(1 << 31), (1 << 31) - 1, -24, -8, 0, 8, 24}
    for threshold in receipt["source"]["thresholds"]:
        points.update(x for x in (threshold - 1, threshold, threshold + 1)
                      if -(1 << 31) <= x < (1 << 31))
    values = np.array(sorted(points), dtype=np.int32)
    actual, = evaluate_ir(parse_mlir_text(ir), [values])
    # The source oracle computes binary32 conversion/multiply/nearest-even,
    # independently of the emitted integer operations and proof partition.
    expected = [quantized(int(x), [scale], relu) for x in values]
    assert actual.tolist() == expected


@pytest.mark.parametrize("scales", [[0.01], [1e-9], [2.0 ** -25], [2.0 ** 40]])
def test_rational_refuses_double_rounding_or_integer_range(scales):
    from merlin.llvmlower.integer_readout import emit_rational_scalar_ir
    with pytest.raises(ValueError, match="no exact signed-i64"):
        emit_rational_scalar_ir(scales)


@pytest.mark.parametrize("scales", [[0], [-0.5], [float("inf")], [float("nan")]])
def test_rational_refuses_nonpositive_scale(scales):
    from merlin.llvmlower.integer_readout import emit_rational_scalar_ir
    with pytest.raises(ValueError, match="positive finite"):
        emit_rational_scalar_ir(scales)


@pytest.mark.parametrize("shift", [-1, 64])
def test_readout_refuses_poison_constant_shifts(shift):
    from merlin.compile.scheduling.dense_contracts import DenseIntegerReadout
    ir = f"""module {{func.func @readout(%x:i64)->i8 {{
      %shift=arith.constant {shift}:i64
      %v=arith.shrsi %x,%shift:i64
      %r=arith.trunci %v:i64 to i8
      func.return %r:i8
    }}}}"""
    with pytest.raises(ValueError, match="non-total"):
        DenseIntegerReadout(ir, "i8", "bad", ()).graph()


def test_readout_refuses_dynamic_shift():
    from merlin.compile.scheduling.fusion_semantics import scalar_graph
    ir = b"""module {func.func @readout(%x:i64,%shift:i64)->i8 {
      %v=arith.shrsi %x,%shift:i64
      %r=arith.trunci %v:i64 to i8
      func.return %r:i8
    }}"""
    with pytest.raises(ValueError, match="non-total"):
        scalar_graph(ir, 128, 65536)
