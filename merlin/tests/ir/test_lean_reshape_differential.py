"""Native MLIR fidelity of reassociation composition; this module requires the host runner."""

import json
import math

import pytest
from test_lean_reshape_compose import _canonicalize, _rank_zero_source, _source

from merlin.verify.lean import differential as diff
from merlin.verify.lean.export import export_file


def _bits(width):
    if width in (32, 64):
        fraction, exponent = (23, 8) if width == 32 else (52, 11)
        sign = 1 << (width - 1)
        infinity = ((1 << exponent) - 1) << fraction
        return [
            0,
            sign,
            1,
            sign | 1,
            1 << fraction,
            infinity - 1,
            infinity,
            sign | infinity,
            infinity | (1 << (fraction - 1)) | 0x123,
            sign | infinity | (1 << (fraction - 1)) | 0x321,
        ]
    return [0, 1, (1 << (width - 1)) - 1, 1 << (width - 1), (1 << width) - 1]


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize(
    "shape,element", [((2, 3, 4), "i13"), ((0, 3, 4), "i8"), ((2, 3, 4), "f32"), ((2, 3, 4), "f64")]
)
def test_reshape_composition_matches_mlir_bits(tmp_path, fidelity_tools, expand, shape, element):
    source = _source(shape, element, expand=expand)
    _, _, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, source)
    target = after.read_text()
    width, count = int(element[1:]), math.prod(shape)
    pool = _bits(width)
    samples = []
    for offset in range(len(pool)):
        values = [pool[(offset + i) % len(pool)] for i in range(count)]
        alternate = [pool[(offset + i + 1) % len(pool)] for i in range(count)]
        samples.append([values, alternate])
    # Reassociation preserves row-major storage; the original input also escapes unchanged.
    expected = [[str(x) for x in sample[0] + sample[0]] for sample in samples]
    for name, program in [("source", source), ("target", target)]:
        work = tmp_path / name
        diff.compare(program, samples, work, fidelity_tools, function="forward")
        results = json.loads((work / "results.json").read_text())
        assert results["native"]["results"] == expected
        assert [[bit for value in sample["values"] for bit in value["bits"]] for sample in results["lean"]] == expected


def test_native_reshape_wrong_input_control(tmp_path, fidelity_tools):
    source = _source((2, 3, 4), "i32")
    _, _, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, source)
    target = after.read_text()
    # The parser verifies this independent mutation before either execution path runs.
    line = next(line for line in target.splitlines() if "tensor.collapse_shape" in line)
    bad = target.replace(line, line.replace("%arg0", "%arg1"))
    assert bad != target
    samples = [[[i for i in range(24)], [100 + i for i in range(24)]]]
    for name, program in [("good", target), ("bad", bad)]:
        diff.compare(program, samples, tmp_path / name, fidelity_tools, function="forward")
    good = json.loads((tmp_path / "good/results.json").read_text())
    bad_result = json.loads((tmp_path / "bad/results.json").read_text())
    assert good["native"]["results"] != bad_result["native"]["results"]
    assert good["native"]["results"][0] == [str(x) for x in samples[0][0] * 2]
    assert bad_result["native"]["results"][0] == [str(x) for x in samples[0][1] + samples[0][0]]


@pytest.mark.parametrize("expand", [False, True])
def test_rank_zero_transition_records_model_coverage_gap(tmp_path, fidelity_tools, expand):
    _, before, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, _rank_zero_source(expand))
    samples = [[["0"]], [["17"]], [["4294967295"]]]
    for name, source in [("source", before), ("target", after)]:
        work = tmp_path / name
        work.mkdir()
        exported = work / "export.json"
        export_file(source, exported, backend="mlir", m2m_python=fidelity_tools.compiler_python)
        inputs, outputs = diff.signature(json.loads(exported.read_text()), "forward")
        lean = diff.evaluate_lean(exported, samples, work, fidelity_tools, function="forward")
        assert all(sample["status"] == "stuck" for sample in lean)
        native = diff.execute_mlir(
            source, samples, work, fidelity_tools, inputs=inputs, outputs=outputs, function="forward"
        )
        assert native["results"] == [["0"], ["17"], ["4294967295"]]
