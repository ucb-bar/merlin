"""Execute the pure byte/min/max candidate and the actual W8A8 denotation against native MLIR.

Other float arithmetic uses the existing host test oracle. This is numerical qualification,
not installation of an approved IEEE model or verification of a complete captured model.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import struct
import subprocess

import numpy as np
import pytest
from test_lean_w8a8_body import exported, lowered
from test_lean_w8a8_native_recipe import execute_recipe_case, samples_for_shape

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def evaluate(document, samples, work, *, byte_model=True):
    request = work / "byte-model-request.json"
    payload = {"export": document.read_text(), "function": "forward", "index_bits": 64, "samples": samples}
    if byte_model:
        payload["contract"] = "w8a8_byte_model"
    request.write_text(json.dumps(payload))
    proc = subprocess.run(
        ["lake", "env", "lean", "--run", "Differential.lean", str(request.absolute())],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return proc, None
    results = json.loads(proc.stdout)
    (work / "byte-model-results.json").write_text(
        json.dumps(
            {
                "scope": "candidate_byte_model_fidelity_only",
                "whole_pass_verified": False,
                "base_arithmetic": "existing host Float32 test oracle",
                "results": results,
            },
            indent=2,
        )
    )
    return proc, results


def compare(source, samples, work, lean_tools, fidelity_tools, *, arithmetic_nans=False):
    document = exported(parse_mlir_text(source), work, lean_tools)
    inputs, outputs = diff.signature(json.loads(document.read_text()), "forward")
    proc, lean = evaluate(document, samples, work)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Establish a defined observation before invoking native execution. Poison is never used
    # as a native expected answer, even when a host conversion happens to produce a value.
    assert all(s["status"] == "ok" and all(None not in v["bits"] for v in s["values"]) for s in lean)
    native = diff.execute_mlir(
        work / "actual.mlir", samples, work, fidelity_tools, inputs=inputs, outputs=outputs, function="forward"
    )
    diff.assert_equal(lean, native["results"], outputs, samples, arithmetic_nans=arithmetic_nans)
    return native["results"]


def test_raw_f32_conversion_against_native(tmp_path, lean_tools, fidelity_tools):
    source = """func.func @forward(%x: f32) -> i8 {
      %q = arith.fptosi %x : f32 to i8
      return %q : i8
    }"""
    # LLVM explicitly defines -128.9 -> -128. The truncated result, rather than the original
    # fraction, must fit. All values below are independently inside that defined domain.
    values = [-128.9, -128, -127.999, -1.9, -1, -0.9, -0.0, 0.0, 1e-45, -1e-45, 0.9, 1.9, 127.999]
    values += np.random.default_rng(987).uniform(-128.99, 127.99, 128).astype(np.float32).tolist()
    values = np.array(values, dtype=np.float32)
    expected = [[str(math.trunc(float(value)) % 256)] for value in values]
    assert all(-128 <= math.trunc(float(value)) <= 127 for value in values)
    samples = [[[str(word(value))]] for value in values]
    assert compare(source, samples, tmp_path, lean_tools, fidelity_tools) == expected


@pytest.mark.parametrize("operation", ["minimumf", "maximumf"])
def test_f32_min_max_bits_and_nan_classes(tmp_path, lean_tools, fidelity_tools, operation):
    source = f"""func.func @forward(%x: f32, %y: f32) -> f32 {{
      %r = arith.{operation} %x, %y : f32
      return %r : f32
    }}"""
    edges = [word(v) for v in [-float("inf"), -127, -1, -0.0, 0.0, 1, 127, float("inf")]]
    edges += [0x7FC00001, 0xFFC00017, 0x7F800001]
    pairs = list(itertools.product(edges, repeat=2))
    rng = random.Random(913)
    pairs += [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(128)]
    samples = [[[str(x)], [str(y)]] for x, y in pairs]
    compare(source, samples, tmp_path, lean_tools, fidelity_tools, arithmetic_nans=True)


QUANTIZER = """func.func @forward(%x: f32, %inverse: f32) -> i8 {
  %upper = arith.constant 127.0 : f32
  %lower = arith.constant -127.0 : f32
  %product = arith.mulf %x, %inverse : f32
  %rounded = math.roundeven %product : f32
  %hi = arith.minimumf %rounded, %upper : f32
  %lo = arith.maximumf %hi, %lower : f32
  %q = arith.fptosi %lo : f32 to i8
  return %q : i8
}"""


def test_complete_scalar_quantizer_against_native(tmp_path, lean_tools, fidelity_tools):
    samples = [
        [[str(word(x))], [str(word(inverse))]]
        for x, inverse in [
            (0, 1),
            (-0.0, 1),
            (0.5, 1),
            (1.5, 1),
            (2.5, 1),
            (-0.5, 1),
            (-1.5, 1),
            (-2.5, 1),
            (128, 1),
            (-128, 1),
            (float("inf"), 1),
            (-float("inf"), 1),
            (float(np.finfo(np.float32).max), 1000),
            (-float(np.finfo(np.float32).max), 1000),
        ]
    ]
    rng = np.random.default_rng(187)
    samples += [
        [[str(word(x))], [str(word(inv))]]
        for x, inv in zip(rng.uniform(-100, 100, 128), rng.uniform(0.001, 20, 128), strict=True)
    ]
    actual = compare(QUANTIZER, samples, tmp_path, lean_tools, fidelity_tools)
    assert actual[:8] == [[str(value % 256)] for value in [0, 0, 0, 2, 2, 0, -2, -2]]
    assert actual[8:14] == [["127"], ["129"], ["127"], ["129"], ["127"], ["129"]]


@pytest.mark.parametrize(
    "x,inverse", [(0x7FC00001, word(1)), (word(float("inf")), word(0)), (word(0), word(float("inf")))]
)
def test_nan_quantizer_is_poison_and_never_executed_natively(tmp_path, lean_tools, x, inverse):
    document = exported(parse_mlir_text(QUANTIZER), tmp_path, lean_tools)
    proc, actual = evaluate(document, [[[str(x)], [str(inverse)]]], tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert actual == [{"status": "ok", "values": [{"shape": None, "bits": [None]}]}]
    assert not (tmp_path / "native-results.json").exists()


@pytest.mark.parametrize("shape", [(2, 8, 3), (3, 9, 4), (1, 0, 2), (0, 3, 2)])
def test_whole_dynamic_w8a8_lean_mlir_and_numpy_agree(tmp_path, lean_tools, fidelity_tools, shape):
    module, _, _ = lowered(*shape)
    samples = samples_for_shape(*shape)
    native, independent = execute_recipe_case(module, samples, tmp_path, lean_tools, fidelity_tools)
    assert native == independent
    document = tmp_path / "actual.json"
    inputs, outputs = diff.signature(json.loads(document.read_text()), "forward")
    assert len(inputs) == 2
    arguments = [[[str(int(v)) for v in matrix.view(np.uint32).flat] for matrix in sample] for sample in samples]
    proc, lean = evaluate(document, arguments, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    diff.assert_equal(lean, native, outputs, arguments)


@pytest.mark.parametrize("byte_model,element,reason", [(False, "f32", "does not implement"), (True, "f64", "only f32")])
def test_model_selection_refuses_unimplemented_operations(tmp_path, lean_tools, byte_model, element, reason):
    module = parse_mlir_text(f"""func.func @forward(%x: {element}, %y: {element}) -> {element} {{
      %r = arith.maximumf %x, %y : {element}
      return %r : {element}
    }}""")
    document = exported(module, tmp_path, lean_tools)
    proc, answer = evaluate(document, [], tmp_path, byte_model=byte_model)
    assert proc.returncode == 2 and answer is None
    assert reason in proc.stderr
