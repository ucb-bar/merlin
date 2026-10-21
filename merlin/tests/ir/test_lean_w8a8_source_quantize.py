"""Qualify the uninstalled captured per-tensor quantization contract against pinned PyTorch.

Native MLIR is a separately constructed reference program, not Merlin's lowering. These tests
qualify candidate semantics and never issue a whole-pass verification verdict.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_lean_w8a8_body import exported

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def contract(request, work):
    path = work / "source-quantize-request.json"
    path.write_text(json.dumps({"contract": "w8a8_source_quantize", **request}))
    proc = subprocess.run(
        ["lake", "env", "lean", "--run", "Differential.lean", str(path.absolute())],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return proc, None
    answer = json.loads(proc.stdout)
    assert answer["scope"] == "source_quantize_candidate_only"
    assert answer["whole_pass_verified"] is False
    (work / "source-quantize-results.json").write_text(json.dumps(answer, indent=2))
    return proc, answer["results"]


def request_for(width, shape, samples):
    return {
        "zero_width": width,
        "shape": shape,
        "quant_min": "-128",
        "quant_max": "127",
        "output_dtype": "int8",
        "samples": samples,
    }


def source_program(width, shape):
    tensor = "tensor<" + "".join(f"{n}x" for n in shape) + "f32>"
    result = tensor.replace("f32", "i8")
    coordinates = ", ".join(f"d{i}" for i in range(len(shape)))
    identity = f"affine_map<({coordinates}) -> ({coordinates})>"
    iters = ", ".join('"parallel"' for _ in shape)
    return parse_mlir_text(f"""func.func @forward(%x: {tensor}, %s: tensor<f32>, %z: tensor<i{width}>) -> {result} {{
      %scale = tensor.extract %s[] : tensor<f32>
      %zero = tensor.extract %z[] : tensor<i{width}>
      %one = arith.constant 1.0 : f64
      %scale64 = arith.extf %scale : f32 to f64
      %inverse64 = arith.divf %one, %scale64 : f64
      %inverse = arith.truncf %inverse64 : f64 to f32
      %zero32 = arith.sitofp %zero : i{width} to f32
      %lower = arith.constant -128.0 : f32
      %upper = arith.constant 127.0 : f32
      %empty = tensor.empty() : {result}
      %r = linalg.generic {{indexing_maps = [{identity}, {identity}], iterator_types = [{iters}]}}
        ins(%x : {tensor}) outs(%empty : {result}) {{
        ^bb0(%value: f32, %unused: i8):
          %product = arith.mulf %value, %inverse : f32
          %rounded = math.roundeven %product : f32
          %shifted = arith.addf %rounded, %zero32 : f32
          %lo = arith.maximumf %shifted, %lower : f32
          %hi = arith.minimumf %lo, %upper : f32
          %q = arith.fptosi %hi : f32 to i8
          linalg.yield %q : i8
      }} -> {result}
      return %r : {result}
    }}""")


def samples_for(width, shape):
    rng = np.random.default_rng(13219)
    cases, arguments = [], []
    scales = [1.0, 0.1, 0.0004902036, 1.4985859, 1e-5, np.finfo(np.float32).tiny, np.finfo(np.float32).max, 1e-45]
    zeros = [0, -3, 17, 0, 0, (1 << (width - 1)) - 1, -(1 << (width - 1)), 0]
    for index, (scale, zero) in enumerate(zip(scales, zeros, strict=True)):
        values = rng.uniform(-500, 500, size=shape).astype(np.float32)
        if index == 0:
            edges = np.array([-128.9, -128, -127, -2.5, -1.5, -0.0, 0, 0.5, 1.5, 2.5, 126.5, 127.5], dtype=np.float32)
            values = np.resize(edges, values.size).reshape(shape)
        elif index == 4:
            values.fill(-0.0)
        elif index == 7:
            # The f64 reciprocal overflows when converted to f32. Nonzero inputs clip safely;
            # zero times that infinity is a separate poison observation, never native evidence.
            values.fill(-1.0)
        bits = [int(v) for v in values.reshape(-1).view(np.uint32)]
        cases.append({"zero_width": width, "shape": shape, "input_bits": bits, "scale_bits": word(scale), "zero": zero})
        arguments.append([[str(v) for v in bits], [str(word(scale))], [str(zero % (1 << width))]])
    return cases, arguments


def pytorch(cases, work):
    selected = os.environ.get("MERLIN_TORCH_PYTHON")
    if not selected:
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail("MERLIN_TORCH_PYTHON is required for source quantization qualification")
        pytest.skip("set MERLIN_TORCH_PYTHON to the pinned CPU source oracle")
    proc = subprocess.run(
        [str(Path(selected).absolute()), str(repo_root() / "merlin/tests/ir/lean_torch_quantize_oracle.py")],
        input=json.dumps(cases),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    answer = json.loads(proc.stdout)
    assert answer["torch"] == "2.10.0+cpu" and answer["numpy"] == "2.4.6"
    (work / "torch-results.json").write_text(json.dumps(answer, indent=2))
    return [[str(v) for v in sample] for sample in answer["results"]]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (1, 3, 5), (0, 3)])
def test_candidate_agrees_with_pytorch_and_native(tmp_path, lean_tools, fidelity_tools, width, shape):
    cases, arguments = samples_for(width, shape)
    expected = pytorch(cases, tmp_path)
    proc, actual = contract(request_for(width, shape, arguments), tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert all(sample["status"] == "ok" for sample in actual)
    assert [[value["bits"] for value in sample["values"]] for sample in actual] == [[sample] for sample in expected]
    document = exported(source_program(width, shape), tmp_path, lean_tools)
    inputs, outputs = diff.signature(json.loads(document.read_text()), "forward")
    native = diff.execute_mlir(
        tmp_path / "actual.mlir",
        arguments,
        tmp_path,
        fidelity_tools,
        inputs=inputs,
        outputs=outputs,
        function="forward",
    )
    assert native["results"] == expected


@pytest.mark.parametrize("scale", [0.0, -0.0, -1.0, float("inf"), -float("inf"), float("nan")])
def test_candidate_refuses_invalid_calibration_scale(tmp_path, scale):
    sample = [[[str(word(1))], [str(word(scale))], ["0"]]]
    proc, actual = contract(request_for(64, (1,), sample), tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert actual == [{"status": "stuck", "reason": "source quantize candidate requires a finite positive scale"}]
    assert not (tmp_path / "native-results.json").exists()


@pytest.mark.parametrize("x,scale", [(float("nan"), 1.0), (0.0, 1e-45)])
def test_undefined_conversion_never_runs_natively(tmp_path, x, scale):
    sample = [[[str(word(x))], [str(word(scale))], ["0"]]]
    proc, actual = contract(request_for(64, (1,), sample), tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert actual == [{"status": "ok", "values": [{"shape": [1], "bits": [None]}]}]
    assert not (tmp_path / "native-results.json").exists()


@pytest.mark.parametrize("change", [{"zero_width": 16}, {"quant_min": "-127"}, {"output_dtype": "uint8"}])
def test_candidate_refuses_unsupported_contract(tmp_path, change):
    proc, answer = contract({**request_for(64, (2, 8), []), **change}, tmp_path)
    assert proc.returncode == 2 and answer is None
