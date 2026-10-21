"""Qualify the additive i32/i64 source contract, without changing semantic dispatch.

The candidate is evaluated by Lean, pinned PyTorch, and a separately constructed standard-MLIR
arithmetic program. Production i64 per-channel lowering is kept as a negative control.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_lean_w8a8_body import exported

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import lower_quant_ext
from merlin.verify.lean import differential as diff


def contract(request, root):
    path = root / "contract-request.json"
    path.write_text(json.dumps({"contract": "w8a8_dequant_width", **request}))
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
    assert answer["scope"] == "dequant_width_candidate_only"
    assert answer["whole_pass_verified"] is False
    (root / "contract-results.json").write_text(json.dumps(answer, indent=2))
    return proc, answer["results"]


def source_program(width, kind, shape, axis, *, reference):
    dims = "".join(f"{n}x" for n in shape)
    parameter_dims = f"{shape[axis]}x" if kind == "per_channel" else ""
    q, s, z, out = (
        f"tensor<{dims}i8>",
        f"tensor<{parameter_dims}f32>",
        f"tensor<{parameter_dims}i{width}>",
        f"tensor<{dims}f32>",
    )
    if reference:
        coordinates = ", ".join(f"d{i}" for i in range(len(shape)))
        identity = f"affine_map<({coordinates}) -> ({coordinates})>"
        parameter = f"affine_map<({coordinates}) -> ({f'd{axis}' if kind == 'per_channel' else ''})>"
        iters = ", ".join('"parallel"' for _ in shape)
        arithmetic = (
            f"%promoted = arith.extsi %qv : i8 to i{width}\n"
            f"%diff = arith.subi %promoted, %zv : i{width}\n"
            f"%unscaled = arith.sitofp %diff : i{width} to f32"
            if kind == "per_channel"
            else f"%qf = arith.sitofp %qv : i8 to f32\n%zf = arith.sitofp %zv : i{width} to f32\n"
            "%unscaled = arith.subf %qf, %zf : f32"
        )
        body = f"""%empty = tensor.empty() : {out}
          %result = linalg.generic {{indexing_maps = [{identity}, {parameter}, {parameter}, {identity}],
            iterator_types = [{iters}]}} ins(%q, %s, %z : {q}, {s}, {z}) outs(%empty : {out}) {{
            ^bb0(%qv : i8, %sv : f32, %zv : i{width}, %unused : f32):
              {arithmetic}
              %scaled = arith.mulf %unscaled, %sv : f32
              linalg.yield %scaled : f32
          }} -> {out}"""
    else:
        attrs = f"axis = {axis} : i64" if kind == "per_channel" else ""
        body = f'%result = "quant_ext.dequantize_{kind}"(%q, %s, %z) <{{{attrs}}}> : ({q}, {s}, {z}) -> {out}'
    return parse_mlir_text(f"""func.func @forward(%q: {q}, %s: {s}, %z: {z}) -> {out} {{
      {body}
      return %result : {out}
    }}""")


def samples(width, kind, shape, axis):
    parameter_shape = (shape[axis],) if kind == "per_channel" else ()
    rng = np.random.default_rng(20261020)
    cases, arguments = [], []
    # Adjacent-to-f32-integer boundaries distinguish subtraction before/after conversion.
    zeros = [0, 16777217, -(1 << (width - 1)), (1 << (width - 1)) - 1, -3]
    for index, zero in enumerate(zeros):
        q = rng.integers(-128, 128, size=shape, dtype=np.int8)
        z = np.full(parameter_shape, zero, dtype=np.int32 if width == 32 else np.int64)
        s = rng.uniform(-2, 2, size=parameter_shape).astype(np.float32)
        if index == 1:
            q.fill(1)
            s.fill(1.0)
        elif index == 2:
            q.fill(127)
            s.fill(1.0)
        elif index == 3:
            q.fill(-128)
            s.fill(1.0)
        elif index == 4:
            s.fill(-0.0)
        cases.append(
            {
                "kind": kind,
                "zero_width": width,
                "shape": shape,
                "axis": axis,
                "q": q.reshape(-1).tolist(),
                "z": z.reshape(-1).tolist(),
                "scale_bits": s.reshape(-1).view(np.uint32).tolist(),
            }
        )
        arguments.append(
            [
                [str(int(x)) for x in q.reshape(-1).view(np.uint8)],
                [str(int(x)) for x in s.reshape(-1).view(np.uint32)],
                [str(int(x)) for x in z.reshape(-1).view(np.uint32 if width == 32 else np.uint64)],
            ]
        )
    return cases, arguments


def pytorch(cases, root):
    selected = os.environ.get("MERLIN_TORCH_PYTHON")
    if not selected:
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail("MERLIN_TORCH_PYTHON is required for source-contract qualification")
        pytest.skip("set MERLIN_TORCH_PYTHON to the pinned CPU source oracle")
    oracle = repo_root() / "merlin/tests/ir/lean_torch_dequant_oracle.py"
    proc = subprocess.run(
        [str(Path(selected).absolute()), str(oracle)],
        input=json.dumps(cases),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    answer = json.loads(proc.stdout)
    assert answer["torch"] == "2.10.0+cpu"
    (root / "torch-results.json").write_text(json.dumps(answer, indent=2))
    return [[str(x) for x in sample] for sample in answer["results"]]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize(
    "kind,shape,axis",
    [("per_tensor", (2, 8), 0), ("per_channel", (2, 8), 1), ("per_tensor", (), 0), ("per_channel", (0, 3), 1)],
)
def test_candidate_against_pytorch_and_native_mlir(tmp_path, lean_tools, fidelity_tools, width, kind, shape, axis):
    cases, arguments = samples(width, kind, shape, axis)
    expected = pytorch(cases, tmp_path)
    request = {"zero_width": width, "kind": kind, "shape": shape, "axis": axis, "samples": arguments}
    proc, actual = contract(request, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert [[value["bits"] for value in sample["values"]] for sample in actual] == [[sample] for sample in expected]
    module = source_program(width, kind, shape, axis, reference=True)
    document = exported(module, tmp_path, lean_tools)
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


def test_actual_i64_channel_lowering_still_has_integer_promotion_discrepancy(tmp_path, lean_tools, fidelity_tools):
    cases, arguments = samples(64, "per_channel", (2, 8), 1)
    # A finite, in-range witness does not depend on overflow or exceptional floats.
    case, sample = cases[1], arguments[1]
    expected = pytorch([case], tmp_path)
    proc, actual = contract(
        {"zero_width": 64, "kind": "per_channel", "shape": (2, 8), "axis": 1, "samples": [sample]}, tmp_path
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert actual[0]["values"][0]["bits"] == expected[0]
    module = source_program(64, "per_channel", (2, 8), 1, reference=False)
    assert lower_quant_ext(module) == 1
    document = exported(module, tmp_path, lean_tools)
    inputs, outputs = diff.signature(json.loads(document.read_text()), "forward")
    native = diff.execute_mlir(
        tmp_path / "actual.mlir", [sample], tmp_path, fidelity_tools, inputs=inputs, outputs=outputs, function="forward"
    )
    assert native["results"] != expected
    assert native["results"][0] == [
        str(int(x)) for x in np.full((2, 8), -16777215, dtype=np.float32).view(np.uint32).flat
    ]


@pytest.mark.parametrize("change", [{"zero_width": 16}, {"kind": "per_group"}, {"axis": 2}])
def test_candidate_refuses_unsupported_contract(tmp_path, change):
    request = {"zero_width": 64, "kind": "per_channel", "shape": (2, 8), "axis": 1, "samples": [], **change}
    proc, answer = contract(request, tmp_path)
    assert proc.returncode == 2
    assert answer is None
