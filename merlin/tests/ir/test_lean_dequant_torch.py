"""Bounded source-model fidelity: actual pinned PyTorch, independent NumPy, and real MLIR folds.

Set MERLIN_TORCH_PYTHON to the isolated CPU oracle interpreter; this does not qualify a capture
producer version or replace the parametric Lean proof. NaN payload and IEEE-instance qualification
remain outside this finite-input differential test.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_lean_dequant_scalar import folded_bits, lowered, scalarize

from merlin.common.paths import repo_root


def test_pinned_torch_source_and_mlir_agree(tmp_path, lean_tools):
    selected = os.environ.get("MERLIN_TORCH_PYTHON")
    if not selected:
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail("MERLIN_TORCH_PYTHON is required for the source-model differential check")
        pytest.skip("set MERLIN_TORCH_PYTHON to run the pinned CPU PyTorch source oracle")
    interpreter = Path(selected).absolute()
    assert interpreter.is_file(), interpreter
    rng = np.random.default_rng(1701)
    cases, expected = [], []
    shapes = [(), (8,), (2, 3), (2, 3, 4), (0, 3)]
    for kind in ("per_tensor", "per_channel"):
        for shape in shapes:
            if kind == "per_channel" and not shape:
                continue
            for repetition in range(7):
                axis = (repetition % len(shape)) if shape else 0
                param_shape = (shape[axis],) if kind == "per_channel" else ()
                q = rng.integers(-128, 128, size=shape, dtype=np.int8)
                z = rng.integers(-(1 << 31), 1 << 31, size=param_shape, dtype=np.int32)
                s = rng.uniform(-2, 2, size=param_shape).astype(np.float32)
                if repetition == 0:
                    q.fill(1)
                    z.fill(16777217)
                    s.fill(1.0)
                elif repetition == 1:
                    q.fill(-128)
                    z.fill((1 << 31) - 1)
                    s.fill(1.0)
                elif repetition == 2:
                    s.fill(-0.0)
                elif repetition == 3:
                    q.fill(127)
                    z.fill(-(1 << 31))
                    s.fill(1.0)
                if kind == "per_channel":
                    broadcast = [1] * len(shape)
                    broadcast[axis] = shape[axis]
                    # The subtraction is explicitly i32, including modular overflow.
                    value = (q.astype(np.int32) - z.reshape(broadcast)).astype(np.float32) * s.reshape(broadcast)
                else:
                    value = (q.astype(np.float32) - z.astype(np.float32)) * s
                expected.append(value.reshape(-1).view(np.uint32).tolist())
                cases.append(
                    {
                        "kind": kind,
                        "shape": shape,
                        "axis": axis,
                        "q": q.reshape(-1).tolist(),
                        "z": z.reshape(-1).tolist(),
                        "scale_bits": s.reshape(-1).view(np.uint32).tolist(),
                    }
                )
    oracle = repo_root() / "merlin" / "tests" / "ir" / "lean_torch_dequant_oracle.py"
    proc = subprocess.run(
        [str(interpreter), str(oracle)], input=json.dumps(cases), capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    observed = json.loads(proc.stdout)
    assert observed["torch"] == "2.10.0+cpu"
    assert observed["results"] == expected
    assert len(cases) == 63
    assert sum(map(len, expected)) == 539
    # Compare selected points to scalarized *actual compiler output* folded by authoritative MLIR.
    compared = 0
    for case, bits in zip(cases, observed["results"], strict=True):
        if not bits:
            continue
        _, generic = lowered(case["kind"])
        scale = np.array(case["scale_bits"][:1], dtype=np.uint32).view(np.float32)[0]
        module = scalarize(generic, case["q"][0], case["z"][0], float(scale))
        assert folded_bits(module, tmp_path, lean_tools) == bits[0], case
        compared += 1
    assert compared == 49
