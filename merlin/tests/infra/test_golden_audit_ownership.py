"""Answer-bearing research stays optional without changing its native replay output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from merlin.common.paths import repo_root


def test_core_rtl_namespace_excludes_private_replay_and_numerical_audit(tmp_path):
    script = """
import importlib.util
import sys
sys.path.insert(0, sys.argv[1])
from merlin.targetgen import rtl
assert importlib.util.find_spec('merlin.targetgen.numeric_falsifiability') is None
assert importlib.util.find_spec('merlin.targetgen.rtl.gen_rocc_replay') is None
assert importlib.util.find_spec('merlin.targetgen.rtl.replay_json_to_h') is not None
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(repo_root() / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_private_audit_modules_have_one_experiments_owner():
    from merlin.targetgen import numeric_falsifiability
    from merlin.targetgen.rtl import gen_rocc_replay

    owner = repo_root() / "packages/merlin-experiments/src"
    for module in (numeric_falsifiability, gen_rocc_replay):
        assert Path(module.__file__).is_relative_to(owner)


def test_relocated_replay_preserves_padding_goldens_and_trace_abi(tmp_path, monkeypatch):
    from merlin.runtime.tensor import Tensor
    from merlin.targetgen.rtl import gen_rocc_replay as replay

    capsule = tmp_path / "capsule.yaml"
    trace = tmp_path / "trace.json"
    output = tmp_path / "replay.json"
    capsule.write_text(
        yaml.safe_dump(
            {
                "name": "synthetic",
                "inputs": [{"name": "A", "role": "input", "shape": [1, 2]}],
                "outputs": [{"name": "Y", "role": "output"}],
            }
        )
    )
    trace.write_text(
        json.dumps(
            {
                "abi": {"custom_opcode": "0x7b"},
                "instructions": [
                    {"class": "CONFIG_ST", "funct": 1, "decoded": {"out_stride_bytes": 32}},
                    {"class": "FENCE", "funct": None},
                    {"class": "MVOUT", "funct": 2, "rs1": 4, "rs2": 8, "decoded": {"readout": "i8"}},
                ],
            }
        )
    )
    monkeypatch.setattr(replay.CG, "materialize_capsule_leaves", lambda _: {"A": Tensor((1, 2), [1, -2], "i8")})
    monkeypatch.setattr(replay.CG, "golden", lambda _: {"Y": [[7, -9]]})
    assert replay.main([str(capsule), str(trace), "--out", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["rocc_opcode"] == 0x7B
    assert result["placements"][0]["bytes_hex"] == (bytes([1, 254]) + bytes(14)).hex()
    assert result["placements"][0]["row_stride"] == 16
    assert result["outputs"][0]["golden_flat"] == [7, -9]
    assert result["outputs"][0]["elem_bytes"] == 1
    assert result["outputs"][0]["stride_bytes"] == 32
    assert result["insns"][1] == {"class": "FENCE", "funct": None}
    assert result["insns"][2] == {"class": "MVOUT", "funct": 2, "rs1": 4, "rs2": 8}
