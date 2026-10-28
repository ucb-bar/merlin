"""The worker's JSON reference retains exact native integer and boolean values."""

import json
import hashlib
import math
import os
import subprocess
from pathlib import Path

import pytest

from merlin.targetgen import _m2m_capture_worker as worker


def test_bundle_abi_restores_only_equivalent_complex_storage_spellings(tmp_path):
    from merlin.targetgen._capture_result_contract import restore_bundle_abi

    inputs = [{"shape": [2], "dtype": "complex<f32>"}]
    outputs = [{"shape": [3], "dtype": "complex<f64>"}, {"shape": [], "dtype": "i1"}]
    meta = {"input_abi": [{"shape": [2], "dtype": "complex64"}],
            "output_abi": [{"shape": [3], "dtype": "complex128"}, outputs[1]], "other": 7}
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta))
    restore_bundle_abi(tmp_path, inputs, outputs)
    assert json.loads(path.read_text()) == {**meta, "input_abi": inputs, "output_abi": outputs}
    for wrong in ([], [{"shape": [4], "dtype": "complex64"}], [{"shape": [2], "dtype": "complex128"}], [None]):
        path.write_text(json.dumps({**meta, "input_abi": wrong}))
        with pytest.raises(ValueError, match="bundle input_abi"):
            restore_bundle_abi(tmp_path, inputs, outputs)


LOADER = """
import os
import torch

class Reference(torch.nn.Module):
    def __init__(self, case):
        super().__init__()
        self.case = case

    def forward(self, x):
        if self.case == "upper":
            return torch.arange(2**63 - 14, 2**63 - 9, dtype=torch.int64, device=x.device)
        if self.case == "lower":
            return torch.arange(-(2**63) + 9, -(2**63) + 14, dtype=torch.int64, device=x.device)
        return x

def get_model_and_inputs():
    case = os.environ["M2M_JSON_REFERENCE_CASE"]
    if case == "float":
        x = torch.tensor([-0.0, 1.25], dtype=torch.float32)
    elif case == "bool":
        x = torch.tensor([True, False], dtype=torch.bool)
    elif case == "upper":
        x = torch.tensor([2**63 - 14, 2**63 - 10], dtype=torch.int64)
    else:
        x = torch.tensor([-(2**63) + 9, -(2**63) + 13], dtype=torch.int64)
    return Reference(case).eval(), (x,)
"""


@pytest.mark.parametrize("case", ("upper", "lower", "float", "bool"))
def test_materialized_worker_json_reference_preserves_native_values(tmp_path, case):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    m2m_dir = os.environ.get("MERLIN_M2M_DIR")
    if not python or not m2m_dir:
        pytest.skip("an explicit materializing M2M interpreter and source are required")
    loader = tmp_path / "loader.py"
    loader.write_text(LOADER, encoding="utf-8")
    result = subprocess.run(
        [
            python,
            str(Path(worker.__file__)),
            "--m2m-dir",
            m2m_dir,
            "--loader",
            str(loader),
            "--dtype",
            "fp32",
            "--seed",
            "0",
            "--materialize-bundle",
            "--out",
            "capture",
        ],
        cwd=tmp_path,
        env={**os.environ, "M2M_JSON_REFERENCE_CASE": case, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    capture = tmp_path / "capture"
    assert (capture / "capture_receipt.json").is_file()
    receipt = json.loads((capture / "capture_receipt.json").read_text())
    meta_bytes = (capture / "meta.json").read_bytes()
    assert receipt["artifacts"]["meta.json"] == {
        "bytes": len(meta_bytes), "sha256": hashlib.sha256(meta_bytes).hexdigest()
    }
    values = json.loads((capture / "golden.json").read_text(encoding="utf-8"))
    inputs = json.loads((capture / "inputs.json").read_text(encoding="utf-8"))
    if case == "upper":
        expected = list(range(2**63 - 14, 2**63 - 9))
        expected_input = [2**63 - 14, 2**63 - 10]
    elif case == "lower":
        expected = list(range(-(2**63) + 9, -(2**63) + 14))
        expected_input = [-(2**63) + 9, -(2**63) + 13]
    elif case == "bool":
        expected = [True, False]
        expected_input = expected
    else:
        expected = [-0.0, 1.25]
        expected_input = expected
    assert values == expected
    assert inputs == [expected_input]
    if case == "float":
        assert math.copysign(1.0, values[0]) == -1.0
        assert math.copysign(1.0, inputs[0][0]) == -1.0
    elif case == "bool":
        assert all(type(value) is bool for value in values)
        assert all(type(value) is bool for value in inputs[0])
    else:
        assert all(type(value) is int for value in values)
        assert all(type(value) is int for value in inputs[0])
