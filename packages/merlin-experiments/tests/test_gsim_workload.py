"""Logical tensor byte encoding is independent of native certificate execution."""

import re
import struct
import subprocess

import pytest
from merlin_experiments.phase2 import gsim_workload as W


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native launch"))


@pytest.mark.parametrize("dtype", ["i8", "u16", "i1", "i٠٨", "u１６", "i8\n", "i", "i+8", "i²", "i0", "i7"])
def test_structural_integer_parser_preserves_historical_grammar(dtype):
    match = re.fullmatch(r"([iu])(\d+)", dtype)
    bits = int(match.group(2)) if match else 0
    if match and (bits == 1 or bits > 0 and bits % 8 == 0):
        width = 8 if bits == 1 else bits
        assert W._encode_scalar(1, dtype) == (1).to_bytes(width // 8, "little", signed=match.group(1) == "i")
    else:
        with pytest.raises(W.ProducerError):
            W._encode_scalar(1, dtype)


def test_declared_outputs_preserve_bytes_and_refuse_shape_mismatch():
    cb = {"tensors": {"y": {"role": "output", "shape": [2], "dtype": "f32"}}}
    _, rows = W.encode_declared_outputs({"y": [[0.0, -0.0]]}, cb)
    import hashlib

    assert rows[0]["sha256"] == hashlib.sha256(struct.pack("<ff", 0.0, -0.0)).hexdigest()
    with pytest.raises(W.ProducerError, match="requires 2"):
        W.encode_declared_outputs({"y": [0.0]}, cb)
