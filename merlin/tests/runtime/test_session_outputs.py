"""Session output assembly checks every step and the recorded byte digest."""

from __future__ import annotations

import hashlib

import pytest

from merlin.llvmlower.session_bundle import MultiProgramSession, Program
from merlin.llvmlower.session_outputs import assembled_output_bytes


def test_assembled_output_bytes_verifies_steps_and_digests(tmp_path):
    program = Program("decode", tmp_path, 2, (), (([1, 2], "f32"),))
    session = MultiProgramSession(tmp_path, {}, (program,), (), "decode", 0)
    dumps = []
    for step, data in enumerate((b"12345678", b"abcdefgh")):
        name = f"p0_s{step}_o0.bin"
        (tmp_path / name).write_bytes(data)
        dumps.append({"program": 0, "step": step, "output": 0,
                      "path": name, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    assert assembled_output_bytes(session, tmp_path, dumps,
                                  program="decode", output=0) == (
                                      b"12345678", b"abcdefgh")
    with pytest.raises(ValueError, match="every step"):
        assembled_output_bytes(session, tmp_path, dumps[:1],
                               program="decode", output=0)
    changed = [dict(row) for row in dumps]
    changed[1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        assembled_output_bytes(session, tmp_path, changed,
                               program="decode", output=0)
    changed = [dict(row) for row in dumps]
    changed[1]["path"] = "../outside.bin"
    with pytest.raises(ValueError, match="ABI"):
        assembled_output_bytes(session, tmp_path, changed,
                               program="decode", output=0)
    (tmp_path / "alias.bin").symlink_to(tmp_path / "p0_s1_o0.bin")
    changed[1]["path"] = "alias.bin"
    with pytest.raises(ValueError, match="regular file"):
        assembled_output_bytes(session, tmp_path, changed,
                               program="decode", output=0)
