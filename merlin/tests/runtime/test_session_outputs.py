"""Session output assembly checks every step and the recorded byte digest."""

from __future__ import annotations

import hashlib

import pytest

from merlin.llvmlower.session_bundle import MultiProgramSession, Program
from merlin.llvmlower.session_outputs import assembled_output_bytes, observed_output_dump_bytes


def console_fixture(tmp_path):
    program = Program("decode", tmp_path, 2, (), (([1, 2], "f32"),))
    session = MultiProgramSession(tmp_path, {}, (program,), (), "decode", 0)
    values = (b"12345678", b"abcdefgh")
    digests = [{"program": 0, "step": step, "output": 0, "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()} for step, raw in enumerate(values)]
    log = "\n".join(f"SESSION_HEX 0 {step} 0 {len(raw)} {raw.hex()}"
                    for step, raw in enumerate(values)) + "\n"
    return session, log, digests, values


def test_console_bytes_can_be_reopened_without_creating_any_files(tmp_path):
    session, log, digests, values = console_fixture(tmp_path)
    actual = observed_output_dump_bytes(log, session, [{"program": 0, "output": 0}], digests, byte_limit=8)
    assert actual == {(0, step, 0): raw for step, raw in enumerate(values)}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mutation", ["changed", "missing", "duplicate", "extra", "limit", "hex",
                                    "duplicate_digest", "duplicate_selection", "wrong_step", "negative"])
def test_console_reader_refuses_partial_ambiguous_and_changed_bytes(tmp_path, mutation):
    session, log, digests, _ = console_fixture(tmp_path)
    selected, limit = [{"program": 0, "output": 0}], 8
    if mutation == "changed":
        log = log.replace("31323334", "01020304")
    elif mutation == "missing":
        log = log.splitlines()[0] + "\n"
    elif mutation == "duplicate":
        log += log.splitlines()[0] + "\n"
    elif mutation == "extra":
        log += log.splitlines()[0].replace("HEX 0 0", "HEX 0 2") + "\n"
    elif mutation == "limit":
        limit = 7
    elif mutation == "hex":
        log = log.replace("31323334", "zzzzzzzz")
    elif mutation == "duplicate_digest":
        digests.append(digests[0])
    elif mutation == "duplicate_selection":
        selected.append(selected[0])
    elif mutation == "wrong_step":
        digests[0]["step"] = -1
    else:
        log = log.replace(" 8 ", " -8 ")
    with pytest.raises(ValueError):
        observed_output_dump_bytes(log, session, selected, digests, byte_limit=limit)
    assert not list(tmp_path.iterdir())


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
