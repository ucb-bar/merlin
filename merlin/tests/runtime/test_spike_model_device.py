"""Explicit extension selection and byte-exact multi-result Spike readback."""

import os
from types import SimpleNamespace

import pytest

from merlin.runtime.backends import spike_model as S


def test_runner_threads_provider_extension_and_dtc_path(tmp_path, monkeypatch):
    def run(argv, **kwargs):
        assert argv[0] == "/selected/spike"
        assert "--extension=synthetic" in argv and "--extlib=/selected/libdevice.so" in argv
        assert "-l" in argv
        assert kwargs["env"]["PATH"].startswith("/selected/tools" + os.pathsep)
        kwargs["stderr"].write(b"core 0: trace\n")
        return SimpleNamespace(returncode=0, stdout=b"OUT_BYTES 0 1 255\nDONE\n")

    monkeypatch.setattr(S.subprocess, "run", run)
    trace = tmp_path / "trace"
    result = S.run(
        "case.elf",
        extension="synthetic",
        extlib="/selected/libdevice.so",
        spike_binary="/selected/spike",
        path_prepend=["/selected/tools"],
        trace_path=trace,
    )
    assert result["output_bytes"] == [b"\xff"] and trace.read_bytes() == b"core 0: trace\n"


def test_byte_protocol_retains_multiple_types_and_empty_results():
    assert S.parse_console("OUT_BYTES 0 4 0 0 128 63\nOUT_BYTES 1 1 255\nOUT_BYTES 2 0 \nDONE\n")["output_bytes"] == [
        b"\0\0\x80?",
        b"\xff",
        b"",
    ]


@pytest.mark.parametrize(
    "console",
    [
        "OUT_BYTES 0 4 00\nDONE\n",
        "OUT_BYTES 0 0\nOUT_BYTES 0 0\nDONE\n",
        "OUT_BYTES 1 0\nDONE\n",
        "OUT_BYTES 0 1 xx\nDONE\n",
        "OUT_BYTES 0 -1\nDONE\n",
        "DONE\nOUT_BYTES 0 0\n",
        "OUT_BYTES 0 0\n",
        "OUT_BYTES 0 0\nDONE\nDONE\n",
        "OUT_BYTES 0 0\nOUT 0\nDONE\n",
    ],
)
def test_malformed_byte_protocol_fails_closed(console):
    with pytest.raises(S.SpikeModelError):
        S.parse_console(console)


def test_byte_console_retains_numeric_hash_spelling_and_rejects_duplicate_metrics():
    result = S.parse_console("OUT_BYTES 0 0\nMETRIC build_hash 000123456789\nDONE\n")
    assert result["metrics"]["build_hash"] == "000123456789"
    with pytest.raises(S.SpikeModelError, match="duplicate"):
        S.parse_console("OUT_BYTES 0 0\nMETRIC cycles 1\nMETRIC cycles 2\nDONE\n")
