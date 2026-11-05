from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.perf import hw_counters
from merlin.targetgen.contract import compile as contract_compile


def _oracle(monkeypatch, tmp_path, *, simulator, metrics=None):
    console = f"{hw_counters.COUNTER_MARKER} OPAQUE_BYTES 73\nDONE\n"
    backend = SimpleNamespace(
        run_elf=lambda *args, **kwargs: console,
        parse_output=lambda text: ({}, {"cycles": 19} if metrics is None else metrics),
        ORACLE={simulator: {"derived_from_rtl": True}},
    )
    monkeypatch.setattr(contract_compile, "compile_lowered_to_elf",
                        lambda *args, **kwargs: tmp_path / "program.elf")
    from merlin.runtime.backends import base
    monkeypatch.setattr(base, "get_backend", lambda target: backend)
    monkeypatch.setattr(hw_counters, "counters_for_target",
                        lambda target: {"status": "absent", "why": "synthetic"})
    return contract_compile.run_on_oracle(
        {}, "module {}", simulator=simulator, target="synthetic", workdir=tmp_path)


def test_oracle_boundary_preserves_named_hardware_counter_readings(monkeypatch, tmp_path):
    """A trusted engine's readings survive the boundary under their own names.

    ``verilator`` is declared ``real`` in merlin/contract/counter_trust.yaml -- same elaborated RTL
    as the FPGA flow, so the same connected counter events.
    """
    got = _oracle(monkeypatch, tmp_path, simulator="verilator")
    assert got["counters"]["readings"] == {"OPAQUE_BYTES": 73}
    assert got["counters"]["discovery"]["status"] == "absent"


def test_an_undeclared_engines_readings_are_refused_with_the_reason(monkeypatch, tmp_path):
    """THE PROPERTY THAT MATTERS MORE, and the one this file did not have.

    An engine absent from the trust contract is UNKNOWN, and UNKNOWN is refused rather than
    reported. The failure that gate exists to prevent is specific: an optimization loop read
    accelerator occupancy counters off a functional ISS that FABRICATES them with rand(), so its
    only feedback signal was noise and nothing in the receipt said so. The refusal must therefore
    reach the receipt where the numbers would have been -- an absent field reads as "not collected",
    which is the same shape as the original defect.
    """
    got = _oracle(monkeypatch, tmp_path, simulator="an_engine_nobody_declared")
    assert got["counters"]["status"] == "unknown"
    assert got["counters"]["readings"] is None
    assert got["counters"]["why"], "the refusal must carry a reason, not just a None"
    assert got["counters"]["engine"]["verdict"] not in ("real",)


def test_a_fabricating_engines_readings_are_refused(monkeypatch, tmp_path):
    """spike is declared ``fabricated``: it increments every counter with rand()."""
    got = _oracle(monkeypatch, tmp_path, simulator="spike")
    assert got["counters"]["readings"] is None
    assert got["counters"]["engine"]["verdict"] == "fabricated"
    assert "rand()" in got["counters"]["engine"].get("evidence", "") or got["counters"]["why"]


def test_failed_readback_retains_complete_native_console(monkeypatch, tmp_path):
    """Exercise the real compile/run/parse seam, not a replacement decoder.

    A complete native invocation may return an incomplete frame. The refusal
    must remain a refusal, but retain every returned byte for diagnosis rather
    than discarding the transcript when parse_output raises.
    """
    from merlin.runtime.backends import base

    console = (
        "%Warning: native diagnostic\n"
        "OUT_B64_BEGIN v1 Y0 1 4 1 s\n"
        "OUT_B64_CHUNK 00000000 0004 gP8Afw==\n"
    )
    elf = tmp_path / "program.elf"
    elf.write_bytes(b"synthetic execution artifact")
    backend = SimpleNamespace(
        run_elf=lambda *args, **kwargs: console,
        parse_output=lambda text: base.parse_console(text, strip_warnings=True),
    )
    monkeypatch.setattr(contract_compile, "compile_lowered_to_elf", lambda *args, **kwargs: elf)
    monkeypatch.setattr(base, "get_backend", lambda target: backend)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="OUT_B64 output ended without END"):
            contract_compile.run_on_oracle(
                {}, "module {}", simulator="synthetic", target="synthetic", workdir=tmp_path
            )
        assert (tmp_path / "oracle_console.log").read_bytes() == console.encode("utf-8")


def test_successful_readback_retains_console_without_changing_result(monkeypatch, tmp_path):
    result = _oracle(monkeypatch, tmp_path, simulator="verilator")
    assert (tmp_path / "oracle_console.log").read_bytes() == result["console"].encode("utf-8")


def test_missing_cycle_marker_is_unmeasured_not_zero(monkeypatch, tmp_path):
    result = _oracle(monkeypatch, tmp_path, simulator="verilator", metrics={})
    assert result["raw_metrics"] == {}
    assert result["cycles"] is None


def test_zero_cycle_marker_is_not_accepted_as_a_measurement(monkeypatch, tmp_path):
    result = _oracle(monkeypatch, tmp_path, simulator="verilator", metrics={"cycles": 0})
    assert result["raw_metrics"] == {"cycles": 0}
    assert result["cycles"] is None


def test_failed_launch_does_not_retain_an_earlier_attempts_console(monkeypatch, tmp_path):
    from merlin.runtime.backends import base

    _oracle(monkeypatch, tmp_path, simulator="verilator")
    previous = tmp_path / "oracle_console.log"
    previous_stderr = tmp_path / "oracle_stderr.log"
    previous_stderr.write_bytes(b"stale diagnostic")
    assert previous.is_file()

    def refused(*args, **kwargs):
        raise RuntimeError("native launch refused")

    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(run_elf=refused))
    with pytest.raises(RuntimeError, match="native launch refused"):
        contract_compile.run_on_oracle(
            {}, "module {}", simulator="synthetic", target="synthetic", workdir=tmp_path
        )
    assert not previous.exists()
    assert not previous_stderr.exists()


@pytest.mark.parametrize("failure", ["timeout", "nonzero"])
@pytest.mark.parametrize("binary", [True, False])
def test_process_failure_retains_partial_console_without_granting_a_result(
    monkeypatch, tmp_path, failure, binary
):
    import subprocess

    from merlin.runtime.backends import base

    stdout = b"OUT_B64_BEGIN v1 Y0 1 4 1 s\npartial\xff"
    stderr = b"native execution diagnostic\xfe"
    if not binary:
        stdout, stderr = stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
    error = (
        subprocess.TimeoutExpired(["synthetic"], 1, output=stdout, stderr=stderr)
        if failure == "timeout"
        else subprocess.CalledProcessError(124, ["synthetic"], output=stdout, stderr=stderr)
    )

    def refused(*args, **kwargs):
        raise error

    monkeypatch.setattr(contract_compile, "compile_lowered_to_elf", lambda *args, **kwargs: tmp_path / "program.elf")
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(run_elf=refused))
    (tmp_path / "oracle_console.log").write_bytes(b"stale stdout")
    (tmp_path / "oracle_stderr.log").write_bytes(b"stale stderr")
    with pytest.raises(type(error)) as caught:
        contract_compile.run_on_oracle(
            {}, "module {}", simulator="synthetic", target="synthetic", workdir=tmp_path
        )
    assert caught.value is error
    for name, expected in (("oracle_console.log", stdout), ("oracle_stderr.log", stderr)):
        assert (tmp_path / name).read_bytes() == (expected if binary else expected.encode("utf-8"))
