from __future__ import annotations

from types import SimpleNamespace

from merlin.perf import hw_counters
from merlin.targetgen.contract import compile as contract_compile


def _oracle(monkeypatch, tmp_path, *, simulator):
    console = f"{hw_counters.COUNTER_MARKER} OPAQUE_BYTES 73\nDONE\n"
    backend = SimpleNamespace(
        run_elf=lambda *args, **kwargs: console,
        parse_output=lambda text: ({}, {"cycles": 19}),
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
