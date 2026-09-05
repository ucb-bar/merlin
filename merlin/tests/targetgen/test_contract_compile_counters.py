from __future__ import annotations

from types import SimpleNamespace

from merlin.perf import hw_counters
from merlin.targetgen.contract import compile as contract_compile


def test_oracle_boundary_preserves_named_hardware_counter_readings(monkeypatch, tmp_path):
    console = f"{hw_counters.COUNTER_MARKER} OPAQUE_BYTES 73\nDONE\n"
    backend = SimpleNamespace(
        run_elf=lambda *args, **kwargs: console,
        parse_output=lambda text: ({}, {"cycles": 19}),
        ORACLE={"rtl": {"derived_from_rtl": True}},
    )
    monkeypatch.setattr(contract_compile, "compile_lowered_to_elf",
                        lambda *args, **kwargs: tmp_path / "program.elf")
    from merlin.runtime.backends import base
    monkeypatch.setattr(base, "get_backend", lambda target: backend)
    monkeypatch.setattr(hw_counters, "counters_for_target",
                        lambda target: {"status": "absent", "why": "synthetic"})

    got = contract_compile.run_on_oracle(
        {}, "module {}", simulator="rtl", target="synthetic", workdir=tmp_path)

    assert got["counters"]["readings"] == {"OPAQUE_BYTES": 73}
    assert got["counters"]["discovery"]["status"] == "absent"
