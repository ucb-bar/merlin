"""The shipped matrix image's run on a simulator that implements the unit: declared, or not run.

The leg exists because the stand-in twin replaces the unit's datapath, so a twin that agrees says nothing
about the unit's words. A provider names a simulator that executes them; absent that declaration, or
absent its binary, the leg reports why and never falls back to a simulator without the unit.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.compile import matrix_leg
from merlin.runtime.backends import zephyr_model as zm


def _pair(tmp_path):
    return SimpleNamespace(ship={"elf": tmp_path / "ship.elf", "ram_bytes": 1 << 20})


def _routing(provider):
    return SimpleNamespace(provider=lambda: provider, unit="u", config="c", support_target="t")


_BRD = SimpleNamespace(vlen=256, dram_base=0x80000000)


def test_no_declaration_is_reported_not_run_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(zm, "run_on_spike", lambda *a, **k: pytest.fail("must not run"))
    got = matrix_leg._run_shipped_on_device_simulator(
        _pair(tmp_path), _routing(SimpleNamespace()), _BRD, cpus=1, timeout=1, refs=None
    )
    assert got["status"] == "not_declared"


def test_a_declared_but_absent_simulator_is_not_run_with_its_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(zm, "run_on_spike", lambda *a, **k: pytest.fail("must not fall back"))
    provider = SimpleNamespace(device_simulator=lambda **k: {"available": False, "reason": "set $SIM"})
    got = matrix_leg._run_shipped_on_device_simulator(_pair(tmp_path), _routing(provider), _BRD, cpus=1,
                                                       timeout=1, refs=None)  # fmt: skip
    assert got == {"status": "not_run", "reason": "set $SIM"}


def test_a_declared_simulator_runs_the_shipped_elf_with_the_units_isa(tmp_path, monkeypatch):
    seen = {}

    def run(elf, **kw):
        seen.update(kw, elf=elf)
        return {"output": [1.0], "prefix": [1.0], "metrics": {"cycles": 7}}

    monkeypatch.setattr(zm, "run_on_spike", run)
    monkeypatch.setattr(zm, "_gate", lambda out, refs: {"ok": False, "w8a8_cos": 0.5})
    provider = SimpleNamespace(
        device_simulator=lambda **k: {"available": True, "binary": "/sim", "isa_extensions": ["zx1", "zx2"]}
    )
    got = matrix_leg._run_shipped_on_device_simulator(_pair(tmp_path), _routing(provider), _BRD, cpus=2,
                                                       timeout=1, refs={"r": 1})  # fmt: skip
    assert seen["elf"] == tmp_path / "ship.elf" and str(seen["spike"]) == "/sim" and seen["harts"] == 2
    assert seen["isa"] == zm.spike_isa(256) + "_zx1_zx2"
    assert got["status"] == "run_mismatch" and got["spike_cycles"] == 7
