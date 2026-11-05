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


# ------------------------------------------------------------------ declared as a spike extension (Kodiak)


def _extension(tmp_path):
    from merlin.targetgen import spike_extension as SE

    return SE.SpikeExtension(target="t", extension_name="unit_model", extlib=tmp_path / "libm.so",
                             library_dir=tmp_path, sha256="0" * 64, declared=True)  # fmt: skip


def _declare_extension(monkeypatch, ext):
    from merlin.targetgen import spike_extension as SE

    monkeypatch.setattr(SE, "declared_extension", lambda target: {"extension_name": ext.extension_name})
    monkeypatch.setattr(SE, "resolve", lambda target, **kw: ext)


def test_a_declared_spike_model_runs_the_shipped_elf_and_must_match_the_twin(tmp_path, monkeypatch):
    """A whole-model build used to run the shipped image only in `op`; the leg is the target's declaration,
    wherever the image is built."""
    import numpy as np

    ext = _extension(tmp_path)
    _declare_extension(monkeypatch, ext)
    seen = {}

    def run(elf, **kw):
        seen.update(kw, elf=elf)
        return {"output": np.arange(4.0), "prefix": np.arange(4.0), "metrics": {"cycles": 9}, "out_hash": {"v": 1}}

    monkeypatch.setattr(zm, "run_on_spike", run)
    monkeypatch.setattr(zm, "_gate", lambda out, refs: {"ok": True, "w8a8_cos": 1.0})
    twin = {"prefix": np.arange(4.0), "out_hash": {"v": 1}}
    got = matrix_leg._run_shipped_on_device_simulator(_pair(tmp_path), _routing(SimpleNamespace()), _BRD, cpus=2,
                                                       timeout=1, refs={"r": 1}, twin=twin)  # fmt: skip
    assert seen["elf"] == tmp_path / "ship.elf" and seen["extension"] is ext and seen["harts"] == 2
    assert got["status"] == "verified" and got["via"] == "spike_extension" and got["bit_exact_vs_twin"]

    twin = {"prefix": np.arange(4.0) + 1, "out_hash": {"v": 2}}  # a datapath that computes something else
    got = matrix_leg._run_shipped_on_device_simulator(_pair(tmp_path), _routing(SimpleNamespace()), _BRD, cpus=2,
                                                       timeout=1, refs={"r": 1}, twin=twin)  # fmt: skip
    assert got["status"] == "run_mismatch" and got["bit_exact_vs_twin"] is False
    assert got["status"] in matrix_leg.SHIPPED_FAILURES


def test_a_declared_spike_model_that_cannot_be_loaded_is_not_run_and_never_replaced(tmp_path, monkeypatch):
    from merlin.targetgen import spike_extension as SE

    def refuse(target, **kw):
        raise SE.SpikeExtensionError("digest mismatch")

    monkeypatch.setattr(SE, "declared_extension", lambda target: {"extension_name": "m"})
    monkeypatch.setattr(SE, "resolve", refuse)
    monkeypatch.setattr(zm, "run_on_spike", lambda *a, **k: pytest.fail("ran on an unverified model"))
    got = matrix_leg._run_shipped_on_device_simulator(_pair(tmp_path), _routing(SimpleNamespace()), _BRD, cpus=1,
                                                       timeout=1, refs=None)  # fmt: skip
    assert got == {"status": "not_run", "reason": "digest mismatch"}


def test_a_shipped_image_that_prints_no_output_fails(tmp_path, monkeypatch):
    _declare_extension(monkeypatch, _extension(tmp_path))
    monkeypatch.setattr(zm, "run_on_spike", lambda *a, **k: {"prefix": None, "console": "FAIL fatal mcause=2"})
    got = matrix_leg._run_shipped_on_device_simulator(_pair(tmp_path), _routing(SimpleNamespace()), _BRD, cpus=1,
                                                       timeout=1, refs=None)  # fmt: skip
    assert got["status"] == "run_failed" and "mcause=2" in got["reason"]
