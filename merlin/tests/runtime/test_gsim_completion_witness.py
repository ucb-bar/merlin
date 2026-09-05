"""A GSIM muon run must show a completion WITNESS, not merely fail to show a failure.

Measured 2026-09-04: a radiance capsule compiled fork-free, fused into the rv64 SoC carrier and ran
386,090 cycles on the GSIM model. Its console carried none of the four contract markers -- no
``Cycles:``, no ``finished execution``, and equally no ``Timeout exceeded`` and no
``FINISHED: cycles=``. The emulator's own stats line read ``dram_aw=0 dram_w=0 writes_resultpage=0
uart_chars=0``: the kernel wrote nothing to DRAM and printed nothing. The adapter passed it, because
the completion test was a double negative -- it asked only whether the two FAILURE markers were
absent, which cannot tell "the GPU went idle having finished" from "this harness never printed a
word."

That is the recurring shape in this repo: a check that could not run reporting success. The Verilator
sibling has always required its positive marker, so requiring one here brings the two engines of the
same tier to one standard rather than inventing a new bar.
"""
from __future__ import annotations

import pytest

from merlin.runtime.backends.base import get_backend

MO = get_backend("muon").muon_oracles
muon = get_backend("muon").muon


@pytest.mark.parametrize("console, why", [
    pytest.param("", "an empty console", id="empty"),
    pytest.param("[gsim-stats] dram_aw=0 dram_w=0 uart_chars=0\n",
                 "the exact console the measured radiance run produced: the emulator ran and stopped, "
                 "the kernel wrote nothing and printed nothing", id="measured-radiance-shape"),
    pytest.param("C0: 386090 [1] pc=[8000002c] inst=[b7e5]\n", "a carrier spin loop and nothing else",
                 id="spin-only"),
])
def test_a_console_with_no_witness_and_no_failure_is_refused(console, why, monkeypatch, tmp_path):
    """Neither-passed-nor-failed is an unread instrument, and the tier must report unavailable."""
    with pytest.raises(muon.MuonUnavailable) as exc:
        _drive(monkeypatch, tmp_path, console)
    assert "no completion witness" in str(exc.value), why


@pytest.mark.parametrize("witness", ["Cycles: 12345\n", "Muon core 0 finished execution.\n"])
def test_either_positive_witness_is_accepted(witness, monkeypatch, tmp_path):
    res = _drive(monkeypatch, tmp_path, witness)
    assert res["oracle"]["kind"] == "rtl_gsim_muon"


@pytest.mark.parametrize("console", [
    "Cycles: 1\nTimeout exceeded\n",
    "Cycles: 1\nFINISHED: cycles=2000000\n",
])
def test_a_failure_marker_still_loses_even_beside_a_witness(console, monkeypatch, tmp_path):
    """A witness does not rescue a run the RTL watchdog killed or that hit the cap."""
    with pytest.raises(muon.MuonUnavailable) as exc:
        _drive(monkeypatch, tmp_path, console)
    assert "did not reach GPU-idle completion" in str(exc.value)


def _drive(monkeypatch, tmp_path, console: str):
    """Run the real adapter with the compile/fuse/exec seams stubbed, so only grading is exercised."""
    import subprocess

    monkeypatch.setenv("MERLIN_MUON_GSIM_MAXCYCLES", "2000000")
    monkeypatch.setattr(MO, "gsim_status", lambda target: (True, "stub"))
    from merlin.targetgen import gsim_emulator as GE
    monkeypatch.setattr(GE, "emulator_path", lambda *a, **k: tmp_path / "emu")
    monkeypatch.setattr(muon, "is_mlir_artifact", lambda src: True)
    monkeypatch.setattr(muon, "compile_mlir_forkfree", lambda *a, **k: tmp_path / "k.elf")
    monkeypatch.setattr(muon, "fuse_soc_elf", lambda elf, wd: tmp_path / "k.soc.elf")
    monkeypatch.setattr(MO, "flops_from_cb", lambda cb: 0)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(muon, "_read_console", lambda log: (console, len(console), False))
    return MO.gsim_muon_adapter("radiance")({"target": "radiance"}, "mlir", tmp_path, 60)
