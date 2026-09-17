"""The chipyard backend can answer at the elaborated-RTL tier with GSIM, not only Verilator.

`rtl_engine_policy` already selects the L3 engine by availability (vcs > gsim > verilator), and
`chipyard_l3_selection` probes with the backend's own `available(engine)`. So the whole distance between
GSIM and being selected for this target was one probe answering False-because-absent instead of raising
unknown-simulator, plus a `run_elf` branch. These tests pin both halves, and the distinction between them:

  * an engine the backend KNOWS but cannot find -> False (the binary is missing; go build it),
  * an engine the backend does NOT know        -> raise (this target has no such engine at all).

Collapsing those two into one answer is what makes an unavailable cert tier unreadable afterwards.

Everything here runs with NO GSIM build present: the base toolchain probes and `subprocess.run` are
substituted, because the assertions are about the argv and the availability contract, not about silicon.
"""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from merlin.runtime.backends import base as _backends


@pytest.fixture()
def G():
    """The gemmini backend MODULE (not the re-exporting package): `run_elf` resolves `gsim_path` and
    friends through its own globals, so a monkeypatch has to land there to be seen."""
    return _backends.get_backend("gemmini").gemmini


@pytest.fixture()
def toolchain_present(monkeypatch, G, tmp_path):
    """Make the gcc + bare-metal-harness half of `available()` True independently of this host.

    Without this the tests would pass on a machine with chipyard installed and skip-by-accident (assert
    False == False for the wrong reason) on one without it.
    """
    common = tmp_path / "common"
    common.mkdir()
    (common / "test.ld").write_text("/* link script */\n", encoding="utf-8")
    gcc = tmp_path / "riscv64-unknown-elf-gcc"
    gcc.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(G, "gcc_path", lambda: gcc)
    monkeypatch.setattr(G, "_common_dir", lambda: common)
    monkeypatch.setattr(G, "_test_ld", lambda: common / "test.ld")


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\necho DONE\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# --- availability -----------------------------------------------------------------------------------

def test_gsim_is_unavailable_not_an_exception_when_nothing_is_built(monkeypatch, G, tmp_path,
                                                                    toolchain_present):
    """The probe in `chipyard_l3_selection` calls this for EVERY engine in the priority list. A raise on
    the common case (no GSIM build on this host) would be recorded as `probe raised ...`, which reads as
    a broken backend rather than an absent binary.

    "Nothing is built" now means BOTH sources empty: the env override AND the derived home under the
    build root. Redirecting the output root is how the assertion stays about the code instead of about
    whether this particular checkout has an emulator installed."""
    monkeypatch.delenv("MERLIN_GEMMINI_GSIM_EMU", raising=False)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    assert G.available("gsim") is False
    ok, why = G.gsim_status()
    assert ok is False
    # Actionable: it names the place a build would be installed, because the reader is who would build it.
    assert "rtl_engines" in why and "gsim" in why


def test_gsim_is_available_when_the_env_points_at_an_executable(monkeypatch, G, tmp_path,
                                                                toolchain_present):
    emu = _executable(tmp_path / "emu")
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_EMU", str(emu))
    assert G.gsim_path() == emu
    assert G.available("gsim") is True


def test_a_non_executable_emu_is_not_available(monkeypatch, G, tmp_path, toolchain_present):
    """The env var can be pointed at anything -- an artifact copied without its mode bit, or the emitted
    .cpp rather than the built binary. Existence alone would report a cert tier that cannot run."""
    emu = tmp_path / "emu.cpp"
    emu.write_text("int main() { return 0; }\n", encoding="utf-8")
    emu.chmod(emu.stat().st_mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_EMU", str(emu))
    assert G.available("gsim") is False


def test_an_unknown_engine_still_raises(G):
    """Adding gsim must not turn every unknown name into a bland False: `test_chipyard_l3_engine_is_selected`
    depends on an engine this backend does not implement being distinguishable from one whose build is
    merely absent."""
    with pytest.raises(G.GemminiError):
        G.available("vcs")


def test_gsim_carries_an_rtl_derived_oracle_record(G):
    """`contract.compile.run_on_oracle` indexes `backend.ORACLE[simulator]`. A selectable engine with no
    entry is not "missing metadata" -- it is a KeyError after the ELF has already been built and run."""
    assert G.ORACLE["gsim"]["derived_from_rtl"] is True


# --- invocation -------------------------------------------------------------------------------------

def _capture(monkeypatch, G, stdout: str = "DONE\n"):
    seen: dict = {}

    class _Proc:
        returncode = 0
        stderr = ""

        def __init__(self, out):
            self.stdout = out

    def fake_run(cmd, **kw):
        seen["cmd"], seen["kw"] = list(cmd), kw
        return _Proc(stdout)

    monkeypatch.setattr(G.subprocess, "run", fake_run)
    return seen


def test_run_elf_builds_the_documented_gsim_argv(monkeypatch, G, tmp_path):
    """`+loadmem` is the image backdoor (GSIM re-roots at ChipTop, so there is no SimTSI to load the ELF)
    and it is passed BESIDE the positional path, not instead of it."""
    emu = _executable(tmp_path / "emu")
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_EMU", str(emu))
    monkeypatch.delenv("MERLIN_GEMMINI_GSIM_MAXCYCLES", raising=False)
    elf = tmp_path / "kernel.elf"
    elf.write_text("", encoding="utf-8")
    seen = _capture(monkeypatch, G, stdout="OUT y 1 1 7\nMETRIC cycles 3\nDONE\n")

    console = G.run_elf(elf, simulator="gsim", timeout=42)

    assert seen["cmd"] == [str(emu), str(elf), f"+max-cycles={G.GSIM_MAX_CYCLES}", f"+loadmem={elf}"]
    assert seen["kw"]["timeout"] == 42
    # The SAME console protocol as Verilator, which is the point: no parser learns a second dialect.
    outputs, raw = G.parse_output(console)
    assert outputs["y"] == [[7]] and raw["cycles"] == 3


def test_the_cycle_cap_is_an_env_override(monkeypatch, G, tmp_path):
    """The cap is the hang bound for a long capsule; raising it must not require editing the backend."""
    emu = _executable(tmp_path / "emu")
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_EMU", str(emu))
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_MAXCYCLES", "12345")
    elf = tmp_path / "kernel.elf"
    elf.write_text("", encoding="utf-8")
    seen = _capture(monkeypatch, G)

    G.run_elf(elf, simulator="gsim")

    assert "+max-cycles=12345" in seen["cmd"]
    assert G.gsim_max_cycles() == "12345"


def test_a_nonzero_gsim_exit_is_an_error_not_a_silent_empty_console(monkeypatch, G, tmp_path):
    """Same fail-closed shape as the Verilator path: an emulator that died must not present as a capsule
    that produced no OUT lines."""
    emu = _executable(tmp_path / "emu")
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_EMU", str(emu))
    elf = tmp_path / "kernel.elf"
    elf.write_text("", encoding="utf-8")

    class _Proc:
        returncode = 3
        stdout = "boom"
        stderr = "assert"

    monkeypatch.setattr(G.subprocess, "run", lambda cmd, **kw: _Proc())
    with pytest.raises(G.GemminiError):
        G.run_elf(elf, simulator="gsim")


def test_run_elf_still_rejects_an_unknown_simulator(G, tmp_path):
    elf = tmp_path / "kernel.elf"
    elf.write_text("", encoding="utf-8")
    with pytest.raises(G.GemminiError):
        G.run_elf(elf, simulator="vcs")


# --- the selection this unlocks ---------------------------------------------------------------------

def test_the_l3_policy_now_selects_gsim_for_this_target(monkeypatch, G, tmp_path, toolchain_present):
    """The end-to-end claim: with a GSIM emu present, the chipyard L3 tier resolves to it over Verilator,
    with no edit to the selection code."""
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import rtl_engine_policy as POL

    emu = _executable(tmp_path / "emu")
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_EMU", str(emu))
    sel = CR.chipyard_l3_selection("gemmini")
    assert sel["engine"] == "gsim", sel
    assert sel["fidelity"] == POL.ELABORATED_RTL
    assert "verilator" not in sel["passed_over"], "the policy stops at the first available engine"


def test_the_derived_home_registers_the_engine_with_no_env_var(monkeypatch, G, tmp_path,
                                                              toolchain_present):
    """Installing the binary IS registering the engine.

    This replaces an older test that asserted the env var was the ONLY thing that could change the
    answer. That property was not a safeguard, it was the defect: a built, working GSIM model on the
    machine resolved to unavailable unless someone had exported a variable by hand, so the cost decision
    the policy exists to make was never reached. The env override still wins (previous test); what
    changed is that it is no longer the only door."""
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import gsim_emulator as GE

    monkeypatch.delenv("MERLIN_GEMMINI_GSIM_EMU", raising=False)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    home = GE.gsim_home("gemmini")
    home.mkdir(parents=True)
    _executable(home / GE.BINARY_NAME)
    assert G.available("gsim") is True
    assert CR.chipyard_l3_selection("gemmini")["engine"] == "gsim"


def test_with_no_build_anywhere_the_tier_falls_back_to_verilator_and_says_why(monkeypatch, G, tmp_path,
                                                                             toolchain_present):
    """Falling back is correct; falling back SILENTLY is the defect this whole change is about. The
    selection record must carry gsim as passed-over WITH a reason that names where a build would go."""
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import rtl_engine_policy as POL

    monkeypatch.delenv("MERLIN_GEMMINI_GSIM_EMU", raising=False)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    vsim = _executable(tmp_path / "simulator-harness")   # so the fallback engine genuinely "exists"
    monkeypatch.setattr(G, "verilator_path", lambda: vsim)
    sel = CR.chipyard_l3_selection("gemmini")
    assert sel["engine"] == "verilator"
    assert "gsim" in sel["passed_over"]
    assert POL.describe(sel).startswith("verilator")
    gsim_row = [c for c in sel["considered"] if c["engine"] == "gsim"][0]
    assert gsim_row["available"] is False
    assert "rtl_engines" in gsim_row["reason"], gsim_row["reason"]
