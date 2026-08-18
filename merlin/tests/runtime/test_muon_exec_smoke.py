"""Unit tests for the Verilator executability-smoke signal derivation (:func:`muon._smoke_signals`).

These are pure-logic tests over synthetic RadianceTapeoutSim consoles — no Verilator, no ELF — so they
run in CI. They lock the behavior that matters for an ADVISORY RTL-legality backstop:

  * hitting the ``+max-cycles`` cap (the ``*** FAILED *** (timeout)`` / TestDriver assertion watchdog)
    is a CLEAN bounded stop, NOT a fault — the exact misread the numeric runner's ``*** FAILED ***`` ==
    trap heuristic would make;
  * a genuine ``%Error`` / illegal-instruction IS a fault;
  * MX engagement is detected structurally from a store landing on the derived MMIO ctrl_base;
  * a run that never boots or never issues an instruction is not "legal".
"""
from merlin.runtime.backends.base import get_backend

# The muon reference backend is loaded out-of-tree (``merlin._oot_backends.muon``); reach its ``.muon``
# submodule through the registry (its package re-exports the public API but not the underscore helpers).
muon = get_backend("muon").muon

_BOOT = ("Cyclotron: created sim object with config: [clusters=1 cores=1 warps=8 lanes=16]\n"
         "Cyclotron: loading ELF file: kernel.soc.elf\n"
         "[UART] UART0 is here (stdin/stdout).\n")

# A verbose commit line issuing the shared store to the MX ctrl_base (0x84000) — the MX PE command.
_MX_STORE = ("[ISSUE]     clid=0 cid=0 wid=0 pc=10002098 inst=00000103a0b400a3 tmask=1 rd=0 "
             "rs1=57 rs1.data=[00084000 00000001] rs2=58 rs2.data=[00000000]\n")
_ISSUE = "[ISSUE]     clid=0 cid=0 wid=1 pc=100034e8 inst=0000000452209433 tmask=1 rd=3\n"
# The REAL bounded-stop signature the RadianceTapeoutSim prints on the cycle cap: the harness "(timeout)"
# banner PLUS the verilator ``$stop`` the TestDriver executes (a ``%Error`` that must NOT read as a fault).
_CAP = ("*** FAILED *** (timeout) after 40001 simulation cycles\n"
        "%Error: /path/gen-collateral/TestDriver.v:147: Verilog $stop\n")


def test_cap_hit_is_legal_not_a_fault_and_mx_engaged():
    console = _BOOT + _ISSUE + _MX_STORE + _CAP
    sig = muon._smoke_signals(console, max_cycles=40000, mx_ctrl_base=0x84000)
    assert sig["ran"] and sig["booted"] and sig["progressed"]
    assert sig["legal"] is True          # bounded cap stop with no fault == legal
    assert sig["fault"] is False         # the TestDriver (timeout) watchdog is NOT a fault
    assert sig["cycles_capped"] is True
    assert sig["cycles"] == 40001
    assert sig["mx_engaged"] is True


def test_finished_execution_is_legal_non_mx_unknown():
    console = _BOOT + _ISSUE + "Muon [cluster 0 core 0] finished execution.\nCycles: 1234\n"
    sig = muon._smoke_signals(console, max_cycles=40000, mx_ctrl_base=None)
    assert sig["legal"] is True and sig["finished"] is True
    assert sig["cycles_capped"] is False
    assert sig["mx_engaged"] is None     # no MX fact supplied -> honestly unknown, never fabricated
    assert sig["cycles"] == 1234


def test_genuine_fault_is_illegal():
    console = _BOOT + _ISSUE + "%Error: illegal instruction at pc=0x10002000\n"
    sig = muon._smoke_signals(console, max_cycles=40000, mx_ctrl_base=0x84000)
    assert sig["fault"] is True
    assert sig["legal"] is False
    assert "fault" in sig["reason"].lower()


def test_no_boot_is_not_legal():
    sig = muon._smoke_signals("garbage with no markers\n", max_cycles=40000, mx_ctrl_base=None)
    assert sig["ran"] is False and sig["legal"] is False


def test_booted_but_no_progress_is_not_legal():
    sig = muon._smoke_signals(_BOOT + _CAP, max_cycles=40000, mx_ctrl_base=None)
    assert sig["booted"] is True
    assert sig["progressed"] is False    # no [ISSUE]/DASM -> cores never fetched the kernel
    assert sig["legal"] is False


def test_mx_ctrl_base_not_touched_is_false_not_none():
    # ctrl_base supplied but no store to it -> False (checked and absent), distinct from None (not checked)
    sig = muon._smoke_signals(_BOOT + _ISSUE + _CAP, max_cycles=40000, mx_ctrl_base=0x84000)
    assert sig["mx_engaged"] is False
