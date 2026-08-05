"""The probe that makes a board report its own vector width.

Every VLEN claim about a board we cannot reach is otherwise an inference: the Kodiak board files
declare no width (`riscv,isa = "rv64gc"`) and its samples' CONFIG_RISCV_VECTOR_MAX_LEN sizes Zephyr's
save area, bounding the truth only from above. Building for the wrong width is the documented K1 trap.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.runtime import vector_probe


def test_the_probe_source_ships_with_the_harness():
    src = merlin_dir().parent / "merlin/runtime/baremetal/spike/vlen_probe.c"
    if not src.is_file():                      # installed layout keeps it under _data
        src = merlin_dir() / "python/merlin/_data/runtime/baremetal/spike/vlen_probe.c"
    assert src.is_file(), "the probe C source must ship with the bare-metal harness"
    text = src.read_text()
    for csr in ("mhartid", "misa", "mstatus", "vlenb"):
        assert csr in text, f"the probe must read {csr}"
    assert "vsetvli" in text, "VLEN must be derived a second way, so a wrong vlenb cannot pass"
    # Reading vlenb with vector state off traps; on someone else's bench a trap reads as a hang.
    assert "would trap" in text and "PROBE vector_state off" in text


def test_parse_reads_a_console_and_cross_checks_the_width():
    good = ("PROBE hartid 0\nPROBE misa_ext_bits 3412269\nPROBE misa_v_bit 1\n"
            "PROBE mstatus_vs 1\nPROBE vlenb 32\nPROBE vlen_bits 256\n"
            "PROBE vlmax_e8 32\nPROBE vlmax_e32 8\nDONE\n")
    r = vector_probe.parse(good)
    assert r["vlen_bits"] == 256 and r["consistent"] and r["complete"]

    # A vlenb that disagrees with the vsetvli derivation is reported, never averaged away.
    bad = good.replace("PROBE vlmax_e8 32", "PROBE vlmax_e8 16")
    assert vector_probe.parse(bad)["consistent"] is False

    # A run that never reached DONE is not a result.
    assert vector_probe.parse(good.replace("DONE\n", ""))["complete"] is False

    # Vector state off: no width, and that must not read as "128".
    off = "PROBE hartid 0\nPROBE mstatus_vs 0\nPROBE vector_state off\nDONE\n"
    r = vector_probe.parse(off)
    assert r["vlen_bits"] is None and r["consistent"] is False and r["mstatus_vs"] == 0


@pytest.mark.skipif(not __import__("merlin.runtime.backends.spike_model",
                                   fromlist=["x"]).available()
                    if hasattr(__import__("merlin.runtime.backends.spike_model",
                                          fromlist=["x"]), "available") else False,
                    reason="spike toolchain unavailable")
def test_the_probe_reports_the_hardwares_width_not_the_one_it_was_built_for(tmp_path):
    """The independence that makes the probe worth shipping: built for 128, it still reports 256."""
    elf = vector_probe.build(tmp_path, vlen=128)
    for width in (128, 256):
        rep = vector_probe.parse(vector_probe.run_on_spike(elf, vlen=width))
        assert rep["complete"], f"probe did not finish at VLEN {width}"
        assert rep["vlen_bits"] == width, f"probe reported {rep['vlen_bits']} at VLEN {width}"
        assert rep["consistent"]
