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


# ---------------------------------------------------------------- the console backend wiring ------
# These read the harness SOURCES rather than a built image, because the bug they guard against was a
# wiring bug: a parameter that was accepted and then dropped. Building an ELF proves the code compiles;
# only checking the call proves the fact reached it.

def _harness(name: str) -> str:
    return (merlin_dir() / "runtime/baremetal/spike" / name).read_text()


def test_both_console_backends_implement_the_same_abi():
    """`console_uart.c` and `htif.c` are two backends for one four-symbol ABI, chosen at link time. If
    they drift, an image links against half a console."""
    api = ("htif_putc", "htif_puts", "htif_putd", "htif_exit", "console_init")
    for backend in ("htif.c", "console_uart.c"):
        src = _harness(backend)
        for sym in api:
            assert f"{sym}(" in src, f"{backend} does not define {sym}"


def test_the_uart_backend_refuses_to_build_without_derived_facts():
    """A defaulted console address produces NO output, which is the one failure a remote user cannot
    debug. So the absence of the derived macros must be a compile error, not a fallback."""
    src = _harness("console_uart.c")
    assert "#error" in src
    for macro in ("MERLIN_UART_BASE", "MERLIN_UART_DIV_OFF", "MERLIN_SYS_CLK_HZ",
                  "MERLIN_UART_TX_FULL_BIT"):
        assert f"!defined({macro})" in src, f"{macro} is not guarded"
    # No MMIO literal may be baked in: the whole point is that these come from the target's headers.
    assert "0x10020000" not in src


def test_the_uart_backend_reprograms_the_divisor_after_raising_the_pll():
    """The divisor is relative to a clock the PLL just changed. Skipping the second write leaves the
    console emitting GARBAGE rather than nothing -- which reads as a corrupt program, not a bad UART.
    So the divisor must be written twice: once for the reset clock, once after the switch."""
    src = _harness("console_uart.c")
    assert src.count("UART_REG(MERLIN_UART_DIV_OFF) = baud_div") == 2
    # And the PLL sequence must be gated on a target frequency being asked for, so that "leave the
    # chip on its reset clock" is expressible.
    assert "#ifdef MERLIN_CHIP_FREQ_HZ" in src


def test_the_model_harness_brings_the_console_up_before_printing():
    """Ordering is the whole bug: the vendor SDK's own comment is that a printf before the UART is
    configured hangs the core. `console_init()` must therefore be called before the first output."""
    for unit in ("model_main.c", "vlen_probe.c"):
        src = _harness(unit)
        assert "console_init();" in src, f"{unit} never brings the console up"
        # Scoped to main's body: both units define print HELPERS above main, and where those are
        # declared says nothing about when they run. What matters is the order inside the entry point.
        body = src[src.index("int main("):]
        assert "console_init();" in body, f"{unit} does not bring the console up in main()"
        assert body.index("console_init();") < body.index("htif_put"), \
            f"{unit} prints before console_init()"


def test_the_trap_handler_reports_through_the_console_abi():
    """A trap used to be reported by writing `tohost` directly, which on a board with no host is a
    silent hang. Routed through the console ABI it PRINTS the cause -- the difference between a
    reported fault and an unexplained lockup on hardware we cannot attach to."""
    src = _harness("crt.S")
    assert "call  htif_exit" in src
    assert "la    t1, tohost" not in src


def test_the_probe_passes_the_console_choice_through_to_the_link():
    """The probe is the FIRST thing run on a board, so it above all must speak that board's channel:
    an HTIF probe on silicon hangs on its second character and reports nothing."""
    import inspect

    src = inspect.getsource(vector_probe.build)
    assert "console_uart.c" in src
    assert "sdk_chip" in src and "derive_uart_console" in src


def test_a_probe_log_from_the_silicon_can_contradict_the_descriptor():
    """The one check that can catch an under-declared VLEN.

    Under-declaring is not a performance bug: Zephyr's per-thread save area is a fixed
    `vreg[32][CONFIG_RISCV_VECTOR_MAX_LEN/8]` while `z_riscv_vstate_save` fills it with a length taken
    from the hardware, so a descriptor below the truth overruns the thread struct on every context
    switch. Every simulator gate is blind to it -- spike is handed the VLEN we declared, so configured
    and actual agree there by construction. Only the part can disagree.
    """
    log = ("PROBE hartid 0\nPROBE mstatus_vs 1\nPROBE vlenb 32\nPROBE vlen_bits 256\n"
           "PROBE vlmax_e8 32\nPROBE vlmax_e32 8\nDONE\n")
    assert vector_probe.verify_declared(256, log)["verdict"] == vector_probe.PROBE_AGREES
    bad = vector_probe.verify_declared(128, log)
    assert bad["verdict"] == vector_probe.PROBE_DISAGREES and bad["measured"] == 256
    # An undeclared VLEN is a disagreement too: the fallback is the V minimum, which is a guess.
    assert vector_probe.verify_declared(None, log)["verdict"] == vector_probe.PROBE_DISAGREES


def test_an_unreadable_probe_log_is_unmeasured_not_agreement():
    """The probe prints from every hart that reaches it, so a multi-hart chip returns interleaved
    characters. That is what a real returned log looked like. It must not read as a pass -- a garbled
    log silently confirming the declaration is how the bug survives a second round."""
    garbled = "PRPORBOE E htid 1\nPBEBEisa_a_bitit\nOBE Estatat_v_v1\nDONE\n"
    got = vector_probe.verify_declared(128, garbled)
    assert got["verdict"] == vector_probe.PROBE_UNMEASURED
    assert got["measured"] is None
    # Truncated but well-formed is also unmeasured: no vlenb line means nothing was measured.
    short = "PROBE hartid 0\nPROBE mstatus_vs 0\nPROBE vector_state off\nDONE\n"
    assert vector_probe.verify_declared(256, short)["verdict"] == vector_probe.PROBE_UNMEASURED
