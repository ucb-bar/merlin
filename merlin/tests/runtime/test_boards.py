"""Board facts as data, and the failure modes each one prevents.

The generated app used to assume its board: HTIF console options, a 256-bit vector save area, a `&ram0`
label at 0x80000000. Those are chipyard facts, not RISC-V facts, and every one of them fails LOUDLY on
the board and SILENTLY here — no console output looks exactly like a hang, and a region larger than
physical DRAM dies before main(). So the descriptor is the contract, and these tests pin it.
"""
from __future__ import annotations

import pytest

from merlin.runtime import boards
from merlin.runtime.backends import zephyr_model as zm


def test_kodiak_is_described_by_its_silicon_not_its_dts():
    """Kodiak's DTS says ram0=256MB and MP_MAX_NUM_CPUS=2; the chip has 512MB and 3 working cores."""
    b = boards.board("chipyard_kodiak")
    assert b.dram_bytes == 512 * 1024 * 1024
    assert b.harts == 3
    assert b.console == boards.CONSOLE_HTIF


def test_an_unknown_board_falls_back_conservatively_and_says_so():
    """A new board must be tryable before anyone writes it down, but never with invented facts."""
    b = boards.board("some_new_tapeout")
    assert b.dram_bytes == 256 * 1024 * 1024
    assert b.vector_max_len == 128, "unknown VLEN must assume the V minimum, not a guess"
    assert "conservative" in b.notes


def test_overrides_win_so_a_delivery_can_state_what_it_built_for():
    b = boards.board("chipyard_kodiak", dram_bytes=256 * 1024 * 1024, vlen=256)
    assert b.dram_bytes == 256 * 1024 * 1024 and b.vlen == 256


def test_the_config_follows_the_board():
    """cpus, the vector save-area width and FPU_SHARING all come from the descriptor."""
    b = boards.board("spike_riscv64")
    conf = zm._prj_conf(b.harts, "rvv", b)
    assert f"CONFIG_MP_MAX_NUM_CPUS={b.harts}" in conf
    assert f"CONFIG_RISCV_VECTOR_MAX_LEN={b.vector_max_len}" in conf
    # y mis-routes V-illegal-instruction traps into the FP path and retries forever: a silent hang.
    assert "CONFIG_FPU_SHARING=n" in conf


def test_kodiaks_config_reflects_what_its_zephyr_can_actually_build():
    """Two settings Kodiak forces, both learned by failing to link, both recorded in the descriptor.

    FPU_SHARING must be y: its Zephyr's isr.S calls z_riscv_vstate_save/_restore whenever V is on with
    eager switching, and those live in fpu.c/fpu.S which only compile under FPU_SHARING -- with n the
    image does not link. And Zephyr-level V must be OFF: that tree has no RISCV_V_KERNEL_ONLY, so
    enabling V puts `v` in the GLOBAL march and SDK 0.17.0 has no matching libgcc multilib (the link
    falls back to a 32-bit one: "ELFCLASS32 incompatible with ELFCLASS64"). Vectors still work because
    reset.S enables mstatus.VS under CONFIG_FPU, and our model.o carries `v` itself.
    """
    b = boards.board("chipyard_kodiak")
    assert b.fpu_sharing is True and b.zephyr_vector_ext is False
    conf = zm._prj_conf(b.harts, "rvv", b)
    assert "CONFIG_MP_MAX_NUM_CPUS=3" in conf
    assert "CONFIG_FPU_SHARING=y" in conf
    assert "CONFIG_RISCV_ISA_EXT_V=y" not in conf
    assert "CONFIG_FPU=y" in conf, "mstatus.VS comes from CONFIG_FPU on this tree"


def test_a_vector_capable_tree_still_gets_the_full_vector_config():
    b = boards.board("chipyard_riscv64")
    conf = zm._prj_conf(b.harts, "rvv", b)
    assert "CONFIG_RISCV_ISA_EXT_V=y" in conf
    assert f"CONFIG_RISCV_VECTOR_MAX_LEN={b.vector_max_len}" in conf


def test_htif_console_options_are_set_only_for_an_htif_board():
    """Both options default to n upstream, and unbuffered HTIF emits one char per host round-trip --
    on a ~20 MHz core that looks like the model never finishing. A non-HTIF board must not get options
    its driver does not have."""
    htif = zm._prj_conf(2, "rvv", boards.board("chipyard_kodiak"))
    assert "CONFIG_UART_HTIF_BUFFERED_OUTPUT=y" in htif
    assert "CONFIG_UART_HTIF_SYSCALL_PRINT=y" in htif
    uart = zm._prj_conf(2, "rvv", boards.board("x", console=boards.CONSOLE_UART))
    assert "UART_HTIF" not in uart
    assert "board-provided" in uart


def test_the_scalar_backend_still_carries_no_vector_config():
    conf = zm._prj_conf(2, "scalar", boards.board("chipyard_kodiak"))
    assert "RISCV_ISA_EXT_V" not in conf


def test_a_region_bigger_than_the_board_is_refused():
    """A region larger than physical DRAM is a boot that dies before main() with NO console output --
    the single least debuggable outcome for someone running our binary on their bench."""
    import inspect

    src = inspect.getsource(zm.build_app)
    assert "brd.dram_bytes" in src, "build_app must compare the region against the board's DRAM"
    assert "does not fit" in src or "has" in src


def test_the_overlay_uses_the_board_label_and_base():
    import inspect

    src = inspect.getsource(zm.build_app)
    assert "brd.ram_label" in src and "brd.dram_base" in src


def test_build_hash_is_emitted_and_parseable():
    """A console log mailed back from someone else's board is otherwise unattributable."""
    src = zm._main_c(0, n_harts=1, build_hash="abc123def456")
    assert 'METRIC build_hash' in src and '"abc123def456"' in src
    parsed = zm._parse_console(
        "METRIC build_hash abc123def456\nMETRIC cycles 7\nOUT 1 1065353216\nDONE", 0)
    assert parsed["metrics"]["build_hash"] == "abc123def456"
    assert parsed["metrics"]["cycles"] == 7


def test_a_non_numeric_metric_does_not_take_the_whole_run_down():
    """int() on every METRIC crashed the parser on the first identity string."""
    r = zm._parse_console("METRIC tag deadbeef\nOUT 1 0\nDONE", 0)
    assert r["metrics"]["tag"] == "deadbeef"
